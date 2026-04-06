"""
OpenVLA eval script for G1 robot with Inspire FTP gripper.

OpenVLA outputs 7-DOF delta end-effector actions:
    [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
for the LEFT arm. Right arm stays at current position.

Usage (robot Jetson NX):
    conda activate unitree_lerobot
    cd /home/unitree/AlphaZ_WS/unitree_lerobot

    # Standard OpenVLA:
    python unitree_lerobot/eval_robot/eval_openvla.py \\
        --server-host 100.96.139.69 --server-port 5555 \\
        --task "pick up the orange bottle and put it in the black box" \\
        --action-scale 4.0 --send-real-robot --motion
"""

import argparse
import base64
import os
import threading
import time
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import zmq

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from teleimager.image_client import ImageClient
from unitree_lerobot.eval_robot.make_robot import setup_robot_interface


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_current_ee_poses(ik_solver, arm_q: np.ndarray, _cache={}):
    """Forward kinematics: returns (L_ee_4x4, R_ee_4x4) from current joint angles."""
    if "data" not in _cache:
        _cache["data"] = pin.Data(ik_solver.reduced_robot.model)
    data = _cache["data"]
    pin.forwardKinematics(ik_solver.reduced_robot.model, data, arm_q)
    pin.updateFramePlacements(ik_solver.reduced_robot.model, data)
    L = data.oMf[ik_solver.L_hand_id].homogeneous.copy()
    R = data.oMf[ik_solver.R_hand_id].homogeneous.copy()
    return L, R


def apply_delta_ee(current_4x4: np.ndarray, delta_xyz: np.ndarray, delta_rpy: np.ndarray) -> np.ndarray:
    """Apply Cartesian delta [dx,dy,dz] and RPY delta [droll,dpitch,dyaw] to a 4x4 SE3 pose."""
    rot_delta = pin.rpy.rpyToMatrix(float(delta_rpy[0]), float(delta_rpy[1]), float(delta_rpy[2]))
    target = current_4x4.copy()
    target[:3, 3] += delta_xyz
    target[:3, :3] = rot_delta @ current_4x4[:3, :3]
    return target


def simulate_chunk_trajectory(L_ee_start: np.ndarray, chunk: np.ndarray, action_scale: float) -> np.ndarray:
    """Forward-simulate a chunk of delta actions from L_ee_start.
    Returns array of shape (k+1, 3) — start position + one position per action step.
    """
    poses = [L_ee_start[:3, 3].copy()]
    pose = L_ee_start.copy()
    for action in chunk:
        physical = action.copy()
        physical[:6] *= action_scale
        pose = apply_delta_ee(pose, physical[:3], physical[3:6])
        poses.append(pose[:3, 3].copy())
    return np.array(poses)


def save_trajectory_plot(planned_trajs: list, actual_positions: list, out_path: str):
    """Save a 3D plot of all VLA-planned trajectories and the actual executed path."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("cool")
    n = len(planned_trajs)
    for i, traj in enumerate(planned_trajs):
        color = cmap(i / max(n - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.4, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=10)

    if actual_positions:
        actual = np.array(actual_positions)
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "r-", linewidth=2, label="executed")
        ax.scatter(*actual[0], color="green", s=50, zorder=5, label="start")
        ax.scatter(*actual[-1], color="red", s=50, zorder=5, label="end")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"VLA planned trajectories (n={n})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_jpeg(img_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy image as JPEG base64 string for ZMQ transport."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def get_policy_frame(img_client: ImageClient) -> tuple[np.ndarray | None, str | None]:
    """Get frame for policy inference from head camera."""
    frame, _ = img_client.get_head_frame()
    return frame, "head" if frame is not None else None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA robot eval (robot side)")
    p.add_argument("--server-host", default="192.168.123.162")
    p.add_argument("--server-port", type=int, default=5555)
    p.add_argument("--task", default="pick up the bottle")
    p.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    p.add_argument("--ee", default="inspire_ftp")
    p.add_argument("--action-scale", type=float, default=None,
                   help="Multiply xyz+rpy deltas by this factor. Defaults to training-fps/eval-fps "
                        "to compensate for frequency mismatch.")
    p.add_argument("--training-fps", type=float, default=10.0,
                   help="FPS the model was trained at (default 10). Used to compute default action-scale.")
    p.add_argument("--frequency", type=float, default=5.0,
                   help="[Standard] Control frequency in Hz")
    p.add_argument("--send-real-robot", action="store_true")
    p.add_argument("--motion", action="store_true")
    p.add_argument("--img-host", default="192.168.123.164")
    p.add_argument("--img-request-port", type=int, default=60000,
                   help="Port for teleimager config requester (default: 60000)")
    p.add_argument("--video-dir", default="/tmp",
                   help="Directory to save the eval video (default: /tmp)")
    args = p.parse_args()
    args.sim = not args.send_real_robot
    return args


# ---------------------------------------------------------------------------
# Shared robot setup + home pose
# ---------------------------------------------------------------------------

_HOME_Q = np.array([
    # left arm (7 DOF) — pick_up_bottle episode_0001 frame 120 (arm extended forward, ee_x=0.328m)
    -0.4393416941165924,
     0.09620936214923859,
     0.05770404636859894,
     0.6954206228256226,
    -0.09713214635848999,
    -0.4131801426410675,
    -0.3374398350715637,
    # right arm (7 DOF)
    -0.28895166516304016,
    -0.01605886220932007,
    -0.2118811011314392,
     1.3727091550827026,
    -0.08853945881128311,
    -0.05060938373208046,
     0.04924318194389343,
])



def apply_action(action, arm_ik, arm_ctrl, ee_shared_mem, action_scale):
    """FK → apply delta → IK → send to robot + gripper.

    The server (openvla_server.py) fully denormalizes actions to physical space
    using BOUNDS_Q99 stats before sending. action_scale compensates for the
    training-fps / eval-fps mismatch (e.g. trained at 10Hz, eval at 5Hz → scale=2).
    Gripper (dim 6) is NOT scaled — it is already in [0, 0.5] physical range.
    """
    physical = action.copy()
    physical[:6] *= action_scale

    current_arm_q = arm_ctrl.get_current_dual_arm_q()
    L_ee, R_ee = get_current_ee_poses(arm_ik, current_arm_q)
    target_L = apply_delta_ee(L_ee, physical[:3], physical[3:6])
    sol_q, sol_tau = arm_ik.solve_ik(target_L, R_ee, current_arm_q)
    sol_q[7:] = current_arm_q[7:]
    sol_tau[7:] = 0.0
    arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)
    # Training gripper range 0–0.5, controller expects 0–1
    gripper_val = float(np.clip(physical[6] * 2.0, 0.0, 1.0))
    if gripper_val < 0.7:
        gripper_val = 0.0
    if ee_shared_mem:
        left_mem = ee_shared_mem.get("left")
        if left_mem is not None and hasattr(left_mem, "value"):
            left_mem.value = gripper_val
    return L_ee, target_L, gripper_val


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    video_writer_head = None
    video_stop = threading.Event()
    img_client = None
    planned_trajs: list = []
    actual_positions: list = []
    try:
        # -- Camera ----------------------------------------------------------
        img_client = ImageClient(host=args.img_host, request_port=args.img_request_port)
        cam_config = img_client.get_cam_config()
        logger_mp.info(f"Image client connected to {args.img_host}:{args.img_request_port}")
        logger_mp.info(f"Camera config: {list(cam_config.keys())}")

        # -- Robot -----------------------------------------------------------
        robot_interface = setup_robot_interface(args)
        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik   = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]

        # -- Move to home position --------------------------------------------
        current_q = arm_ctrl.get_current_dual_arm_q()
        logger_mp.info(f"Moving arm to home position: {np.round(_HOME_Q, 3)}")
        for i in range(1, 201):
            interp_q = current_q + (_HOME_Q - current_q) * (i / 200)
            arm_ctrl.ctrl_dual_arm(interp_q, np.zeros(len(interp_q)))
            time.sleep(0.01)
        logger_mp.info("Arm ready.")

        # -- Open gripper at start -------------------------------------------
        if ee_shared_mem:
            left_mem = ee_shared_mem.get("left")
            if left_mem is not None and hasattr(left_mem, "value"):
                left_mem.value = 1.0
                logger_mp.info("Gripper opened.")

        # -- User confirm ----------------------------------------------------
        if input("Enter 's' to start evaluation: ").strip().lower() != "s":
            logger_mp.info("Aborted.")
            return

        action_scale = args.action_scale if args.action_scale is not None else args.training_fps / args.frequency
        logger_mp.info(f"action_scale={action_scale:.2f} (training {args.training_fps}Hz / eval {args.frequency}Hz)")
        # gripper_filter = GripperHysteresis()

        time.sleep(0.5)
        frame0, frame0_src = None, None
        warmup_deadline = time.time() + 3.0
        while time.time() < warmup_deadline and frame0 is None:
            frame0, frame0_src = get_policy_frame(img_client)
            if frame0 is None:
                time.sleep(0.02)
        if frame0 is not None:
            cv2.imwrite("/tmp/eval_openvla_frame0.jpg", frame0.copy())
            logger_mp.info(f"Saved first frame ({frame0_src}) to /tmp/eval_openvla_frame0.jpg")
        else:
            logger_mp.warning("Policy frame not ready after warmup; skipped saving /tmp/eval_openvla_frame0.jpg")

        # -- Video recording -------------------------------------------------
        os.makedirs(args.video_dir, exist_ok=True)
        task_slug = args.task.replace(" ", "_")[:40]
        video_path = os.path.join(
            args.video_dir,
            f"eval_openvla_{time.strftime('%Y%m%d_%H%M%S')}_{task_slug}.mp4",
        )
        video_writer_head = cv2.VideoWriter(
            video_path.replace(".mp4", "_head.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            15,
            (640, 480),
        )
        def _record_frames():
            interval = 1.0 / 15
            while not video_stop.is_set():
                t = time.perf_counter()
                head, _ = img_client.get_head_frame()
                if head is not None:
                    video_writer_head.write(head)
                elapsed = time.perf_counter() - t
                time.sleep(max(0.0, interval - elapsed))

        threading.Thread(target=_record_frames, daemon=True).start()
        logger_mp.info(f"Recording video to {video_path}")

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect(f"tcp://{args.server_host}:{args.server_port}")
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        logger_mp.info(f"scale={action_scale} freq={args.frequency}Hz | task: '{args.task}'")

        step = 0
        last_no_frame_warn_t = 0.0
        while True:
            t0 = time.perf_counter()

            policy_img, policy_src = get_policy_frame(img_client)
            if policy_img is None:
                now = time.time()
                if now - last_no_frame_warn_t > 1.0:
                    logger_mp.warning("Policy frame not ready — retrying.")
                    last_no_frame_warn_t = now
                time.sleep(0.02)
                continue
            img_b64 = encode_image_jpeg(policy_img)
            try:
                socket.send_json({"image": img_b64, "task": args.task})
                resp = socket.recv_json()
            except zmq.Again:
                logger_mp.warning("Server timeout — skipping step.")
                continue

            if resp.get("status") != "ok":
                logger_mp.warning(f"Server error: {resp.get('status')}")
                continue

            if "actions" in resp:
                chunk = np.array(resp["actions"], dtype=np.float64)
                action = chunk[0]
            elif "action" in resp:
                action = np.array(resp["action"], dtype=np.float64)
                chunk = action[np.newaxis]
            else:
                logger_mp.warning("Server response missing action — skipping step.")
                continue

            L_ee, target_L, gripper_val = apply_action(
                action, arm_ik, arm_ctrl, ee_shared_mem, action_scale
            )
            actual_positions.append(L_ee[:3, 3].copy())
            planned_trajs.append(simulate_chunk_trajectory(L_ee, chunk, action_scale))

            logger_mp.info(
                f"[step {step}] cam={policy_src} dz={action[2]:.4f} gripper={gripper_val:.2f} | "
                f"L_pos {np.round(L_ee[:3,3],3)} -> {np.round(target_L[:3,3],3)} | "
                f"ik={int((time.perf_counter()-t0)*1000)}ms"
            )
            step += 1

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, (1.0 / args.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        video_stop.set()
        if video_writer_head is not None:
            video_writer_head.release()
            logger_mp.info(f"Head video saved to {video_path.replace('.mp4', '_head.mp4')}")
        if planned_trajs:
            traj_plot_path = video_path.replace(".mp4", "_trajectories.png")
            save_trajectory_plot(planned_trajs, actual_positions, traj_plot_path)
            logger_mp.info(f"Trajectory plot saved to {traj_plot_path}")
        if img_client is not None:
            img_client.close()
        logger_mp.info("End of eval.")


if __name__ == "__main__":
    main()

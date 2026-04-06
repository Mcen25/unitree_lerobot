"""
V-GPS eval script for G1 robot with Inspire FTP gripper.

Samples N actions from the OpenVLA server and either:
  - Has the server score and select the best action (server-side V-GPS, recommended),
  - Or scores locally with a Cal-QL value function (client-side V-GPS, requires JAX).

Usage (robot Jetson NX):
    conda activate unitree_lerobot
    cd /home/unitree/AlphaZ_WS/unitree_lerobot

    # V-GPS with server-side Cal-QL scoring (no JAX needed on Jetson):
    python unitree_lerobot/eval_robot/eval_g1_v_gps.py \\
        --server-host 100.96.139.69 --server-port 5555 \\
        --task "pick up the orange bottle and put it in the black box" \\
        --use-vgps --num-samples 10 \\
        --action-scale 4.0 --send-real-robot --motion

    # V-GPS with client-side Cal-QL scoring (requires JAX CUDA on Jetson):
    python unitree_lerobot/eval_robot/eval_g1_v_gps.py \\
        --server-host 100.96.139.69 --server-port 5555 \\
        --task "pick up the orange bottle and put it in the black box" \\
        --use-vgps \\
        --vgps-checkpoint /home/unitree/AlphaZ_WS/V-GPS/checkpoints/checkpoint_500000 \\
        --num-samples 10 --action-temp 1.0 \\
        --action-scale 4.0 --send-real-robot --motion

    # Without V-GPS (standard OpenVLA):
    python unitree_lerobot/eval_robot/eval_g1_v_gps.py \\
        --server-host 100.96.139.69 --server-port 5555 \\
        --task "pick up the orange bottle and put it in the black box" \\
        --action-scale 4.0 --send-real-robot --motion
"""

import argparse
import base64
import os
import sys
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

_VGPS_ROOT = "/home/unitree/AlphaZ_WS/V-GPS"
_OPENVLA_ROOT = "/home/unitree/AlphaZ_WS/openvla"
_VGPS_PRETRAINED_CONFIG = os.path.join(_VGPS_ROOT, "experiments/configs/pretrained_checkpoint.yaml")


def get_current_ee_poses(ik_solver, arm_q: np.ndarray, _cache={}):
    if "data" not in _cache:
        _cache["data"] = pin.Data(ik_solver.reduced_robot.model)
    data = _cache["data"]
    pin.forwardKinematics(ik_solver.reduced_robot.model, data, arm_q)
    pin.updateFramePlacements(ik_solver.reduced_robot.model, data)
    L = data.oMf[ik_solver.L_hand_id].homogeneous.copy()
    R = data.oMf[ik_solver.R_hand_id].homogeneous.copy()
    return L, R


def apply_delta_ee(current_4x4: np.ndarray, delta_xyz: np.ndarray, delta_rpy: np.ndarray) -> np.ndarray:
    rot_delta = pin.rpy.rpyToMatrix(float(delta_rpy[0]), float(delta_rpy[1]), float(delta_rpy[2]))
    target = current_4x4.copy()
    target[:3, 3] += delta_xyz
    target[:3, :3] = rot_delta @ current_4x4[:3, :3]
    return target


def simulate_chunk_trajectory(L_ee_start: np.ndarray, chunk: np.ndarray, action_scale: float) -> np.ndarray:
    """Forward-simulate a chunk of delta actions. Returns (k+1, 3) xyz positions."""
    poses = [L_ee_start[:3, 3].copy()]
    pose = L_ee_start.copy()
    for action in chunk:
        physical = action.copy()
        physical[:6] *= action_scale
        pose = apply_delta_ee(pose, physical[:3], physical[3:6])
        poses.append(pose[:3, 3].copy())
    return np.array(poses)


def save_trajectory_plot(
    steps: list,          # list of (trajs, selected_idx) per step; selected_idx=-1 for no V-GPS
    actual_positions: list,
    out_path: str,
):
    """Save a 3D plot of VLA-planned trajectories.

    Each entry in `steps` is (np.ndarray of shape (N, k+1, 3), selected_idx).
    Non-selected candidates are drawn in gray; the selected one in the step's color.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("cool")
    n_steps = len(steps)
    for i, (trajs, sel_idx) in enumerate(steps):
        color = cmap(i / max(n_steps - 1, 1))
        for j, traj in enumerate(trajs):
            if sel_idx >= 0 and j != sel_idx:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="gray", alpha=0.15, linewidth=0.8)
            else:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7, linewidth=1.5)

    if actual_positions:
        actual = np.array(actual_positions)
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "r-", linewidth=2, label="executed")
        ax.scatter(*actual[0], color="green", s=50, zorder=5, label="start")
        ax.scatter(*actual[-1], color="red", s=50, zorder=5, label="end")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"VLA planned trajectories ({n_steps} steps)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def encode_image_jpeg(img_bgr: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# Bridge dataset bounds for normalising actions to [-1, 1] for Cal-QL critic
_BRIDGE_ACT_MIN = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0])
_BRIDGE_ACT_MAX = np.array([ 0.05,  0.05,  0.05,  0.25,  0.25,  0.25, 1.0])


def rescale_actions(actions: np.ndarray, safety_margin: float = 1e-5) -> np.ndarray:
    scaled = (actions - _BRIDGE_ACT_MIN) / (_BRIDGE_ACT_MAX - _BRIDGE_ACT_MIN) * 2 - 1
    return np.clip(scaled, -1 + safety_margin, 1 - safety_margin)


def load_vgps_checkpoint(path: str, wandb_run_name: str = ""):
    assert os.path.exists(path), f"V-GPS checkpoint not found: {path}"

    for p in [_VGPS_ROOT, _OPENVLA_ROOT]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import jax
    import yaml
    from flax.training import checkpoints
    from jaxrl_m.agents import agents
    from jaxrl_m.vision import encoders
    from jaxrl_m.data.text_processing import text_processors

    os.environ.setdefault("TFHUB_CACHE_DIR", "/tmp/tfhub")

    if wandb_run_name == "":
        with open(_VGPS_PRETRAINED_CONFIG, "r") as f:
            config = yaml.safe_load(f)
    else:
        import wandb
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
    example_batch = {
        "observations": {"image": np.zeros((1, 256, 256, 3), dtype=np.uint8)},
        "goals":        {"language": np.zeros((1, 512), dtype=np.float32)},
        "actions":      np.zeros((1, 7), dtype=np.float32),
    }
    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        encoder_def=encoder_def,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        **config["agent_kwargs"],
    )
    critic_text_processor = text_processors[config["text_processor"]]()
    agent = checkpoints.restore_checkpoint(path, agent)

    def get_values(observations, goals, actions):
        return agent.get_q_values(observations, goals, actions)

    logger_mp.info(f"[V-GPS] Checkpoint loaded from {path}")
    return get_values, critic_text_processor


def parse_args():
    p = argparse.ArgumentParser(description="V-GPS robot eval (robot side)")
    p.add_argument("--server-host", default="192.168.123.162")
    p.add_argument("--server-port", type=int, default=5555)
    p.add_argument("--task", default="pick up the bottle")
    p.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    p.add_argument("--ee", default="inspire_ftp")
    p.add_argument("--action-scale", type=float, default=None,
                   help="Multiply xyz+rpy deltas by this factor. Defaults to training-fps/eval-fps.")
    p.add_argument("--training-fps", type=float, default=10.0,
                   help="FPS the model was trained at (default 10).")
    p.add_argument("--use-vgps", action="store_true",
                   help="Enable V-GPS: sample N actions and score with Cal-QL value function")
    p.add_argument("--vgps-checkpoint", default="",
                   help="Path to Cal-QL checkpoint for client-side scoring. "
                        "Leave empty to use server-side scoring (requires --vgps-checkpoint on server).")
    p.add_argument("--vgps-wandb", default="",
                   help="Wandb run name for checkpoint config (or '' to use pretrained_checkpoint.yaml)")
    p.add_argument("--num-samples", type=int, default=10,
                   help="[V-GPS] Number of actions to sample per step (default 10)")
    p.add_argument("--action-temp", type=float, default=1.0,
                   help="[V-GPS] Boltzmann temperature for action selection (0=argmax, default 1.0)")
    p.add_argument("--sample-temperature", type=float, default=1.5,
                   help="[V-GPS] OpenVLA sampling temperature for diversity (default 1.5)")
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


_HOME_Q = np.array([
    -0.3494240641593933,
     0.06934072822332382,
     0.08838365972042084,
     0.9953857660293579,
     0.014213290996849537,
    -0.8925731182098389,
    -0.18670225143432617,
    -0.2933139204978943, -0.1350741982460022, -0.20893298089504242,
     1.3649553060531616, -0.07726229727268219,  0.029828736558556557,
     0.052934322506189346,
])


def apply_action(action, arm_ik, arm_ctrl, ee_shared_mem, action_scale):
    """Gripper values below 0.9 (normalized) are snapped to 0.0 for firm grasp."""
    physical = action.copy()
    physical[:6] *= action_scale

    current_arm_q = arm_ctrl.get_current_dual_arm_q()
    L_ee, R_ee = get_current_ee_poses(arm_ik, current_arm_q)
    target_L = apply_delta_ee(L_ee, physical[:3], physical[3:6])
    sol_q, sol_tau = arm_ik.solve_ik(target_L, R_ee, current_arm_q)
    sol_q[7:] = current_arm_q[7:]
    sol_tau[7:] = 0.0
    arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)
    gripper_val = float(np.clip(physical[6] * 2.0, 0.0, 1.0))
    if gripper_val < 0.7:
        gripper_val = 0.0
    if ee_shared_mem:
        left_mem = ee_shared_mem.get("left")
        if left_mem is not None and hasattr(left_mem, "value"):
            left_mem.value = gripper_val
    return L_ee, target_L, gripper_val


def main():
    args = parse_args()

    video_writer_head = None
    video_stop = threading.Event()
    img_client = None
    traj_steps: list = []
    actual_positions: list = []
    try:
        img_client = ImageClient(host=args.img_host, request_port=args.img_request_port)
        cam_config = img_client.get_cam_config()
        logger_mp.info(f"Image client connected to {args.img_host}:{args.img_request_port}")
        logger_mp.info(f"Camera config: {list(cam_config.keys())}")

        robot_interface = setup_robot_interface(args)
        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik   = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]

        current_q = arm_ctrl.get_current_dual_arm_q()
        logger_mp.info(f"Moving arm to home position: {np.round(_HOME_Q, 3)}")
        for i in range(1, 201):
            interp_q = current_q + (_HOME_Q - current_q) * (i / 200)
            arm_ctrl.ctrl_dual_arm(interp_q, np.zeros(len(interp_q)))
            time.sleep(0.01)
        logger_mp.info("Arm ready.")

        if ee_shared_mem:
            left_mem = ee_shared_mem.get("left")
            if left_mem is not None and hasattr(left_mem, "value"):
                left_mem.value = 1.0
                logger_mp.info("Gripper opened.")

        if input("Enter 's' to start evaluation: ").strip().lower() != "s":
            logger_mp.info("Aborted.")
            return

        action_scale = args.action_scale if args.action_scale is not None else args.training_fps / args.frequency
        logger_mp.info(f"action_scale={action_scale:.2f} (training {args.training_fps}Hz / eval {args.frequency}Hz)")

        get_values = None
        critic_text_processor = None
        prompt_embeds_cache = None
        vgps_client_side = False
        if args.use_vgps:
            if args.vgps_checkpoint:
                # Client-side Cal-QL scoring (requires JAX on Jetson)
                get_values, critic_text_processor = load_vgps_checkpoint(
                    args.vgps_checkpoint, args.vgps_wandb
                )
                _prompt_embed = critic_text_processor.encode(args.task)
                prompt_embeds_cache = np.repeat(_prompt_embed, args.num_samples, axis=0)
                vgps_client_side = True
                logger_mp.info(
                    f"[V-GPS] Client-side scoring | num_samples={args.num_samples} "
                    f"action_temp={args.action_temp} sample_temp={args.sample_temperature}"
                )
            else:
                # Server-side scoring — server must be started with --vgps-checkpoint
                logger_mp.info(
                    f"[V-GPS] Server-side scoring | num_samples={args.num_samples} "
                    f"sample_temp={args.sample_temperature}"
                )

        time.sleep(1.0)
        _frame0, _ = img_client.get_head_frame()
        if _frame0 is not None:
            cv2.imwrite("/tmp/eval_vgps_frame0.jpg", _frame0)
            logger_mp.info("Saved first frame to /tmp/eval_vgps_frame0.jpg")

        os.makedirs(args.video_dir, exist_ok=True)
        task_slug = args.task.replace(" ", "_")[:40]
        video_path = os.path.join(
            args.video_dir,
            f"eval_vgps_{time.strftime('%Y%m%d_%H%M%S')}_{task_slug}.mp4",
        )
        video_writer_head = cv2.VideoWriter(
            video_path.replace(".mp4", "_head.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480),
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

        if vgps_client_side:
            import jax
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect(f"tcp://{args.server_host}:{args.server_port}")
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        logger_mp.info(
            f"[{'V-GPS' if args.use_vgps else 'Standard'}] "
            f"scale={action_scale} freq={args.frequency}Hz | task: '{args.task}'"
        )

        step = 0
        while True:
            t0 = time.perf_counter()

            head_img, _ = img_client.get_head_frame()
            if head_img is None:
                logger_mp.warning("Head frame not ready — skipping step.")
                continue
            img_b64 = encode_image_jpeg(head_img)

            if args.use_vgps:
                try:
                    socket.send_json({
                        "image": img_b64,
                        "task": args.task,
                        "num_samples": args.num_samples,
                        "sample_temperature": args.sample_temperature,
                    })
                    resp = socket.recv_json()
                except zmq.Again:
                    logger_mp.warning("Server timeout — skipping step.")
                    continue

                if resp.get("status") != "ok":
                    logger_mp.warning(f"Server error: {resp.get('status')}")
                    continue

                if vgps_client_side:
                    # Client-side scoring: server returns all N candidate actions
                    if "actions" not in resp or resp["actions"] is None:
                        logger_mp.warning("Server response missing actions — skipping step.")
                        continue

                    candidate_actions = np.array(resp["actions"], dtype=np.float64)
                    n = len(candidate_actions)
                    critic_image = cv2.resize(head_img, (256, 256))
                    critic_images = np.repeat(critic_image[None], n, axis=0)
                    critic_actions = rescale_actions(candidate_actions).astype(np.float32)
                    values = np.array(get_values(
                        observations={"image": critic_images},
                        goals={"language": prompt_embeds_cache},
                        actions=critic_actions,
                    ))
                    if args.action_temp > 0:
                        rng = jax.random.PRNGKey(np.random.randint(0, 2**31))
                        idx = int(jax.random.categorical(rng, values / args.action_temp))
                    else:
                        idx = int(np.argmax(values))
                    action = candidate_actions[idx]
                    logger_mp.info(
                        f"[step {step}] V-GPS (client) max={values.max():.2f} min={values.min():.2f} "
                        f"selected={idx} | dz={action[2]:.4f} gripper={action[6]:.2f}"
                    )
                    # Trajectory visualization uses the EE pose before the action is applied.
                    # L_ee is available after apply_action; simulate from current IK state.
                    _current_q = arm_ctrl.get_current_dual_arm_q()
                    _L_ee_now, _ = get_current_ee_poses(arm_ik, _current_q)
                    candidate_trajs = np.array([
                        simulate_chunk_trajectory(_L_ee_now, a[np.newaxis], action_scale)
                        for a in candidate_actions
                    ])
                else:
                    # Server-side scoring: server returns the single best action
                    if "action" not in resp or resp["action"] is None:
                        logger_mp.warning("Server response missing action — skipping step.")
                        continue

                    action = np.array(resp["action"], dtype=np.float64)
                    vgps_max = resp.get("vgps_max", float("nan"))
                    vgps_min = resp.get("vgps_min", float("nan"))
                    logger_mp.info(
                        f"[step {step}] V-GPS (server) max={vgps_max:.2f} min={vgps_min:.2f} "
                        f"| dz={action[2]:.4f} gripper={action[6]:.2f}"
                    )
                    candidate_trajs = None

                L_ee, target_L, gripper_val = apply_action(
                    action, arm_ik, arm_ctrl, ee_shared_mem, action_scale
                )
                actual_positions.append(L_ee[:3, 3].copy())
                if candidate_trajs is not None:
                    traj_steps.append((candidate_trajs, idx))
                else:
                    traj_steps.append((
                        simulate_chunk_trajectory(L_ee, action[np.newaxis], action_scale)[np.newaxis], -1
                    ))

            else:
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
                traj_steps.append((simulate_chunk_trajectory(L_ee, chunk, action_scale)[np.newaxis], -1))

                logger_mp.info(f"[step {step}] dz={action[2]:.4f} gripper={action[6]:.2f}")

            logger_mp.info(
                f"[step {step}] gripper_out={gripper_val:.2f} | "
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
        if traj_steps:
            traj_plot_path = video_path.replace(".mp4", "_trajectories.png")
            save_trajectory_plot(traj_steps, actual_positions, traj_plot_path)
            logger_mp.info(f"Trajectory plot saved to {traj_plot_path}")
        if img_client is not None:
            img_client.close()
        logger_mp.info("End of eval.")


if __name__ == "__main__":
    main()

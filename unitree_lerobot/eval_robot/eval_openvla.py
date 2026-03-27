"""
OpenVLA eval script for G1 robot with Inspire FTP gripper.

Supports both standard OpenVLA (token-based, closed-loop) and OpenVLA-OFT
(action head + chunking + temporal ensembling). Pass --oft to enable OFT mode.

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

    # OpenVLA-OFT:
    python unitree_lerobot/eval_robot/eval_openvla.py \\
        --server-host 100.96.139.69 --server-port 5555 \\
        --task "pick up the orange bottle and put it in the black box" \\
        --oft --chunk-size 8 --ensemble-lambda 0.01 \\
        --action-scale 4.0 --send-real-robot --motion
"""

import argparse
import base64
import os
import threading
import time
import traceback
from multiprocessing import shared_memory

import cv2
import numpy as np
import pinocchio as pin
import zmq

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
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


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_jpeg(img_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode BGR numpy image as JPEG base64 string for ZMQ transport."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# OFT: Temporal ensembling + background inference worker
# ---------------------------------------------------------------------------

def temporal_ensemble(
    buffer: list,
    current_step: int,
    lam: float = 0.01,
) -> np.ndarray | None:
    """Average overlapping chunk predictions weighted by recency (exp(-lam * age))."""
    weighted = np.zeros(7)
    total_w = 0.0
    for chunk, pred_step in buffer:
        idx = current_step - pred_step
        if 0 <= idx < len(chunk):
            w = np.exp(-lam * idx)
            weighted += w * chunk[idx]
            total_w += w
    return weighted / total_w if total_w > 1e-9 else None


def inference_worker(
    server_host: str,
    server_port: int,
    task: str,
    tv_img_array: np.ndarray,
    buffer: list,
    buffer_lock: threading.Lock,
    exec_step_ref: list,
    first_chunk_event: threading.Event,
    stop_event: threading.Event,
    max_buffer: int = 8,
):
    """OFT background inference thread — continuously fetches action chunks."""
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{server_host}:{server_port}")
    socket.setsockopt(zmq.RCVTIMEO, 15000)
    logger_mp.info(f"[inference] Connected to {server_host}:{server_port}")

    while not stop_event.is_set():
        img_b64 = encode_image_jpeg(tv_img_array.copy())
        with buffer_lock:
            pred_step = exec_step_ref[0]
        try:
            socket.send_json({"image": img_b64, "task": task})
            resp = socket.recv_json()
        except zmq.Again:
            logger_mp.warning("[inference] Server timeout — retrying.")
            continue

        if resp.get("status") != "ok":
            logger_mp.warning(f"[inference] Server error: {resp.get('status')}")
            continue

        raw = resp.get("actions") or ([resp["action"]] if resp.get("action") else None)
        if raw is None:
            continue
        chunk = np.array(raw, dtype=np.float64)

        with buffer_lock:
            buffer.append((chunk, pred_step))
            while len(buffer) > max_buffer:
                buffer.pop(0)
            first_chunk_event.set()

    socket.close()
    ctx.term()
    logger_mp.info("[inference] Worker stopped.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA robot eval (robot side)")
    p.add_argument("--server-host", default="192.168.123.162")
    p.add_argument("--server-port", type=int, default=5555)
    p.add_argument("--task", default="pick up the orange bottle and put it in the black box")
    p.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    p.add_argument("--ee", default="inspire_ftp")
    p.add_argument("--action-scale", type=float, default=1.0,
                   help="Multiply xyz+rpy deltas by this factor (e.g. 4.0 to amplify small model outputs)")
    # OFT-specific
    p.add_argument("--oft", action="store_true",
                   help="Enable OFT mode: background inference thread + temporal ensembling")
    p.add_argument("--chunk-size", type=int, default=8,
                   help="[OFT] Action chunk size (match --num-actions-chunk on server)")
    p.add_argument("--ensemble-lambda", type=float, default=0.01,
                   help="[OFT] Temporal ensemble decay rate")
    # Standard mode frequency control
    p.add_argument("--frequency", type=float, default=5.0,
                   help="[Standard] Control frequency in Hz")
    p.add_argument("--send-real-robot", action="store_true")
    p.add_argument("--motion", action="store_true")
    p.add_argument("--img-host", default="192.168.123.164")
    p.add_argument("--img-port", type=int, default=55555)
    args = p.parse_args()
    args.sim = not args.send_real_robot
    return args


# ---------------------------------------------------------------------------
# Shared robot setup + home pose
# ---------------------------------------------------------------------------

_HOME_Q = np.array([
    # left arm (7 DOF) — episode_0033 frame 0
     0.3927501978988448,  -0.005632476526715517,  0.047096701881357644,
    -0.41113749204796995,  0.1807073597099622,   -0.6146751848814046,
     0.04119819118506424,
    # right arm (7 DOF)
    -0.37670122639527814, -0.14609176253092604,  -0.014642830131267721,
     1.3952594342819227,  -0.3987154953992208,    0.1006491992615599,
     0.01619944107577735,
])


def apply_action(action, arm_ik, arm_ctrl, ee_shared_mem, action_scale):
    """FK → apply delta → IK → send to robot + gripper."""
    current_arm_q = arm_ctrl.get_current_dual_arm_q()
    L_ee, R_ee = get_current_ee_poses(arm_ik, current_arm_q)
    target_L = apply_delta_ee(L_ee, action[:3] * action_scale, action[3:6] * action_scale)
    sol_q, sol_tau = arm_ik.solve_ik(target_L, R_ee, current_arm_q)
    sol_q[7:] = current_arm_q[7:]
    sol_tau[7:] = 0.0
    arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)
    # Training gripper range 0–0.5, controller expects 0–1
    gripper_val = float(np.clip(action[6] * 2.0, 0.0, 1.0))
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

    tv_img_shm = None
    stop_event = threading.Event()
    try:
        # -- Camera ----------------------------------------------------------
        tv_img_shape = (480, 640, 3)
        tv_img_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(tv_img_shape)) * np.uint8().itemsize
        )
        tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            server_address=args.img_host,
            port=args.img_port,
        )
        threading.Thread(target=img_client.receive_process, daemon=True).start()
        logger_mp.info(f"Image client connecting to {args.img_host}:{args.img_port} ...")

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

        time.sleep(1.0)
        cv2.imwrite("/tmp/eval_openvla_frame0.jpg", tv_img_array.copy())
        logger_mp.info("Saved first frame to /tmp/eval_openvla_frame0.jpg")

        # ====================================================================
        # OFT mode: background inference + temporal ensembling
        # ====================================================================
        if args.oft:
            ensemble_buffer: list = []
            buffer_lock = threading.Lock()
            exec_step_ref = [0]
            first_chunk_event = threading.Event()

            threading.Thread(
                target=inference_worker,
                args=(args.server_host, args.server_port, args.task,
                      tv_img_array, ensemble_buffer, buffer_lock,
                      exec_step_ref, first_chunk_event, stop_event,
                      args.chunk_size),
                daemon=True,
            ).start()

            logger_mp.info(
                f"[OFT] chunk_size={args.chunk_size} lambda={args.ensemble_lambda} "
                f"scale={args.action_scale} | task: '{args.task}'"
            )
            logger_mp.info("Waiting for first inference chunk ...")
            if not first_chunk_event.wait(timeout=30.0):
                logger_mp.error("Timed out waiting for inference server.")
                return
            logger_mp.info("First chunk received — starting execution loop.")

            step = 0
            while True:
                t0 = time.perf_counter()
                with buffer_lock:
                    exec_step_ref[0] = step
                    action = temporal_ensemble(ensemble_buffer, step, args.ensemble_lambda)

                if action is None:
                    time.sleep(0.05)
                    continue

                L_ee, target_L, gripper_val = apply_action(
                    action, arm_ik, arm_ctrl, ee_shared_mem, args.action_scale
                )

                if step % 5 == 0:
                    with buffer_lock:
                        n_chunks = len(ensemble_buffer)
                    logger_mp.info(
                        f"[step {step}] action={np.round(action, 4)} | "
                        f"L_pos {np.round(L_ee[:3,3],3)} -> {np.round(target_L[:3,3],3)} | "
                        f"gripper={gripper_val:.2f} | buf={n_chunks} | "
                        f"ik={int((time.perf_counter()-t0)*1000)}ms"
                    )
                step += 1

        # ====================================================================
        # Standard OpenVLA mode: simple closed-loop, ZMQ in main thread
        # ====================================================================
        else:
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REQ)
            socket.connect(f"tcp://{args.server_host}:{args.server_port}")
            socket.setsockopt(zmq.RCVTIMEO, 15000)
            logger_mp.info(
                f"[Standard] scale={args.action_scale} freq={args.frequency}Hz | "
                f"task: '{args.task}'"
            )

            step = 0
            while True:
                t0 = time.perf_counter()

                img_b64 = encode_image_jpeg(tv_img_array.copy())
                try:
                    socket.send_json({"image": img_b64, "task": args.task})
                    resp = socket.recv_json()
                except zmq.Again:
                    logger_mp.warning("Server timeout — skipping step.")
                    continue

                if resp.get("status") != "ok":
                    logger_mp.warning(f"Server error: {resp.get('status')}")
                    continue

                action = np.array(resp.get("action") or resp["actions"][0], dtype=np.float64)

                L_ee, target_L, gripper_val = apply_action(
                    action, arm_ik, arm_ctrl, ee_shared_mem, args.action_scale
                )

                if step % 5 == 0:
                    logger_mp.info(
                        f"[step {step}] action={np.round(action, 4)} | "
                        f"L_pos {np.round(L_ee[:3,3],3)} -> {np.round(target_L[:3,3],3)} | "
                        f"gripper={gripper_val:.2f} | "
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
        stop_event.set()
        if tv_img_shm is not None:
            try:
                tv_img_shm.close()
                tv_img_shm.unlink()
            except Exception:
                pass
        logger_mp.info("End of eval.")


if __name__ == "__main__":
    main()

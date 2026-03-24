"""
OpenVLA eval script for G1 robot with Inspire FTP gripper.

OpenVLA outputs 7-DOF delta end-effector actions:
    [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
for the LEFT arm. Right arm stays at current position.

Inference runs on a remote GPU machine (openvla_server.py).
This script handles DDS, camera capture, FK/IK, and action execution.

Usage (robot Jetson NX):
    conda activate unitree_lerobot
    cd /home/unitree/AlphaZ_WS/unitree_lerobot
    python unitree_lerobot/eval_robot/eval_openvla.py \\
        --server-host 192.168.123.162 \\
        --server-port 5555 \\
        --task "pick up the orange bottle and place it in the black box" \\
        --frequency 5 \\
        --send-real-robot
"""

import argparse
import base64
import time
import traceback

import cv2
import numpy as np
import pinocchio as pin
import zmq

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from unitree_lerobot.eval_robot.make_robot import setup_image_client, setup_robot_interface
from unitree_lerobot.eval_robot.utils.utils import cleanup_resources


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_current_ee_poses(ik_solver, arm_q: np.ndarray, _cache={}):
    """Forward kinematics: returns (L_ee_4x4, R_ee_4x4) from current joint angles.

    Creates a fresh pin.Data on first call (reduced_robot.data is stale because
    L_ee / R_ee frames are added to the model after data was originally created).
    """
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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA robot eval (robot side)")
    p.add_argument("--server-host", default="192.168.123.162",
                   help="IP of the OpenVLA inference server")
    p.add_argument("--server-port", type=int, default=5555,
                   help="ZMQ port of the inference server")
    p.add_argument("--task", default="pick up the orange bottle and place it in the black box",
                   help="Language instruction passed to OpenVLA")
    p.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    p.add_argument("--ee", default="inspire_ftp",
                   help="End-effector type (inspire_ftp / dex1 / '' for none)")
    p.add_argument("--frequency", type=float, default=5.0,
                   help="Control frequency in Hz (OpenVLA is slow; 5 Hz recommended)")
    p.add_argument("--send-real-robot", action="store_true",
                   help="Enable real robot DDS (omit for dry-run)")
    p.add_argument("--motion", action="store_true",
                   help="Use rt/arm_sdk motion topic instead of rt/lowcmd debug topic")
    args = p.parse_args()
    # setup_robot_interface expects args.sim
    args.sim = not args.send_real_robot
    return args


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- ZMQ client ----------------------------------------------------------
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{args.server_host}:{args.server_port}")
    socket.setsockopt(zmq.RCVTIMEO, 15000)  # 15 s timeout per inference call
    logger_mp.info(f"Connecting to OpenVLA server at {args.server_host}:{args.server_port}")

    image_info = None
    try:
        # -- Camera ----------------------------------------------------------
        image_info = setup_image_client(args)
        tv_img_array = image_info["tv_img_array"]
        tv_img_shape = image_info["tv_img_shape"]
        is_binocular = image_info["is_binocular"]

        # -- Robot -----------------------------------------------------------
        robot_interface = setup_robot_interface(args)
        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]

        # -- User confirm ----------------------------------------------------
        user_input = input("Enter 's' to start evaluation: ").strip().lower()
        if user_input != "s":
            logger_mp.info("Aborted.")
            return

        logger_mp.info(
            f"Starting OpenVLA eval at {args.frequency} Hz | task: '{args.task}'"
        )

        step = 0
        while True:
            t0 = time.perf_counter()

            # 1. Capture camera (single / left camera for binocular)
            img = tv_img_array.copy()
            if is_binocular:
                img = img[:, : tv_img_shape[1] // 2]
            img_b64 = encode_image_jpeg(img)

            # 2. Forward kinematics — get current EE poses
            current_arm_q = arm_ctrl.get_current_dual_arm_q()
            L_ee, R_ee = get_current_ee_poses(arm_ik, current_arm_q)

            # 3. Send image + task to OpenVLA server, receive 7-DOF action
            try:
                socket.send_json({"image": img_b64, "task": args.task})
                resp = socket.recv_json()
            except zmq.Again:
                logger_mp.warning("Server timeout — skipping step.")
                continue

            if resp.get("status") != "ok":
                logger_mp.warning(f"Server error: {resp.get('status')}")
                continue

            action = np.array(resp["action"], dtype=np.float64)
            # action = [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]

            # 4. Apply delta to left EE; right EE stays fixed
            target_L = apply_delta_ee(L_ee, action[:3], action[3:6])
            target_R = R_ee

            # 5. IK → joint positions + torques
            sol_q, sol_tau = arm_ik.solve_ik(target_L, target_R, current_arm_q)
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)

            # 6. Gripper (single open/close value per hand)
            gripper_val = float(np.clip(action[6], 0.0, 1.0))
            if ee_shared_mem:
                left_mem = ee_shared_mem.get("left")
                right_mem = ee_shared_mem.get("right")
                if left_mem is not None and hasattr(left_mem, "value"):
                    left_mem.value = gripper_val
                if right_mem is not None and hasattr(right_mem, "value"):
                    right_mem.value = gripper_val

            if step % 10 == 0:
                logger_mp.info(
                    f"[step {step}] action={np.round(action, 4)} | "
                    f"L_pos {np.round(L_ee[:3,3],3)} -> {np.round(target_L[:3,3],3)} | "
                    f"gripper={gripper_val:.2f}"
                )
            step += 1

            # 7. Maintain target frequency
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, (1.0 / args.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        if image_info:
            cleanup_resources(image_info)
        logger_mp.info("End of eval.")


if __name__ == "__main__":
    main()

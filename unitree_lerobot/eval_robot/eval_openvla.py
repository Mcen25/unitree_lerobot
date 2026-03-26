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
    p.add_argument("--img-host", default="192.168.123.164",
                   help="IP of the teleimager-server (image server)")
    p.add_argument("--img-port", type=int, default=55555,
                   help="ZMQ PUB port of the teleimager-server")
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

    _SAVED_Q_PATH = "/tmp/eval_openvla_last_q.npy"
    tv_img_shm = None
    arm_ctrl = None
    try:
        # -- Camera ----------------------------------------------------------
        # teleimager-server publishes 480×640 single head camera on port 55555
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
        image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
        image_receive_thread.start()
        logger_mp.info(f"Image client connecting to {args.img_host}:{args.img_port} ...")

        # -- Robot -----------------------------------------------------------
        robot_interface = setup_robot_interface(args)
        arm_ctrl = robot_interface["arm_ctrl"]
        arm_ik = robot_interface["arm_ik"]
        ee_shared_mem = robot_interface["ee_shared_mem"]

        # Default home pose from training data (episode_0190)
        _HOME_Q = np.array([
            # left arm (7 DOF)
             0.402046799659729,   0.0664764940738678,   0.10903248190879822,
            -0.13733921945095062, -0.22342191636562347, -0.6033697724342346,
            -0.2659778892993927,
            # right arm (7 DOF)
            -0.18102172017097473, -0.11210044473409653,  0.08644221723079681,
             1.0941717624664307,  -0.1569933444261551,  -0.12699683010578156,
             0.25151294469833374,
        ])

        # -- Move to home position --------------------------------------------
        current_q = arm_ctrl.get_current_dual_arm_q()
        logger_mp.info(f"Moving arm to home position: {np.round(_HOME_Q, 3)}")
        steps = 200
        for i in range(1, steps + 1):
            interp_q = current_q + (_HOME_Q - current_q) * (i / steps)
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
        user_input = input("Enter 's' to start evaluation: ").strip().lower()
        if user_input != "s":
            logger_mp.info("Aborted.")
            return

        logger_mp.info(
            f"Starting OpenVLA eval at {args.frequency} Hz | task: '{args.task}'"
        )

        # Wait briefly for first frame to arrive
        time.sleep(1.0)
        # Save the first frame for debugging (verify camera view matches training)
        cv2.imwrite("/tmp/eval_openvla_frame0.jpg", tv_img_array.copy())
        logger_mp.info("Saved first frame to /tmp/eval_openvla_frame0.jpg")

        step = 0
        while True:
            t0 = time.perf_counter()

            # 1. Capture head camera frame
            img = tv_img_array.copy()
            if step % 10 == 0:
                img_hash = int(np.sum(img.astype(np.int64)) % 100000)
                logger_mp.info(f"[step {step}] img_hash={img_hash} (changes → camera updating)")
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

            # 5. IK → joint positions + torques (left arm only)
            sol_q, sol_tau = arm_ik.solve_ik(target_L, target_R, current_arm_q)
            # Lock right arm at current joint angles — only left arm should move
            sol_q[7:] = current_arm_q[7:]
            sol_tau[7:] = 0.0
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)

            # 6. Left gripper only — right gripper stays fixed
            gripper_val = float(np.clip(action[6], 0.0, 1.0))
            if ee_shared_mem:
                left_mem = ee_shared_mem.get("left")
                if left_mem is not None and hasattr(left_mem, "value"):
                    left_mem.value = gripper_val

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
        if tv_img_shm is not None:
            try:
                tv_img_shm.close()
                tv_img_shm.unlink()
            except Exception:
                pass
        logger_mp.info("End of eval.")


if __name__ == "__main__":
    main()

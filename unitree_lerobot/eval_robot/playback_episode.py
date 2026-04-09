"""
Playback a recorded episode on the real G1 robot.

Two modes:
  --mode joint  (default)  Replay actions.left_arm.qpos directly — no IK involved.
                            This is the ground truth for "did the robot do the right thing".
  --mode ik                 Apply left_ee_delta.qpos through FK→delta→IK, exactly as
                            eval_openvla.py does at inference time.

Usage:
    cd /home/unitree/AlphaZ_WS/unitree_lerobot
    conda activate unitree_lerobot

    # Direct joint replay — safest, most faithful:
    python unitree_lerobot/eval_robot/playback_episode.py \\
        --episode ~/AlphaZ_WS/xr_teleoperate/teleop/utils/data/pick_n_place_can_into_black_box/episode_0000/data.json \\
        --send-real-robot --motion

    # IK-based delta replay (tests the eval pipeline):
    python unitree_lerobot/eval_robot/playback_episode.py \\
        --episode ~/AlphaZ_WS/xr_teleoperate/teleop/utils/data/pick_n_place_can_into_black_box/episode_0000/data.json \\
        --send-real-robot --motion --mode ik

    # Half speed:
    python unitree_lerobot/eval_robot/playback_episode.py \\
        --episode .../episode_0000/data.json --send-real-robot --motion --speed 0.5
"""

import argparse
import json
import time
import traceback

import numpy as np
import pinocchio as pin

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from unitree_lerobot.eval_robot.make_robot import setup_robot_interface
from unitree_lerobot.eval_robot.eval_openvla import (
    get_current_ee_poses,
    apply_delta_ee,
    _HOME_Q,
)
from unitree_lerobot.eval_robot.utils.weighted_moving_filter import WeightedMovingFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rpy_of(mat4x4: np.ndarray) -> np.ndarray:
    return pin.rpy.matrixToRpy(mat4x4[:3, :3])


def gravity_torque(arm_ik, q14: np.ndarray) -> np.ndarray:
    """Compute static gravity-compensation torques via RNEA (v=0, a=0)."""
    nv = arm_ik.reduced_robot.model.nv
    return pin.rnea(
        arm_ik.reduced_robot.model,
        arm_ik.reduced_robot.data,
        q14,
        np.zeros(nv),
        np.zeros(nv),
    )


def move_to_q(arm_ctrl, arm_ik, target_q: np.ndarray, n_steps: int = 200, step_dt: float = 0.01):
    """Linearly interpolate from current joint positions to target_q with gravity compensation."""
    current_q = arm_ctrl.get_current_dual_arm_q()
    for i in range(1, n_steps + 1):
        interp_q = current_q + (target_q - current_q) * (i / n_steps)
        tau = gravity_torque(arm_ik, interp_q)
        arm_ctrl.ctrl_dual_arm(interp_q, tau)
        time.sleep(step_dt)


def wait_for_convergence(arm_ctrl, target_q: np.ndarray, tol: float = 0.05,
                          timeout: float = 5.0, check_dt: float = 0.05) -> bool:
    """Wait until arm is within tol rad of target_q (max-norm) or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current_q = arm_ctrl.get_current_dual_arm_q()
        err = np.max(np.abs(current_q - target_q))
        if err < tol:
            return True
        time.sleep(check_dt)
    err = np.max(np.abs(arm_ctrl.get_current_dual_arm_q() - target_q))
    logger_mp.warning(f"Convergence timeout — max joint error = {np.degrees(err):.1f}° (tol {np.degrees(tol):.1f}°)")
    return False



# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Replay a recorded episode on the real robot")
    p.add_argument("--episode", required=True, help="Path to episode data.json")
    p.add_argument("--mode", default="joint", choices=["joint", "ik", "delta"],
                   help="joint=replay recorded joint commands directly; "
                        "ik=IK from absolute EE targets (best-case IK test); "
                        "delta=IK from delta integration, mirrors eval_openvla exactly")
    p.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    p.add_argument("--ee",  default="inspire_ftp")
    p.add_argument("--send-real-robot", action="store_true")
    p.add_argument("--motion", action="store_true")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (0.5 = half speed). Default 1.0.")
    p.add_argument("--start-steps", type=int, default=400,
                   help="Interpolation steps to reach starting pose (default 400 = 4s at 10ms/step)")
    p.add_argument("--converge-tol", type=float, default=0.05,
                   help="Joint convergence tolerance in rad before playback starts (default 0.05)")
    args = p.parse_args()
    args.sim = not args.send_real_robot
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Load episode --------------------------------------------------------
    with open(args.episode, encoding="utf-8") as f:
        ep = json.load(f)

    frames = ep["data"]
    fps    = ep["info"]["image"]["fps"]
    dt     = (1.0 / fps) / args.speed
    n      = len(frames)
    logger_mp.info(f"Episode : {args.episode}")
    logger_mp.info(f"Mode    : {args.mode}")
    logger_mp.info(f"Frames  : {n}  FPS: {fps}  dt: {dt:.3f}s  speed: {args.speed}x")

    # Pre-extract arrays
    left_arm_actions = np.array([f["actions"]["left_arm"]["qpos"] for f in frames], dtype=np.float64)
    right_arm_q_ref  = np.array([f["actions"]["right_arm"]["qpos"] for f in frames], dtype=np.float64)
    deltas  = np.array([f["actions"]["left_ee_delta"]["qpos"] for f in frames], dtype=np.float64)
    gt_abs  = np.array([f["states"]["left_ee_abs"]["qpos"]   for f in frames], dtype=np.float64)

    # -- Robot setup ---------------------------------------------------------
    robot_interface = setup_robot_interface(args)
    arm_ctrl   = robot_interface["arm_ctrl"]
    arm_ik     = robot_interface["arm_ik"]
    ee_mem     = robot_interface["ee_shared_mem"]

    # -- Build start target --------------------------------------------------
    # Left arm from episode frame 0 actions; right arm from home pose
    start_left_q  = left_arm_actions[0]
    start_right_q = _HOME_Q[7:]
    target_start_q = np.concatenate([start_left_q, start_right_q])

    current_q = arm_ctrl.get_current_dual_arm_q()
    logger_mp.info(f"Current q (7 left): {np.round(current_q[:7], 4)}")
    logger_mp.info(f"Target  q (7 left): {np.round(start_left_q,  4)}")
    logger_mp.info(f"Max joint delta    : {np.degrees(np.max(np.abs(target_start_q - current_q))):.1f}°")

    logger_mp.info(f"Moving to episode starting configuration ({args.start_steps} steps × 10ms) ...")
    move_to_q(arm_ctrl, arm_ik, target_start_q, n_steps=args.start_steps, step_dt=0.01)
    logger_mp.info("Interpolation done. Waiting for convergence ...")
    reached = wait_for_convergence(arm_ctrl, target_start_q, tol=args.converge_tol, timeout=5.0)

    # Report where the arm actually ended up
    actual_q = arm_ctrl.get_current_dual_arm_q()
    logger_mp.info(f"Actual  q (7 left): {np.round(actual_q[:7], 4)}")
    logger_mp.info(f"Target  q (7 left): {np.round(start_left_q, 4)}")
    logger_mp.info(f"Joint error (max): {np.degrees(np.max(np.abs(actual_q - target_start_q))):.2f}°  "
                   f"{'OK' if reached else 'WARN-not-converged'}")

    if args.mode in ("ik", "delta"):
        # Identity smooth filter for both IK modes — no lag, no attenuation.
        arm_ik.smooth_filter = WeightedMovingFilter(np.array([1.0]), 14)

        L_ee_start, _ = get_current_ee_poses(arm_ik, actual_q)
        logger_mp.info(f"EE at start (arm actual): xyz={np.round(L_ee_start[:3,3],4)}  "
                       f"rpy={np.round(rpy_of(L_ee_start),4)}")
        logger_mp.info(f"GT  at start (episode)  : xyz={np.round(gt_abs[0,:3],4)}  "
                       f"rpy={np.round(gt_abs[0,3:6],4)}")

        if args.mode == "ik":
            # Seed init_data with the episode starting joints so the first IK
            # solve starts from the correct solution branch.
            arm_ik.init_data = np.concatenate([left_arm_actions[0], actual_q[7:]])

    # Open gripper to match episode start
    if ee_mem:
        left_mem = ee_mem.get("left")
        if left_mem is not None and hasattr(left_mem, "value"):
            left_mem.value = float(gt_abs[0, 6] * 2.0)

    if input("Enter 's' to start playback: ").strip().lower() != "s":
        logger_mp.info("Aborted.")
        return

    # -- Playback loop -------------------------------------------------------
    xyz_errors = []
    rpy_errors = []
    # Delta mode: initialise from the GT starting joints (target_start_q), NOT
    # from the sensor reading.  The recorded deltas are relative to the
    # recorded trajectory which starts at left_arm_actions[0].  If we start
    # integration from the sensor position (which may be 10-20 mm off after
    # imperfect convergence) the trajectory drifts immediately.  Starting from
    # target_start_q answers the meaningful question: "given correct starting
    # position, does the delta IK pipeline track the trajectory?"
    delta_prev_sol_q = target_start_q.copy()

    try:
        for i in range(n):
            t0 = time.perf_counter()
            gt = gt_abs[i]

            if args.mode == "joint":
                # ── Direct joint replay ─────────────────────────────────────
                target_q_14 = np.concatenate([left_arm_actions[i], actual_q[7:]])
                tau = gravity_torque(arm_ik, target_q_14)
                arm_ctrl.ctrl_dual_arm(target_q_14, tau)

                # FK for logging
                current_arm_q = arm_ctrl.get_current_dual_arm_q()
                L_ee, _ = get_current_ee_poses(arm_ik, current_arm_q)
                xyz_e = float(np.linalg.norm(L_ee[:3, 3] - gt[:3]))
                rpy_e = float(np.linalg.norm(rpy_of(L_ee) - gt[3:6]))

                # Gripper from actions
                raw_grip  = float(np.clip(deltas[i, 6] * 2.0, 0.0, 1.0))
                grip_out  = raw_grip

                logger_mp.info(
                    f"[{i:3d}/{n}] JOINT  "
                    f"cmd_q={np.round(left_arm_actions[i], 4)}  "
                    f"grip={grip_out:.2f} | "
                    f"L_pos={np.round(L_ee[:3,3],4)} GT={np.round(gt[:3],4)} "
                    f"xyz_err={xyz_e*1000:.1f}mm  rpy_err={np.degrees(rpy_e):.1f}°"
                )

            elif args.mode == "ik":
                # ── IK-based EE replay ──────────────────────────────────────
                #
                # Use the ground-truth absolute EE pose from left_ee_abs[i]
                # as the IK target — not delta-integrated from the sensor.
                # Delta integration accumulates error because the sensor
                # lags the commanded position (1-step feedback delay at 10Hz
                # = 100ms offset).  Absolute targets are exact.
                #
                # Seed init_data with left_arm_actions[i] — the recorded
                # joints for this exact frame.  This gives IPOPT the right
                # answer as a warm-start so it stays on the correct solution
                # branch and doesn't drift via regularization.
                current_arm_q = arm_ctrl.get_current_dual_arm_q()
                _, R_ee = get_current_ee_poses(arm_ik, current_arm_q)

                # Build 4×4 SE3 target from recorded absolute EE pose
                target_L = np.eye(4)
                target_L[:3, 3]  = gt_abs[i, :3]
                target_L[:3, :3] = pin.rpy.rpyToMatrix(
                    float(gt_abs[i, 3]), float(gt_abs[i, 4]), float(gt_abs[i, 5])
                )

                # Warm-start: current frame's recorded joints
                arm_ik.init_data = np.concatenate([left_arm_actions[i], current_arm_q[7:]])

                sol_q, sol_tau = arm_ik.solve_ik(target_L, R_ee)  # no current_arm_q — keep our seed
                sol_q[7:]   = current_arm_q[7:]
                sol_tau[7:] = gravity_torque(arm_ik, sol_q)[7:]
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)

                # Measure FK of actual arm vs ground truth (same 1-step lag as joint mode)
                L_ee, _ = get_current_ee_poses(arm_ik, current_arm_q)
                xyz_e = float(np.linalg.norm(L_ee[:3, 3] - gt[:3]))
                rpy_e = float(np.linalg.norm(rpy_of(L_ee) - gt[3:6]))

                raw_grip  = float(np.clip(deltas[i, 6] * 2.0, 0.0, 1.0))
                grip_out  = raw_grip

                logger_mp.info(
                    f"[{i:3d}/{n}] IK  "
                    f"target_xyz={np.round(gt_abs[i,:3],4)}  "
                    f"grip={grip_out:.2f} | "
                    f"L_pos={np.round(L_ee[:3,3],4)} IK_target={np.round(target_L[:3,3],4)} | "
                    f"GT={np.round(gt[:3],4)} "
                    f"xyz_err={xyz_e*1000:.1f}mm  rpy_err={np.degrees(rpy_e):.1f}°"
                )

            elif args.mode == "delta":
                # ── Delta IK ────────────────────────────────────────────────
                # Apply delta to FK of the PREVIOUS IK solution (commanded
                # state), not the sensor reading. The deltas were computed
                # from FK(recorded joints), so integrating from FK(sensor_q)
                # compounds the 1-step sensor lag into every target.
                delta = deltas[i]
                current_arm_q = arm_ctrl.get_current_dual_arm_q()

                # FK from commanded state for delta integration
                L_ee_cmd, R_ee = get_current_ee_poses(arm_ik, delta_prev_sol_q)
                target_L = apply_delta_ee(L_ee_cmd, delta[:3], delta[3:6])

                # IK seeded from previous commanded solution
                sol_q, sol_tau = arm_ik.solve_ik(target_L, R_ee, delta_prev_sol_q)
                sol_q[7:]   = current_arm_q[7:]
                nv = arm_ik.reduced_robot.model.nv
                sol_tau = pin.rnea(arm_ik.reduced_robot.model, arm_ik.reduced_robot.data,
                                   sol_q, np.zeros(nv), np.zeros(nv))
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)
                delta_prev_sol_q = sol_q.copy()

                # Log against sensor FK (same 1-step lag as other modes)
                L_ee, _ = get_current_ee_poses(arm_ik, current_arm_q)
                xyz_e = float(np.linalg.norm(L_ee[:3, 3] - gt[:3]))
                rpy_e = float(np.linalg.norm(rpy_of(L_ee) - gt[3:6]))

                raw_grip  = float(np.clip(delta[6] * 2.0, 0.0, 1.0))
                grip_out  = raw_grip

                logger_mp.info(
                    f"[{i:3d}/{n}] DELTA  "
                    f"delta_xyz={np.round(delta[:3],5)}  delta_rpy={np.round(delta[3:6],4)}  "
                    f"grip={grip_out:.2f} | "
                    f"L_pos={np.round(L_ee[:3,3],4)} → {np.round(target_L[:3,3],4)} | "
                    f"GT={np.round(gt[:3],4)} "
                    f"xyz_err={xyz_e*1000:.1f}mm  rpy_err={np.degrees(rpy_e):.1f}°"
                )

            xyz_errors.append(xyz_e)
            rpy_errors.append(rpy_e)

            if ee_mem:
                left_mem = ee_mem.get("left")
                if left_mem is not None and hasattr(left_mem, "value"):
                    left_mem.value = grip_out

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("Interrupted.")
    except Exception:
        traceback.print_exc()

    # -- Summary -------------------------------------------------------------
    if xyz_errors:
        xe = np.array(xyz_errors)
        re = np.array(rpy_errors)
        logger_mp.info("=== Tracking summary ===")
        logger_mp.info(f"  XYZ error:  mean={xe.mean()*1000:.1f}mm  max={xe.max()*1000:.1f}mm")
        logger_mp.info(f"  RPY error:  mean={np.degrees(re.mean()):.1f}°  max={np.degrees(re.max()):.1f}°")
        logger_mp.info("(errors = FK-vs-GT at start of each step, i.e. where the arm actually is vs where recording was)")


if __name__ == "__main__":
    main()

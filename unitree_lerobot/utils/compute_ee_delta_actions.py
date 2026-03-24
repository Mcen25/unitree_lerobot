"""
Post-process xr_teleoperate JSON episodes to add EE delta actions.

Reads recorded joint-space actions (left_arm.qpos, right_arm.qpos),
runs forward kinematics using the G1 pinocchio model, and writes
EE delta actions [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
back into each episode's data.json under actions.left_ee_delta and
actions.right_ee_delta.

This matches the 7-DOF action format used by OpenVLA pretraining data.

Usage:
    python compute_ee_delta_actions.py \\
        --dataset-dir /path/to/task_dir \\
        --robot g1_29 \\
        --repo-root /path/to/xr_teleoperate

Example:
    python compute_ee_delta_actions.py \\
        --dataset-dir ~/AlphaZ_WS/xr_teleoperate/teleop/utils/data/pick_n_place_orange_bottle \\
        --robot g1_29
"""

import argparse
import glob
import json
import os

import numpy as np
import pinocchio as pin


# ── Robot configs: which joints to lock, where L_ee/R_ee attach ─────────────

URDF_CONFIGS = {
    "g1_29": {
        "urdf":      "assets/g1/g1_body29_hand14.urdf",
        "model_dir": "assets/g1/",
        "joints_to_lock": [
            "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
            "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
            "left_hand_thumb_0_joint",  "left_hand_thumb_1_joint",  "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint",  "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint","right_hand_middle_1_joint",
        ],
        "L_ee_parent": "left_wrist_yaw_joint",
        "R_ee_parent": "right_wrist_yaw_joint",
        "ee_offset": np.array([0.05, 0.0, 0.0]),
    },
    "g1_23": {
        "urdf":      "assets/g1/g1_body23.urdf",
        "model_dir": "assets/g1/",
        "joints_to_lock": [
            "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
            "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
        ],
        "L_ee_parent": "left_wrist_roll_joint",
        "R_ee_parent": "right_wrist_roll_joint",
        "ee_offset": np.array([0.20, 0.0, 0.0]),
    },
}


# ── Model loading ────────────────────────────────────────────────────────────

def build_reduced_robot(repo_root: str, robot_key: str) -> pin.RobotWrapper:
    cfg = URDF_CONFIGS[robot_key]
    urdf_path  = os.path.join(repo_root, cfg["urdf"])
    model_dir  = os.path.join(repo_root, cfg["model_dir"])

    full_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, model_dir)
    reduced    = full_robot.buildReducedRobot(
        list_of_joints_to_lock=cfg["joints_to_lock"],
        reference_configuration=np.zeros(full_robot.model.nq),
    )

    # Add L_ee and R_ee operational frames (same offsets as robot_arm_ik.py)
    for name, parent_joint in [("L_ee", cfg["L_ee_parent"]), ("R_ee", cfg["R_ee_parent"])]:
        reduced.model.addFrame(pin.Frame(
            name,
            reduced.model.getJointId(parent_joint),
            pin.SE3(np.eye(3), cfg["ee_offset"]),
            pin.FrameType.OP_FRAME,
        ))

    reduced.data = reduced.model.createData()
    return reduced


# ── FK and delta helpers ─────────────────────────────────────────────────────

def joints_to_ee_poses(
    model: pin.Model,
    data:  pin.Data,
    L_id:  int,
    R_id:  int,
    q14:   np.ndarray,
):
    """Run FK on 14-DOF arm config, return (T_L, T_R) as pin.SE3 copies."""
    pin.framesForwardKinematics(model, data, q14)
    return data.oMf[L_id].copy(), data.oMf[R_id].copy()


def se3_delta(T_prev: pin.SE3, T_curr: pin.SE3) -> np.ndarray:
    """
    Compute 6-DOF delta between two SE3 poses (world frame).
      translation: T_curr.t - T_prev.t
      rotation:    RPY of  R_curr @ R_prev.T
    Returns shape (6,): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
    """
    delta_t   = T_curr.translation - T_prev.translation
    R_delta   = T_curr.rotation @ T_prev.rotation.T
    delta_rpy = pin.rpy.matrixToRpy(R_delta)
    return np.concatenate([delta_t, delta_rpy])


# ── Episode processing ───────────────────────────────────────────────────────

def process_episode(
    episode_path: str,
    model: pin.Model,
    data:  pin.Data,
    L_id:  int,
    R_id:  int,
    target_fps: float | None = None,
    left_only: bool = False,
) -> tuple[int, int]:
    json_path = os.path.join(episode_path, "data.json")
    with open(json_path, encoding="utf-8") as f:
        ep = json.load(f)

    source_fps = ep["info"]["image"]["fps"]
    all_frames = ep["data"]

    # Subsample frames if target_fps requested
    if target_fps is not None and target_fps < source_fps:
        stride = round(source_fps / target_fps)
        frames = all_frames[::stride]
    else:
        stride = 1
        frames = all_frames

    # Re-index frames after subsampling
    for new_idx, frame in enumerate(frames):
        frame["idx"] = new_idx

    # Pass 1: FK for every sampled frame
    poses_L, poses_R = [], []
    for frame in frames:
        left_q  = np.array(frame["actions"]["left_arm"]["qpos"],  dtype=np.float64)
        right_q = np.array(frame["actions"]["right_arm"]["qpos"], dtype=np.float64)
        T_L, T_R = joints_to_ee_poses(model, data, L_id, R_id, np.concatenate([left_q, right_q]))
        poses_L.append(T_L)
        poses_R.append(T_R)

    # Pass 2: compute deltas + absolute poses, write into each frame
    for i, frame in enumerate(frames):
        l_grip = frame["actions"]["left_ee"]["qpos"][:1] if frame["actions"]["left_ee"]["qpos"] else [0.0]

        # Absolute EE pose as state: [x, y, z, roll, pitch, yaw, gripper]
        T_L = poses_L[i]
        abs_L = np.concatenate([T_L.translation, pin.rpy.matrixToRpy(T_L.rotation), l_grip])
        frame["states"]["left_ee_abs"] = {"qpos": abs_L.tolist()}

        # Delta action: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
        d_L = np.zeros(6) if i == 0 else se3_delta(poses_L[i - 1], poses_L[i])
        frame["actions"]["left_ee_delta"] = {"qpos": np.concatenate([d_L, l_grip]).tolist()}

        if not left_only:
            r_grip = frame["actions"]["right_ee"]["qpos"][:1] if frame["actions"]["right_ee"]["qpos"] else [0.0]
            T_R = poses_R[i]
            abs_R = np.concatenate([T_R.translation, pin.rpy.matrixToRpy(T_R.rotation), r_grip])
            frame["states"]["right_ee_abs"] = {"qpos": abs_R.tolist()}
            d_R = np.zeros(6) if i == 0 else se3_delta(poses_R[i - 1], poses_R[i])
            frame["actions"]["right_ee_delta"] = {"qpos": np.concatenate([d_R, r_grip]).tolist()}

    # Update episode: replace data list and record new fps
    ep["data"] = frames
    if target_fps is not None and target_fps < source_fps:
        ep["info"]["image"]["fps"] = source_fps / stride

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ep, f, ensure_ascii=False, indent=4)

    return len(all_frames), len(frames)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add EE delta actions to xr_teleoperate JSON episodes."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Task directory containing episode_XXXX/ sub-folders",
    )
    parser.add_argument(
        "--robot", default="g1_29", choices=list(URDF_CONFIGS.keys()),
        help="Robot model to use for FK",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Path to xr_teleoperate repo root (auto-detected if omitted)",
    )
    parser.add_argument(
        "--target-fps", type=float, default=None,
        help="Subsample episodes to this Hz (e.g. 10). Source fps read from data.json info.",
    )
    parser.add_argument(
        "--left-only", action="store_true",
        help="Only compute and store left arm EE delta (skips right arm).",
    )
    args = parser.parse_args()

    if args.repo_root is None:
        # Auto-detect: assume this script is in unitree_lerobot/unitree_lerobot/utils/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.repo_root = os.path.normpath(os.path.join(script_dir, "../../../xr_teleoperate"))

    print(f"Loading pinocchio model '{args.robot}' from {args.repo_root} ...")
    robot  = build_reduced_robot(args.repo_root, args.robot)
    model  = robot.model
    data   = robot.data
    L_id   = model.getFrameId("L_ee")
    R_id   = model.getFrameId("R_ee")
    print(f"  nq={model.nq}  L_ee id={L_id}  R_ee id={R_id}")

    # Support both flat (dataset_dir/episode_*) and nested (dataset_dir/task/episode_*) layouts
    episode_dirs = sorted(glob.glob(os.path.join(args.dataset_dir, "episode_*")))
    if not episode_dirs:
        episode_dirs = sorted(glob.glob(os.path.join(args.dataset_dir, "**/episode_*"), recursive=True))

    if not episode_dirs:
        print(f"No episode_* directories found in {args.dataset_dir}")
        return

    fps_msg = f" (subsampling to {args.target_fps} Hz)" if args.target_fps else ""
    arm_msg = " [left arm only]" if args.left_only else ""
    print(f"Processing {len(episode_dirs)} episodes{fps_msg}{arm_msg} ...")
    total_src = 0
    total_out = 0
    for ep_dir in episode_dirs:
        n_src, n_out = process_episode(ep_dir, model, data, L_id, R_id, args.target_fps, args.left_only)
        total_src += n_src
        total_out += n_out
        print(f"  {os.path.relpath(ep_dir, args.dataset_dir)}: {n_src} → {n_out} frames")

    print(f"\nDone. {len(episode_dirs)} episodes, {total_src} → {total_out} frames total.")
    print("Each frame now has:")
    print("  actions.left_ee_delta.qpos  = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]  (7-DOF)")
    print("  actions.right_ee_delta.qpos = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]  (7-DOF)")


if __name__ == "__main__":
    main()

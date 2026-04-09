"""
Verify left_ee_delta.qpos correctness by forward-integrating deltas
and comparing against stored left_ee_abs.qpos ground truth.

Usage:
    python verify_ee_deltas.py path/to/episode_XXXX/data.json
    python verify_ee_deltas.py path/to/episode_XXXX/data.json --plot
"""

import argparse
import json
import sys

import numpy as np
import pinocchio as pin


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    return pin.rpy.matrixToRpy(R)


def integrate_step(pos: np.ndarray, R: np.ndarray, delta: np.ndarray):
    """
    Apply one delta action to (pos, R).
      delta: [dx, dy, dz, droll, dpitch, dyaw]
    Returns (new_pos, new_R).
    """
    new_pos = pos + delta[:3]
    R_delta = rpy_to_matrix(delta[3:6])
    new_R = R_delta @ R
    return new_pos, new_R


def verify(json_path: str, plot: bool = False):
    with open(json_path, encoding="utf-8") as f:
        ep = json.load(f)

    frames = ep["data"]
    n = len(frames)
    print(f"Episode: {json_path}")
    print(f"Frames : {n}")
    print()

    # Ground-truth absolute poses [x, y, z, roll, pitch, yaw, gripper]
    gt_abs = np.array(
        [frame["states"]["left_ee_abs"]["qpos"] for frame in frames],
        dtype=np.float64,
    )

    # Delta actions [dx, dy, dz, droll, dpitch, dyaw, gripper]
    deltas = np.array(
        [frame["actions"]["left_ee_delta"]["qpos"] for frame in frames],
        dtype=np.float64,
    )

    # Forward-integrate starting from frame 0
    integrated = np.zeros_like(gt_abs)
    integrated[0] = gt_abs[0]

    pos = gt_abs[0, :3].copy()
    R   = rpy_to_matrix(gt_abs[0, 3:6])

    for i in range(n - 1):
        pos, R = integrate_step(pos, R, deltas[i])
        rpy = matrix_to_rpy(R)
        integrated[i + 1, :3]  = pos
        integrated[i + 1, 3:6] = rpy
        integrated[i + 1, 6]   = deltas[i + 1, 6]   # carry gripper from next delta

    # Fix last gripper: use ground truth (last delta gripper is terminal zeros)
    integrated[-1, 6] = gt_abs[-1, 6]

    # Compute errors
    err_xyz     = np.abs(integrated[:, :3] - gt_abs[:, :3])          # (n, 3)
    err_rpy     = np.abs(integrated[:, 3:6] - gt_abs[:, 3:6])        # (n, 3)
    err_gripper = np.abs(integrated[:, 6] - gt_abs[:, 6])            # (n,)

    labels_xyz = ["x", "y", "z"]
    labels_rpy = ["roll", "pitch", "yaw"]

    print("=== Per-axis max absolute error ===")
    for j, lbl in enumerate(labels_xyz):
        print(f"  {lbl}     : max={err_xyz[:, j].max():.6f} m   mean={err_xyz[:, j].mean():.6f} m")
    for j, lbl in enumerate(labels_rpy):
        print(f"  {lbl} : max={err_rpy[:, j].max():.6f} rad  mean={err_rpy[:, j].mean():.6f} rad")
    print(f"  gripper: max={err_gripper.max():.6f}           mean={err_gripper.mean():.6f}")
    print()

    # Frame-by-frame table (first 10 + last 5)
    def print_frame(i):
        gt  = gt_abs[i]
        ig  = integrated[i]
        e   = np.abs(ig - gt)
        print(
            f"  [{i:3d}] "
            f"GT  xyz=({gt[0]:+.4f},{gt[1]:+.4f},{gt[2]:+.4f}) "
            f"rpy=({gt[3]:+.3f},{gt[4]:+.3f},{gt[5]:+.3f}) g={gt[6]:.2f}"
        )
        print(
            f"       "
            f"INT xyz=({ig[0]:+.4f},{ig[1]:+.4f},{ig[2]:+.4f}) "
            f"rpy=({ig[3]:+.3f},{ig[4]:+.3f},{ig[5]:+.3f}) g={ig[6]:.2f}"
        )
        print(
            f"       "
            f"ERR xyz=({e[0]:.5f},{e[1]:.5f},{e[2]:.5f}) "
            f"rpy=({e[3]:.4f},{e[4]:.4f},{e[5]:.4f}) g={e[6]:.4f}"
        )

    print("=== Frame-by-frame (first 10) ===")
    for i in range(min(10, n)):
        print_frame(i)

    if n > 15:
        print("  ...")
        print("=== Frame-by-frame (last 5) ===")
        for i in range(n - 5, n):
            print_frame(i)

    # Summary verdict
    xyz_max = err_xyz.max()
    rpy_max = err_rpy.max()
    print()
    if xyz_max < 1e-4 and rpy_max < 1e-3:
        print("PASS  — integration matches ground truth (numeric roundtrip error only)")
    elif xyz_max < 5e-3 and rpy_max < 0.05:
        print("WARN  — small but non-trivial drift; check RPY wrap-around or floating-point accumulation")
    else:
        print("FAIL  — large mismatch; delta convention or SE3 integration may be wrong")

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            t = np.arange(n)

            # XYZ
            ax = axes[0]
            for j, lbl in enumerate(labels_xyz):
                ax.plot(t, gt_abs[:, j],         label=f"gt_{lbl}",  linewidth=1.5)
                ax.plot(t, integrated[:, j], "--", label=f"int_{lbl}", linewidth=1)
            ax.set_ylabel("position (m)")
            ax.legend(ncol=3, fontsize=8)
            ax.set_title("XYZ: ground truth vs integrated")

            # RPY
            ax = axes[1]
            for j, lbl in enumerate(labels_rpy):
                ax.plot(t, gt_abs[:, 3 + j],         label=f"gt_{lbl}",  linewidth=1.5)
                ax.plot(t, integrated[:, 3 + j], "--", label=f"int_{lbl}", linewidth=1)
            ax.set_ylabel("angle (rad)")
            ax.legend(ncol=3, fontsize=8)
            ax.set_title("RPY: ground truth vs integrated")

            # Gripper
            ax = axes[2]
            ax.plot(t, gt_abs[:, 6],         label="gt_gripper",  linewidth=1.5)
            ax.plot(t, integrated[:, 6], "--", label="int_gripper", linewidth=1)
            ax.set_ylabel("gripper")
            ax.set_xlabel("frame")
            ax.legend(fontsize=8)
            ax.set_title("Gripper: ground truth vs integrated")

            plt.tight_layout()
            out_path = json_path.replace("data.json", "delta_verify.png")
            plt.savefig(out_path, dpi=120)
            print(f"\nPlot saved → {out_path}")
            plt.show()
        except ImportError:
            print("\nmatplotlib not available — skipping plot")

    return integrated, gt_abs


def main():
    parser = argparse.ArgumentParser(description="Verify left_ee_delta forward integration")
    parser.add_argument("json_path", help="Path to episode data.json")
    parser.add_argument("--plot", action="store_true", help="Save/show comparison plot")
    args = parser.parse_args()
    verify(args.json_path, plot=args.plot)


if __name__ == "__main__":
    main()

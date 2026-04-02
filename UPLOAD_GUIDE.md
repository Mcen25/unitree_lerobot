# HuggingFace Upload Guide: xr_teleoperate JSON → LeRobot (EE Format)

## Overview

This guide converts raw `xr_teleoperate` JSON datasets (joint-space, 30 Hz) into the
LeRobot format with **left-arm end-effector (EE) delta actions at 10 Hz**, then uploads
to HuggingFace.

**Action space:** `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]` — 7-DOF EE delta  
**State space:** `[x, y, z, roll, pitch, yaw, gripper]` — 7-DOF EE absolute  
**Cameras:** `cam_high` (color_0) + `cam_down` (color_3)  
**Robot config:** `Unitree_G1_Inspire_LeftArm`  
**Conda env:** `unitree_lerobot`

---

## Prerequisites

```bash
conda activate unitree_lerobot
huggingface-cli whoami          # must be logged in (token in ~/.cache/huggingface)
huggingface-cli login           # if not logged in
```

---

## Step 1 — Fix Any Truncated JSON Files

Raw episodes occasionally have truncated `data.json` (missing closing `]}`).
Detect and fix them:

```bash
TASK_DIR=/home/unitree/AlphaZ_WS/xr_teleoperate/teleop/utils/data/YOUR_TASK

for ep in $TASK_DIR/episode_*/data.json; do
    python3 -c "import json; json.load(open('$ep'))" 2>/dev/null \
        || { echo "Fixing: $ep"; printf '\n]\n}' >> "$ep"; }
done
```

Verify all are valid:
```bash
for ep in $TASK_DIR/episode_*/data.json; do
    python3 -c "import json; json.load(open('$ep'))" 2>/dev/null || echo "STILL INVALID: $ep"
done
```

---

## Step 2 — Compute EE Delta Actions + Subsample to 10 Hz

This step runs forward kinematics (pinocchio) on every episode, adds
`states.left_ee_abs.qpos` and `actions.left_ee_delta.qpos`, and downsamples
from 30 Hz → 10 Hz. **Modifies `data.json` in-place.**

```bash
conda run -n unitree_lerobot python \
    /home/unitree/AlphaZ_WS/unitree_lerobot/unitree_lerobot/utils/compute_ee_delta_actions.py \
    --dataset-dir $TASK_DIR \
    --robot g1_29 \
    --repo-root /home/unitree/AlphaZ_WS/unitree_lerobot/unitree_lerobot/eval_robot \
    --target-fps 10 \
    --left-only
```

**Key flags:**
| Flag | Value | Meaning |
|------|-------|---------|
| `--robot` | `g1_29` | Uses `g1_body29_hand14.urdf` (G1 with 14-DOF inspire hands) |
| `--repo-root` | `.../eval_robot` | Root containing `assets/g1/` URDF folder |
| `--target-fps` | `10` | Subsample from 30 Hz source |
| `--left-only` | flag | Only compute left arm EE (skip right arm) |

**Expected output per episode:** `episode_XXXX: N_src → N_out frames` where `N_out ≈ N_src / 3`.

After this step, each frame in `data.json` has:
- `states.left_ee_abs.qpos` = `[x, y, z, roll, pitch, yaw, gripper]`
- `actions.left_ee_delta.qpos` = `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`

---

## Step 3 — Convert to LeRobot Format and Push to HuggingFace

The converter expects `--raw-dir` to be the **parent** of the task folder.
Use a temporary symlink to avoid processing other tasks in the same data directory:

```bash
mkdir -p /tmp/lerobot_data
ln -sfn $TASK_DIR /tmp/lerobot_data/$(basename $TASK_DIR)

conda run -n unitree_lerobot python \
    /home/unitree/AlphaZ_WS/unitree_lerobot/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /tmp/lerobot_data \
    --repo-id YOUR_HF_USER/YOUR_DATASET_NAME \
    --robot_type Unitree_G1_Inspire_LeftArm \
    --push_to_hub
```

Replace `YOUR_HF_USER/YOUR_DATASET_NAME` (e.g. `Mcen27/pick_n_place_can_into_black_box`).

**What happens internally:**
1. Converts JSON episodes → LeRobot parquet + mp4 videos locally
2. Uploads videos via `upload_large_folder` (Xet chunked transfer)
3. **Re-uploads all parquets via `upload_file`** — bypasses Xet to prevent corruption
4. Verifies each parquet is readable on HuggingFace

> **Why the re-upload?** HuggingFace's Xet chunked transfer (used by `upload_large_folder`)
> can corrupt small files like parquets. The convert script explicitly re-uploads all
> `.parquet` files with `upload_file` after the large folder upload to guarantee integrity.
> This is handled automatically — no manual fix needed.

Output is cached at `~/.cache/huggingface/lerobot/REPO_ID/`.

---

## Verify Upload

```bash
conda run -n unitree_lerobot python -c "
from huggingface_hub import HfApi
for f in sorted(HfApi().list_repo_files('YOUR_HF_USER/YOUR_DATASET_NAME', repo_type='dataset')):
    print(f)
"
```

Expected files:
```
data/chunk-000/file-000.parquet
meta/episodes/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
videos/observation.images.cam_down/chunk-000/file-000.mp4
videos/observation.images.cam_high/chunk-000/file-000.mp4
```

---

## Re-upload Only (dataset already converted locally)

If the local dataset exists at `~/.cache/huggingface/lerobot/REPO_ID/` and you
just need to push again:

```bash
conda run -n unitree_lerobot python \
    /home/unitree/AlphaZ_WS/unitree_lerobot/test/test_local_push_to_hub.py \
    --repo-id YOUR_HF_USER/YOUR_DATASET_NAME \
    --root-path ~/.cache/huggingface/lerobot/YOUR_HF_USER/YOUR_DATASET_NAME
```

---

## Troubleshooting

### Truncated JSON (`Expecting ',' delimiter`)
The episode recording was interrupted. Fix: `printf '\n]\n}' >> data.json`

### Multiple tasks in data directory
The converter picks up **all** task subdirectories under `--raw-dir`. Use the
symlink trick in Step 3 to isolate one task.

### pinocchio import error
```bash
conda run -n unitree_lerobot python -c "import pinocchio; print(pinocchio.__version__)"
```
Should print `3.9.0`. If missing: `conda install -n unitree_lerobot pinocchio -c conda-forge`

---

## File Layout Reference

```
xr_teleoperate/teleop/utils/data/
└── TASK_NAME/
    └── episode_XXXX/
        ├── data.json          ← modified in Step 2 (EE fields + subsampled)
        └── colors/
            ├── NNNNNN_color_0.jpg   ← cam_high (head camera)
            └── NNNNNN_color_3.jpg   ← cam_down (wrist/down camera)

unitree_lerobot/unitree_lerobot/eval_robot/
└── assets/g1/
    └── g1_body29_hand14.urdf  ← used for FK in Step 2

unitree_lerobot/unitree_lerobot/utils/
├── compute_ee_delta_actions.py   ← Step 2 script
└── convert_unitree_json_to_lerobot.py  ← Step 3 script (includes Xet bypass)
```

---

## Worked Example: pick_n_place_can_into_black_box

```bash
TASK_DIR=/home/unitree/AlphaZ_WS/xr_teleoperate/teleop/utils/data/pick_n_place_can_into_black_box
REPO_ID=Mcen27/pick_n_place_can_into_black_box

# Fix truncated JSONs
for ep in $TASK_DIR/episode_*/data.json; do
    python3 -c "import json; json.load(open('$ep'))" 2>/dev/null \
        || { echo "Fixing $ep"; printf '\n]\n}' >> "$ep"; }
done

# Compute EE + subsample
conda run -n unitree_lerobot python \
    /home/unitree/AlphaZ_WS/unitree_lerobot/unitree_lerobot/utils/compute_ee_delta_actions.py \
    --dataset-dir $TASK_DIR --robot g1_29 \
    --repo-root /home/unitree/AlphaZ_WS/unitree_lerobot/unitree_lerobot/eval_robot \
    --target-fps 10 --left-only

# Convert + upload
mkdir -p /tmp/lerobot_data
ln -sfn $TASK_DIR /tmp/lerobot_data/$(basename $TASK_DIR)
conda run -n unitree_lerobot python \
    /home/unitree/AlphaZ_WS/unitree_lerobot/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /tmp/lerobot_data --repo-id $REPO_ID \
    --robot_type Unitree_G1_Inspire_LeftArm --push_to_hub
```

Result: 85 episodes, 8367 frames at 10 Hz → `Mcen27/pick_n_place_can_into_black_box`

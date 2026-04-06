# Dataset Upload Pipeline — Unitree G1 → LeRobot → HuggingFace

This guide documents the full pipeline for converting xr_teleoperate JSON datasets
to the LeRobot format and uploading them to HuggingFace.

---

## Overview

Raw data lives in:
```
/home/unitree/AlphaZ_WS/xr_teleoperate/teleop/utils/data/<task_name>/
```

Each task directory contains `episode_XXXX/` subdirectories with:
- `data.json` — joint states, actions, metadata
- `colors/` — RGB images (color_0 = cam_high, color_3 = cam_down)
- `depths/` — depth images
- `audios/` — audio recordings

---

## Step 1 — Check and Fix Corrupt Episodes

Some recordings end with incomplete JSON. Find and remove them before proceeding.

```bash
BASE="/path/to/task_dir"
for ep in $(ls $BASE | grep episode | sort); do
    python3 -c "
import json
try:
    with open('$BASE/$ep/data.json') as f: json.load(f)
    print('OK $ep')
except Exception as e:
    print('ERR $ep:', str(e)[:60])
" 2>&1
done | grep ERR
```

Remove corrupt episodes:
```bash
rm -rf "$BASE/episode_XXXX"
```

---

## Step 2 — Sort and Rename Episodes

Ensures episodes are numbered sequentially from `episode_0000`.
Must be run even if episodes look sequential — handles gaps from removed corrupt episodes.

```bash
conda activate unitree_lerobot
cd ~/AlphaZ_WS/unitree_lerobot

python unitree_lerobot/utils/sort_and_rename_folders.py \
    --data-dir /path/to/task_dir
```

**Warning:** Run `sort_and_rename_folders.py` ONLY on the task directory,
not a parent that contains multiple tasks — it will merge all tasks into one dataset.

---

## Step 3 — Compute EE Delta Actions

Converts joint-space (qpos) data to Cartesian EE delta actions using FK (Pinocchio).
Also subsamples from source fps to target fps.

```bash
python unitree_lerobot/utils/compute_ee_delta_actions.py \
    --dataset-dir /path/to/task_dir \
    --robot g1_29 \
    --target-fps 10 \
    --left-only
```

This adds to each frame:
- `actions.left_ee_delta.qpos` = `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]` (7-DOF)
- `states.left_ee_abs.qpos` = absolute EE pose in world frame

If source fps == target fps (already 10Hz), `--target-fps 10` is a no-op (safe to run).

Robot choices: `g1_29` (G1 with 29-DOF), `g1_23`.

---

## Step 4 — Create Isolated Symlink Directory

**Critical:** The converter scans ALL subdirectories of `--raw-dir`. If you point it
at the parent data directory, it will merge all tasks into one dataset.

Create a symlink directory containing only the target task:

```bash
mkdir -p /tmp/convert_<task_name>
ln -sf /path/to/task_dir /tmp/convert_<task_name>/<task_name>
```

Example:
```bash
mkdir -p /tmp/convert_pick_up_bottle
ln -sf ~/AlphaZ_WS/xr_teleoperate/teleop/utils/data/pick_up_bottle \
    /tmp/convert_pick_up_bottle/pick_up_bottle
```

---

## Step 5 — Convert to LeRobot Format

```bash
conda activate unitree_lerobot
cd ~/AlphaZ_WS/unitree_lerobot

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /tmp/convert_<task_name> \
    --repo-id Mcen27/<task_name> \
    --robot-type Unitree_G1_Inspire_LeftArm
```

**Robot type for EE delta, left arm only:** `Unitree_G1_Inspire_LeftArm`
- Action: `left_ee_delta.qpos` (7-DOF EE delta)
- State: `left_ee_abs.qpos` (7-DOF EE absolute)
- Cameras: `cam_high` (color_0), `cam_down` (color_3)

Other robot types (see `unitree_lerobot/utils/constants.py`):
- `Unitree_G1_Inspire` — both arms, EE delta
- `Unitree_G1_Inspire_SingleCam` — single cam variant
- `Unitree_G1_Inspire_FTP_SingleCam` — joint-space (qpos), single cam

Output is cached at:
```
~/.cache/huggingface/lerobot/Mcen27/<task_name>/
```

For 250+ episodes this takes ~30-45 minutes due to video encoding.

---

## Step 6 — Upload to HuggingFace

**Use `upload_folder` (NOT `upload_large_folder`).** The large_folder API uses
Xet pointer storage which makes the parquet viewer show corrupt data.

```python
from huggingface_hub import HfApi
api = HfApi()

local_path = "~/.cache/huggingface/lerobot/Mcen27/<task_name>"
repo_id = "Mcen27/<task_name>"

api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path=local_path,
    repo_id=repo_id,
    repo_type="dataset",
)
```

Or run the convenience script:
```bash
conda run -n unitree_lerobot python ~/upload_datasets.py
```

---

## Quick Reference — All Steps for One Dataset

```bash
TASK="pick_up_bottle"
DATA_DIR=~/AlphaZ_WS/xr_teleoperate/teleop/utils/data/$TASK
REPO_ID="Mcen27/$TASK"
conda activate unitree_lerobot && cd ~/AlphaZ_WS/unitree_lerobot

# 1. Find corrupt episodes
for ep in $(ls $DATA_DIR | grep episode | sort); do
    python3 -c "import json; json.load(open('$DATA_DIR/$ep/data.json'))" 2>/dev/null \
        || echo "CORRUPT: $ep"
done

# 2. Sort/rename
python unitree_lerobot/utils/sort_and_rename_folders.py --data-dir $DATA_DIR

# 3. Compute EE deltas
python unitree_lerobot/utils/compute_ee_delta_actions.py \
    --dataset-dir $DATA_DIR --robot g1_29 --target-fps 10 --left-only

# 4. Isolated symlink dir
mkdir -p /tmp/convert_$TASK && ln -sf $DATA_DIR /tmp/convert_$TASK/$TASK

# 5. Convert
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /tmp/convert_$TASK \
    --repo-id $REPO_ID \
    --robot-type Unitree_G1_Inspire_LeftArm

# 6. Upload
conda run -n unitree_lerobot python -c "
from huggingface_hub import HfApi; import os
api = HfApi()
local = os.path.expanduser(f'~/.cache/huggingface/lerobot/$REPO_ID')
api.create_repo('$REPO_ID', repo_type='dataset', exist_ok=True)
api.upload_folder(folder_path=local, repo_id='$REPO_ID', repo_type='dataset')
print('Done: https://huggingface.co/datasets/$REPO_ID')
"
```

---

## Notes

- **fps**: All data should be at 10Hz before conversion. Source at 30fps is subsampled
  with `--target-fps 10` (stride=3). Source already at 10Hz: skip or use `--target-fps 10` (no-op).
- **Gripper**: Training range 0–0.5, eval controller expects 0–1. Multiply by 2 in eval script.
- **Action scale**: If training at 10Hz and eval at 5Hz, use `--action-scale 2.0` (or let it
  auto-compute: `training_fps / eval_fps`).
- **Home pose**: Set `_HOME_Q` in `eval_openvla.py` to a reference frame's joint angles
  (left + right arm qpos, 14-DOF total). Use a frame where the arm is extended toward the object.
- **Existing uploads**: `Mcen27/pick_up_bottle_2` — 149 episodes, 10Hz, EE delta, left arm.

#!/bin/bash
set -e

SESSION="eval_openvla"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n "eval"

tmux send-keys -t "$SESSION:eval" \
    'conda activate unitree_lerobot && cd ~/AlphaZ_WS/unitree_lerobot && python unitree_lerobot/eval_robot/eval_openvla.py \
  --server-host 100.96.139.69 \
  --server-port 5555 \
  --task "pick up the bottle" \
  --arm G1_29 --ee inspire_ftp \
  --action-scale 4.0 \
  --send-real-robot --motion' Enter

tmux attach-session -t "$SESSION"

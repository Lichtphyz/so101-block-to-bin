#!/usr/bin/env bash
# Run the trained policy on the real arm for N episodes, recording them for review.
# Between episodes the arm holds its start pose while you reset the block.
#
#   ./eval.sh                              # 10 episodes, latest ACT checkpoint
#   ./eval.sh 10 outputs/train/act_block_to_bin/checkpoints/030000/pretrained_model
#
# SAFETY: keep a hand on the 12V follower power switch for the first runs.
# Keys (in THIS TERMINAL): →  end episode early   ←  discard/redo   Esc  stop
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
check_hw || true   # leader not needed here

N=${1:-10}
POLICY=${POLICY:-act}
CKPT=${2:-outputs/train/${POLICY}_${TASK_NAME}/checkpoints/last/pretrained_model}
[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }

EVAL_DATASET=$HF_USER/rollout_so101_${TASK_NAME}
rm -rf "$HOME/.cache/huggingface/lerobot/$EVAL_DATASET"

EXTRA=()
[ "$POLICY" = smolvla ] && EXTRA=(--rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}')

lerobot-rollout \
    --strategy.type=episodic \
    --policy.path="$CKPT" \
    "${ROBOT_ARGS[@]}" "${EXTRA[@]}" \
    --dataset.repo_id="$EVAL_DATASET" \
    --dataset.no_stamp=true \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes="$N" \
    --dataset.episode_time_s=${EPISODE_S:-30} \
    --dataset.reset_time_s=${RESET_S:-20} \
    --dataset.push_to_hub=false \
    --task="$TASK" \
    --display_data=true
echo "Score each episode success/fail as it runs. Review videos with:"
echo "  lerobot-dataset-viz --repo-id $EVAL_DATASET --episode-index 0"

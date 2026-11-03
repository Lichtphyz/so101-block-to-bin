#!/usr/bin/env bash
# Train a policy on the recorded dataset (local GPU). Safe to re-run: resumes if the run exists.
#
#   ./train.sh                 # ACT, 50k steps (~4.1 h measured on the 4070 laptop, batch 8)
#   STEPS=100000 ./train.sh    # longer
#   POLICY=smolvla ./train.sh  # fine-tune SmolVLA from lerobot/smolvla_base instead
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

POLICY=${POLICY:-act}
STEPS=${STEPS:-50000}
OUT=outputs/train/${POLICY}_${TASK_NAME}

if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    echo "Resuming $OUT"
    exec lerobot-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true
fi

case $POLICY in
  act)     POLICY_ARGS=(--policy.type=act) ;;
  smolvla) POLICY_ARGS=(--policy.path=lerobot/smolvla_base
             --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}') ;;
  *) echo "unknown POLICY=$POLICY"; exit 1 ;;
esac

lerobot-train \
    "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --steps="$STEPS" \
    --save_freq=10000 \
    --log_freq=200 \
    --output_dir="$OUT" \
    --job_name="${POLICY}_${TASK_NAME}" \
    --wandb.enable=false

#!/usr/bin/env bash
# Runs every no-robot test of the learning pipeline sequentially (one GPU), logging each to
# learning/outputs/test_logs/<name>.log and a one-line PASS/FAIL + duration to summary.txt.
# Usage (from the repo root):  bash learning/run_all_tests.sh
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate so_arm101

OUT=learning/outputs
LOGS=$OUT/test_logs
mkdir -p "$LOGS"
SUMMARY=$LOGS/summary.txt
: > "$SUMMARY"
DS=lerobot/svla_so101_pickplace
COMMON_TRAIN=(--dataset.repo_id=$DS --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false)

run() {  # run <name> <cmd...>
    local name=$1; shift
    echo ">>> $(date +%T) START $name" | tee -a "$SUMMARY"
    local t0=$SECONDS
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$LOGS/$name.gpu" 2>/dev/null &
    local mon=$!
    "$@" > "$LOGS/$name.log" 2>&1
    local rc=$? dt=$((SECONDS - t0))
    kill $mon 2>/dev/null; wait $mon 2>/dev/null
    local peak; peak=$(sort -n "$LOGS/$name.gpu" | tail -1)
    local status=PASS; [ $rc -ne 0 ] && status="FAIL(rc=$rc)"
    printf '%-28s %-12s %5ds  peak GPU %s MiB\n' "$name" "$status" "$dt" "${peak:-?}" | tee -a "$SUMMARY"
}

# --- Guide exercises ---
run ex1_explore_dataset   python learning/explore_dataset.py --frame 500
run ex3_act_pretrained    python learning/train_act_mini.py --steps 1500
run ex3_act_scratch       python learning/train_act_mini.py --steps 1500 --no-pretrained
run ex3_plot              python learning/plot_losses.py
run inference_mini_ckpt   python learning/test_inference.py learning/outputs/act_imagenet_backbone
cp -r $OUT/act_imagenet_backbone $OUT/act_imagenet_backbone_1500   # batch16 test below overwrites it
run batch16_memory        python learning/train_act_mini.py --steps 150 --batch-size 16

# --- Real trainer: ACT, checkpoint + resume ---
rm -rf $OUT/train_act
run lerobot_train_act     lerobot-train --policy.type=act "${COMMON_TRAIN[@]}" \
                            --output_dir=$OUT/train_act --steps=2000 --save_freq=1000 --log_freq=100
run lerobot_train_resume  lerobot-train --config_path=$OUT/train_act/checkpoints/last/pretrained_model/train_config.json \
                            --resume=true --steps=2300
run inference_real_ckpt   python learning/test_inference.py $OUT/train_act/checkpoints/last/pretrained_model

# --- Other policies on 8 GB ---
rm -rf $OUT/train_diffusion
run lerobot_train_diffusion lerobot-train --policy.type=diffusion "${COMMON_TRAIN[@]}" \
                            --output_dir=$OUT/train_diffusion --steps=500 --save_freq=500 --log_freq=50
rm -rf $OUT/train_smolvla
run lerobot_train_smolvla lerobot-train --policy.path=lerobot/smolvla_base "${COMMON_TRAIN[@]}" \
                            --batch_size=4 --output_dir=$OUT/train_smolvla --steps=300 --save_freq=300 --log_freq=25

echo ">>> $(date +%T) ALL DONE" | tee -a "$SUMMARY"

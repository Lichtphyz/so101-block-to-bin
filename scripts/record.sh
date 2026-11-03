#!/usr/bin/env bash
# Record teleop demonstrations into a LeRobotDataset.
#
#   scripts/record.sh 3          # dry run: 3 episodes into a throwaway dataset (..._test), wiped each time
#   scripts/record.sh 50         # real dataset; re-run later to ADD episodes (resumes automatically)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
check_hw

N=${1:-}
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: ./record.sh <number of episodes>    e.g.  scripts/record.sh 3  (dry run)   scripts/record.sh 50  (real)"
    exit 1
fi
EPISODE_S=${EPISODE_S:-25}   # max length; press → as soon as the block is in the bin
RESET_S=${RESET_S:-15}       # time to move the block to the next start spot
# Live camera viewer (Rerun). VIEW=0 disables it: suspected cause of the recording loop freezing
# (2026-10-05). VIEW=1 sends compressed images to the viewer to reduce load.
VIEW=${VIEW:-1}
if [ "$VIEW" = 1 ]; then VIEW_ARGS=(--display_data=true --display_compressed_images=true); else VIEW_ARGS=(--display_data=false); fi
pkill -f "rerun_cli/rerun --port" 2>/dev/null || true   # don't reuse a stale viewer from a previous session

if [ "$N" -le 5 ]; then
    DATASET=${DATASET}_test
    DATASET_DIR=${DATASET_DIR}_test
    rm -rf "$DATASET_DIR"
    PUSH=false
    MODE="DRY RUN → $DATASET (local only, wiped each run)"
else
    PUSH=true
    MODE="REAL → $DATASET (pushed to HF Hub as private at the end)"
fi

RESUME=false
DONE_EPS=0
if [ -f "$DATASET_DIR/meta/info.json" ]; then
    RESUME=true
    DONE_EPS=$(python -c "import json;print(json.load(open('$DATASET_DIR/meta/info.json'))['total_episodes'])")
    MODE="$MODE — resuming: $DONE_EPS episodes already recorded, adding $N"
fi

cat <<EOF

================================ RECORDING ================================
 $MODE
 Task: "$TASK"   ($N episodes, max ${EPISODE_S}s each, ${RESET_S}s reset between)

 The loop runs by itself. NO key is needed to start an episode:
   "Recording episode N"   → do the task with the leader arm (recorded)
                             press  n  (or →) when the block is in the bin
   "Reset the environment" → NOT recorded: arm to rest pose, move the block.
                             press  n  (or →) when ready, or wait ${RESET_S}s
   ...then the episode is SAVED (few-second pause, arm holds still)
   "Recording episode N+1" → starts AUTOMATICALLY after the save

 KEYS — type them in THIS TERMINAL (Wayland: the camera window won't see them):
   n  or →   next: finish this episode / end the reset early
   r  or ←   redo: throw away this episode and record it again
   q  or Esc quit: stop the session; saved episodes are kept
             (re-run the same command later to continue adding episodes)

 Do NOT use Ctrl+Z (freezes the program while it holds the arms) or Ctrl+C.
===========================================================================
EOF
read -rp " Block at its first spot, leader arm at rest pose? Press Enter to START episode 1... "

lerobot-record \
    "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" \
    --dataset.repo_id="$DATASET" \
    --dataset.no_stamp=true \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes="$N" \
    --dataset.episode_time_s="$EPISODE_S" \
    --dataset.reset_time_s="$RESET_S" \
    --dataset.push_to_hub=$PUSH \
    --dataset.private=true \
    --resume=$RESUME \
    "${VIEW_ARGS[@]}"

echo
echo "Saved to $DATASET_DIR"

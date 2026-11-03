# Shared hardware/dataset settings for record.sh, train.sh, eval.sh. Sourced, not run.
# Ports/cameras use stable by-id paths; change them to yours (see docs/hardware.md), or set them in the environment.

# Leave any active venv and use the LeRobot conda env.
deactivate 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV:-so_arm101}

FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E089183-if00}
LEADER_PORT=${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D046103-if00}
FOLLOWER_ID=${FOLLOWER_ID:-neon_green_follower_arm}   # calibration name
LEADER_ID=${LEADER_ID:-cobalt_leader_arm}

WRIST_CAM=${WRIST_CAM:-/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0}
FRONT_CAM=${FRONT_CAM:-/dev/v4l/by-id/usb-Web_Camera_Web_Camera_240729163401-video-index0}
# MJPG: raw YUYV from two cameras on a shared USB hub caused corrupted frames (2026-10-05).
CAMERAS="{ front: {type: opencv, index_or_path: $FRONT_CAM, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30, fourcc: MJPG}}"

HF_USER=${HF_USER:-lichtphyz}   # your Hugging Face user name
# Stage 1 task. Change TASK_NAME + TASK for later stages (each stage = its own dataset).
TASK_NAME=${TASK_NAME:-block_to_bin}
TASK=${TASK:-"Put the block in the bin."}
DATASET=$HF_USER/so101_$TASK_NAME
DATASET_DIR=$HOME/.cache/huggingface/lerobot/$DATASET

ROBOT_ARGS=(--robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID --robot.cameras="$CAMERAS")
TELEOP_ARGS=(--teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID)

check_hw() {
    local missing=0
    for dev in "$FOLLOWER_PORT:follower arm" "$LEADER_PORT:leader arm" "$FRONT_CAM:front camera" "$WRIST_CAM:wrist camera"; do
        [ -e "${dev%%:*}" ] || { echo "MISSING: ${dev##*:} (${dev%%:*})"; missing=1; }
    done
    return $missing
}

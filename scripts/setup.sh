#!/usr/bin/env bash
# Idempotent setup/health check for recording on the arm (laptop side). For a training-only GPU machine, use env/setup_env.sh.
# Safe to re-run any time — creates what's missing, reports status for the rest.
set -uo pipefail

REPO_DIR="${LEROBOT_DIR:-$HOME/lerobot}"   # your LeRobot clone
CONDA_ENV="so_arm101"

pass() { echo "  [OK]   $1"; }
warn() { echo "  [WARN] $1"; }
fail() { echo "  [FAIL] $1"; }

echo "== SO-101 environment setup/check =="
echo "Repo: $REPO_DIR"

# --- conda env -------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
    pass "conda env '$CONDA_ENV' exists"
else
    echo "Creating conda env '$CONDA_ENV' (python 3.12)..."
    conda create -y -n "$CONDA_ENV" python=3.12
fi

conda activate "$CONDA_ENV"

# --- lerobot + feetech extra ------------------------------------------------
if python -c "import lerobot" 2>/dev/null; then
    pass "lerobot importable ($(python -c 'import lerobot; print(lerobot.__version__)' 2>/dev/null))"
else
    echo "Installing lerobot[feetech] from $REPO_DIR (editable)..."
    pip install -e "${REPO_DIR}[feetech]"
fi

# --- git lfs -----------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    (cd "$REPO_DIR" && git lfs install --local >/dev/null 2>&1 && pass "git lfs installed for repo") \
        || warn "git lfs install failed — run 'git lfs install' manually in $REPO_DIR"
else
    warn "$REPO_DIR is not a git repo — skipping git lfs setup"
fi

# --- Hugging Face auth --------------------------------------------------------
if hf auth whoami >/dev/null 2>&1; then
    pass "Hugging Face auth: logged in as $(hf auth whoami 2>/dev/null | head -1)"
else
    warn "Not logged in to Hugging Face — run 'hf auth login' before recording/pushing datasets"
fi

# --- GPU / CUDA ---------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    pass "nvidia-smi OK: $GPU_NAME"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        pass "torch sees CUDA"
    else
        warn "torch does NOT see CUDA even though nvidia-smi works — check torch/CUDA build match"
    fi
else
    fail "nvidia-smi cannot reach the driver — training will silently fall back to CPU"
    echo "         A GPU may be present in hardware (check: lspci | grep -i nvidia) but the driver"
    echo "         isn't loaded/working. Common fixes: reboot after a driver install, or reinstall"
    echo "         the NVIDIA driver package for this kernel."
fi

# --- Serial ports (robot arms) ------------------------------------------------
PORTS=(/dev/ttyACM*)
if [ -e "${PORTS[0]}" ]; then
    pass "serial port(s) found: ${PORTS[*]}"
else
    warn "no /dev/ttyACM* ports found — plug in the leader/follower arms"
fi

if groups | grep -qw dialout; then
    pass "user is in 'dialout' group (serial port access OK)"
else
    warn "user is NOT in 'dialout' group — run: sudo usermod -aG dialout \$USER (then log out/in)"
fi

# --- Cameras -------------------------------------------------------------------
CAMS=(/dev/video*)
if [ -e "${CAMS[0]}" ]; then
    pass "camera device(s) found: ${CAMS[*]}"
else
    warn "no /dev/video* devices found — plug in a camera before recording"
fi

# --- Calibration status ---------------------------------------------------------
CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration"
FOLLOWER_CALIB=$(find "$CALIB_DIR/robots/so_follower" -name '*.json' ! -name 'None.json' 2>/dev/null)
LEADER_CALIB=$(find "$CALIB_DIR/teleoperators/so_leader" -name '*.json' ! -name 'None.json' 2>/dev/null)

if [ -n "$FOLLOWER_CALIB" ]; then
    pass "follower calibration found: $(basename "$FOLLOWER_CALIB" .json)"
else
    warn "no named follower calibration found — run lerobot-calibrate --robot.type=so101_follower"
fi

if [ -n "$LEADER_CALIB" ]; then
    pass "leader calibration found: $(basename "$LEADER_CALIB" .json)"
else
    warn "no named leader calibration found — run lerobot-calibrate --teleop.type=so101_leader"
fi

echo "== Done =="

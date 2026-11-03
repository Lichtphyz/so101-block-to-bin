#!/usr/bin/env bash
# One-time setup of a training-only LeRobot environment on a GPU machine (used on an RTX 4080 desktop).
# Same LeRobot commit, conda env name and package versions as the machine that drives the arm.
# Safe to re-run: every step checks first and skips what is already done.
# Does NOT install system software (drivers, conda, gh) — it tells you what's missing instead.
#
# Usage (from the repo root):  bash env/setup_env.sh      # LEROBOT_DIR=<path> to put the LeRobot clone elsewhere
set -uo pipefail

LEROBOT_COMMIT=1e3a158e1395db7e5ac7639f993902ed85748a57
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/lerobot}"
CONDA_ENV=so_arm101
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$HERE/requirements-lock.txt"
DATASET=lichtphyz/so101_block_to_bin_v3

pass() { echo "  [OK]   $*"; }
fail() { echo "  [FAIL] $*"; exit 1; }
info() { echo "  [..]   $*"; }

echo "== 1. Platform"
if grep -qi microsoft /proc/version 2>/dev/null; then
    info "Running inside WSL2 (driver comes from Windows; do not install a Linux NVIDIA driver here)"
fi
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. Ubuntu: sudo ubuntu-drivers install && reboot. WSL2: install the NVIDIA driver on Windows."
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
[ "${DRV%%.*}" -ge 580 ] || fail "Driver $DRV is too old: torch 2.11 (CUDA 13) needs driver >= 580."
pass "$GPU, driver $DRV"
command -v git >/dev/null || fail "git missing: sudo apt install git"
command -v conda >/dev/null || fail "conda missing. Install Miniforge: https://github.com/conda-forge/miniforge (then open a new terminal)"
pass "git and conda present"

echo "== 2. LeRobot source at commit ${LEROBOT_COMMIT:0:8}"
if [ ! -d "$LEROBOT_DIR/.git" ]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR" || fail "clone failed"
fi
cd "$LEROBOT_DIR"
if [ "$(git rev-parse HEAD)" != "$LEROBOT_COMMIT" ]; then
    [ -z "$(git status --porcelain)" ] || fail "$LEROBOT_DIR has local changes; refusing to switch commits."
    git fetch origin && git checkout --detach "$LEROBOT_COMMIT" || fail "checkout failed"
fi
pass "$LEROBOT_DIR @ $(git rev-parse --short HEAD)"

echo "== 3. Conda env '$CONDA_ENV' (Python 3.12 + ffmpeg)"
eval "$(conda shell.bash hook)"
# conda-forge only, like the laptop's Miniforge. With Anaconda, '-c conda-forge' alone still lets 'defaults' win:
# its ffmpeg pulls in a libtiff that breaks torchcodec (undefined symbol jpeg12_write_raw_data).
CF=(--override-channels -c conda-forge)
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" "${CF[@]}" python=3.12 "ffmpeg=8.0.1" || fail "conda create failed"
fi
conda activate "$CONDA_ENV" || fail "could not activate $CONDA_ENV"
# torchcodec decodes the dataset videos through ffmpeg's shared libraries; the laptop uses conda-forge ffmpeg 8.0.1.
if ! conda list ffmpeg | grep -q '^ffmpeg'; then
    conda install -y "${CF[@]}" "ffmpeg=8.0.1" || fail "ffmpeg install failed"
fi
NONCF=$(conda list --show-channel-urls | awk '!/^#/ && $NF!="conda-forge" && $NF!="pypi" {print $1}' | xargs)
[ -z "$NONCF" ] || fail "env $CONDA_ENV has non-conda-forge packages ($NONCF). Remove it (conda env remove -n $CONDA_ENV) and re-run."
pass "python $(python -V | cut -d' ' -f2), ffmpeg $(conda list ffmpeg | awk '/^ffmpeg/{print $2}')"

echo "== 4. LeRobot + training/diffusion extras (versions pinned to the laptop)"
if ! python -c "import lerobot, diffusers, torchcodec" 2>/dev/null; then
    pip install -c "$LOCK" -e "$LEROBOT_DIR[training,diffusion,smolvla]" || fail "pip install failed"
fi
python - <<'EOF' || fail "torch/CUDA check failed"
import torch, lerobot
assert torch.cuda.is_available(), "CUDA not available to torch"
print(f"  [OK]   torch {torch.__version__}, lerobot {lerobot.__version__}, GPU {torch.cuda.get_device_name(0)}")
EOF

echo "== 5. Hugging Face login"
if hf auth whoami >/dev/null 2>&1; then
    pass "logged in as $(hf auth whoami 2>/dev/null | sed -n 's/.*user=//p' | head -1)"
else
    echo "  [TODO] Run: hf auth login   (token with WRITE access, from https://huggingface.co/settings/tokens), then re-run this script."
    exit 0
fi

echo "== 6. Download + load the training dataset ($DATASET)"
python - "$DATASET" <<'EOF' || fail "dataset download/load failed"
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1])
item = ds[ds.num_frames // 2]
print(f"  [OK]   {ds.num_episodes} episodes, {ds.num_frames} frames; decoded front {tuple(item['observation.images.front'].shape)}")
EOF

echo
echo "Setup complete. Next: the smoke test in docs/training.md."

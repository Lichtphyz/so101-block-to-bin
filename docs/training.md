# Training and evaluation

The commands behind the three public models. Training ran on an RTX 4080 desktop (16 GB) that has no arm attached;
recording and evaluation ran on an RTX 4070 laptop connected to the arm. Data and models move between the two through
the Hugging Face Hub.

All commands use plain LeRobot CLIs (`lerobot-train`, `lerobot-rollout`) at LeRobot commit `1e3a158e`, which
[`env/setup_env.sh`](../env/setup_env.sh) installs.

## 1. Environment (training machine)

Prerequisites: NVIDIA driver ≥ 580 (torch 2.11 / CUDA 13), `git`, and conda ([Miniforge](https://github.com/conda-forge/miniforge)).

```bash
hf auth login            # a token with WRITE access, if you want to push models
bash env/setup_env.sh    # re-runnable; LEROBOT_DIR=<path> to put the LeRobot clone elsewhere (default ~/lerobot)
```

The script clones LeRobot at `1e3a158e`, creates the conda env `so_arm101` (Python 3.12 + conda-forge ffmpeg 8.0.1,
needed for video decoding), installs `lerobot[training,diffusion,smolvla]` pinned by
[`env/requirements-lock.txt`](../env/requirements-lock.txt), checks CUDA, and test-loads the training dataset.

- The env must come from **conda-forge only**. With Anaconda's `defaults` channel, its ffmpeg pulls in a libtiff that
  breaks torchcodec (`undefined symbol jpeg12_write_raw_data`). The script refuses to continue if it finds one.
- Works under WSL2 too (install the NVIDIA driver on the Windows side only).

## 2. Training data

[`lichtphyz/so101_block_to_bin_v3`](https://huggingface.co/datasets/lichtphyz/so101_block_to_bin_v3): 125 teleoperated
episodes (55,677 frames, 30 fps), front and wrist cameras at 640×480:

| Part | Episodes | Notes |
|---|---|---|
| Block-to-bin demonstrations | 100 | The first 50 varied the grasp; the second 50 always used a grid-aligned block and a hover-then-descend approach, 40% of them with a shorter block |
| Idle: empty table, the arm stays at rest | 15 | 12 s each: empty bin, blocks already in the bin, or hands passing through |
| Recovery: deliberate miss, regrasp, bin | 10 | Teaches retrying after a missed grasp |

Each part was recorded with leader–follower teleoperation (`lerobot-record`) as its own dataset and then merged into
one. LeRobot's `lerobot-edit-dataset --operation.type merge` does the merge.
[`scripts/record.sh`](../scripts/record.sh) wraps `lerobot-record` with this setup's ports and cameras.

## 3. Smoke test (about 5 minutes)

Check memory and speed before committing to an overnight run:

```bash
conda activate so_arm101
lerobot-train \
  --policy.type=diffusion \
  --policy.resize_shape='[240,320]' \
  --policy.crop_ratio=0.95 \
  --dataset.repo_id=lichtphyz/so101_block_to_bin_v3 \
  --dataset.image_transforms.enable=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --num_workers=8 \
  --steps=400 \
  --log_freq=100 \
  --output_dir=outputs/train/smoke_b32 \
  --job_name=smoke_b32 \
  --wandb.enable=false
```

Watch `nvidia-smi -l 2` in a second terminal. In the log, `updt_s` is GPU time per step and `data_s` is time spent
waiting for data; steps/s ≈ 1 / (updt_s + data_s). Size `--steps` for the real run from that.

Measured on the RTX 4080 (Ryzen 7 7800X3D): 8.8 GB, 46 samples/s, so 60k steps at batch 32 take about 11 h.
The GPU sat idle about two thirds of the time. The bottleneck was **image augmentation**, not video decoding: per
640×480 frame, hue costs 20.5 ms and sharpness 16.3 ms, while brightness/contrast/saturation cost ≈ 0.3 ms.
`--num_workers=12` was slower than 8.

## 4. Diffusion Policy → [`diffusion_v3_desktop`](https://huggingface.co/lichtphyz/diffusion_v3_desktop) (8/10)

```bash
lerobot-train \
  --policy.type=diffusion \
  --policy.resize_shape='[240,320]' \
  --policy.crop_ratio=0.95 \
  --dataset.repo_id=lichtphyz/so101_block_to_bin_v3 \
  --dataset.image_transforms.enable=true \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=<your_hf_user>/diffusion_v3_desktop \
  --batch_size=32 \
  --num_workers=8 \
  --steps=60000 \
  --save_freq=10000 \
  --output_dir=outputs/train/diffusion_v3_desktop \
  --job_name=diffusion_v3_desktop \
  --wandb.enable=false
```

About 11 h. The learning rate follows a cosine schedule, so set `--steps` to what will actually finish: a checkpoint
from partway through a longer run is less finished than a run planned for that length.

**Ablation, [`diffusion_v3_desktop_skip_hue_sharpness`](https://huggingface.co/lichtphyz/diffusion_v3_desktop_skip_hue_sharpness)**
(not evaluated on the arm): the same command without the two slow transforms, about 4.5 h. Nested keys like
`--dataset.image_transforms.tfs.hue.weight=0` are rejected by the CLI, so pass the whole transform dict:

```bash
TFS='{"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"affine":{"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
# then the command above plus:  --dataset.image_transforms.tfs="$TFS"
```

## 5. ACT → [`act_v3_aug_desktop`](https://huggingface.co/lichtphyz/act_v3_aug_desktop) (9/10)

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lichtphyz/so101_block_to_bin_v3 \
  --dataset.image_transforms.enable=true \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=<your_hf_user>/act_v3_aug_desktop \
  --batch_size=8 \
  --num_workers=8 \
  --steps=80000 \
  --save_freq=10000 \
  --output_dir=outputs/train/act_v3_aug_desktop \
  --job_name=act_v3_aug_desktop \
  --wandb.enable=false
```

About 2.5 h (8.9 steps/s, 3.7 GB). ACT uses a constant learning rate, so every checkpoint is a valid stopping point.

## 6. SmolVLA → [`smolvla_v3_desktop`](https://huggingface.co/lichtphyz/smolvla_v3_desktop) (7/10)

Fine-tunes the pretrained [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (450M parameters,
vision encoder frozen, about 100M trained). The base model expects cameras named `camera1..3`, so `--rename_map` maps
front → camera1 and wrist → camera2 (SmolVLA skips the missing camera3).

```bash
F='"observation.images.front": "observation.images.camera1"'
W='"observation.images.wrist": "observation.images.camera2"'
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --rename_map="{$F, $W}" \
  --dataset.repo_id=lichtphyz/so101_block_to_bin_v3 \
  --dataset.image_transforms.enable=true \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=<your_hf_user>/smolvla_v3_desktop \
  --batch_size=32 \
  --num_workers=8 \
  --steps=40000 \
  --save_freq=10000 \
  --output_dir=outputs/train/smolvla_v3_desktop \
  --job_name=smolvla_v3_desktop \
  --wandb.enable=false
```

About 5 h (2.2 steps/s, 7.9 GB; batch 64 did not fit in 16 GB). 40k steps at batch 32 is the same number of samples as
the official recipe (20k at batch 64). This was a first attempt with the default recipe, not tuned for this task.

## 7. Resuming

Any run can be resumed after an interruption:

```bash
lerobot-train --resume=true \
  --config_path=outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json
```

## 8. Running a model on the arm

All three evaluations used `lerobot-rollout` with episodic recording (10 × 30 s, 15 s resets, the same 10 block
placements), so each run is saved as a dataset you can review. Cameras and ports as in
[`scripts/common.sh`](../scripts/common.sh); the model-specific run-time flags are what matters:

| Model | Run-time flags | Why |
|---|---|---|
| ACT | `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1` | The default executes each 100-step chunk (3.3 s) open-loop and missed grasps often. Re-planning every frame and averaging overlapping chunks fixed it. |
| Diffusion | `--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=5` | 5 DDIM steps instead of 100 DDPM steps: fast enough to plan without pauses. |
| SmolVLA | `--inference.type=rtc` plus the same `--rename_map` as in training | With synchronous inference the loop stalled on every 50-action chunk (≈9.5 Hz instead of 30 Hz) and the arm jerked. |

Example (ACT, recorded as an evaluation dataset):

```bash
source scripts/common.sh
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=lichtphyz/act_v3_aug_desktop \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.n_action_steps=1 \
  "${ROBOT_ARGS[@]}" \
  --dataset.repo_id=$HF_USER/rollout_act_v3_aug \
  --dataset.no_stamp=true \
  --dataset.single_task="Put the block in the bin." \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false
```

Rollout dataset names must start with `rollout_` in this LeRobot version. Keep a hand near the follower's power switch
for the first runs.

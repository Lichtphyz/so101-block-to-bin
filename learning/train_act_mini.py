"""Exercise 2 & 3: short ACT training run on a public SO-101 dataset (no robot needed).

Adapted from LeRobot's examples/tutorial/act/act_training_example.py: CUDA, an 8GB-friendly batch size,
loss logged to CSV, and a switch to turn off the ImageNet-pretrained ResNet18 (exercise 3).

    conda activate so_arm101
    python learning/train_act_mini.py --steps 500                      # pretrained backbone (default)
    python learning/train_act_mini.py --steps 500 --no-pretrained      # backbone from random init
    python learning/plot_losses.py                                     # compare the two runs

This is a learning loop, not the real thing. For actual training use `lerobot-train` (it adds
checkpointing, LR scheduling, data augmentation, wandb, resume, etc.).
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--no-pretrained", action="store_true", help="random-init ResNet18 instead of ImageNet weights")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = "act_scratch_backbone" if args.no_pretrained else "act_imagenet_backbone"
out_dir = Path("learning/outputs") / run_name
out_dir.mkdir(parents=True, exist_ok=True)

# Inputs/outputs come straight from the dataset's feature list: images + joint state in, joint actions out.
meta = LeRobotDatasetMetadata(args.repo_id)
features = dataset_to_policy_features(meta.features)
output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {k: ft for k, ft in features.items() if k not in output_features}

cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    pretrained_backbone_weights=None if args.no_pretrained else "ResNet18_Weights.IMAGENET1K_V1",
)
policy = ACTPolicy(cfg).to(device).train()
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

n_params = sum(p.numel() for p in policy.parameters())
n_backbone = sum(p.numel() for n, p in policy.named_parameters() if n.startswith("model.backbone"))
print(f"{run_name}: {n_params / 1e6:.1f}M params ({n_backbone / 1e6:.1f}M in ResNet18 backbone), device={device}")

# Ask the dataset for a chunk of future actions per sample (ACT's action chunking).
delta_timestamps = {"action": [i / meta.fps for i in cfg.action_delta_indices]}
dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)
optimizer = cfg.get_optimizer_preset().build(policy.parameters())

log = open(out_dir / "loss.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["step", "loss", "l1_loss", "kld_loss"])

step, t0 = 0, time.perf_counter()
while step < args.steps:
    for batch in loader:
        batch = preprocessor(batch)
        loss, info = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.writerow([step, loss.item(), info["l1_loss"], info.get("kld_loss", "")])
        if step % 25 == 0:
            rate = (step + 1) / (time.perf_counter() - t0)
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            print(f"step {step:5d}  loss {loss.item():.3f}  l1 {info['l1_loss']:.3f}  {rate:.1f} it/s  peak GPU {mem:.1f} GB")
        step += 1
        if step >= args.steps:
            break
log.close()

policy.save_pretrained(out_dir)
preprocessor.save_pretrained(out_dir)
postprocessor.save_pretrained(out_dir)
elapsed = time.perf_counter() - t0
print(f"done: {args.steps} steps in {elapsed / 60:.1f} min -> {out_dir}")
print(f"extrapolated: 50k steps ~ {elapsed / args.steps * 50_000 / 3600:.1f} h at batch {args.batch_size}")

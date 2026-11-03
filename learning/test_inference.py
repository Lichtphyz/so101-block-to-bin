"""Load a trained ACT checkpoint and run it the way the robot loop does (select_action per frame),
on one episode of the dataset: measures per-step latency and compares predicted vs demonstrated actions.

    python learning/test_inference.py learning/outputs/act_imagenet_backbone
    python learning/test_inference.py outputs/train/<run>/checkpoints/last/pretrained_model
"""

import argparse
import time

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
parser.add_argument("--episode", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=args.checkpoint,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
policy.reset()

ds = LeRobotDataset(args.repo_id, episodes=[args.episode])
keys = ["observation.state", *policy.config.image_features]

latencies, errors = [], []
for i in range(len(ds)):
    frame = ds[i]
    obs = {k: frame[k].unsqueeze(0) for k in keys}
    obs["task"] = [frame["task"]]
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(obs)))
    if device.type == "cuda":
        torch.cuda.synchronize()
    latencies.append((time.perf_counter() - t0) * 1e3)
    errors.append((action.squeeze(0).cpu() - frame["action"]).abs())

lat = torch.tensor(latencies)
err = torch.stack(errors)
chunk = policy.config.n_action_steps
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print(f"episode {args.episode}: {len(ds)} frames, n_action_steps={chunk}")
print(f"latency: median {lat.median():.1f} ms | inference steps (every {chunk}th) median "
      f"{lat[::chunk].median():.1f} ms, max {lat.max():.1f} ms | budget at 30 fps = 33 ms/frame")
print("mean |predicted - demonstrated| action per joint (calibrated units):")
for n, e in zip(names, err.mean(0), strict=True):
    print(f"  {n:14s} {e:6.2f}")
print(f"  {'overall':14s} {err.mean():6.2f}")

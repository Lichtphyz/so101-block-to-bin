"""Exercise 1: look inside a public SO-101 LeRobot dataset.

Shows what one training sample looks like (the same shape your own recordings will have),
and saves the camera images from one frame as JPEGs so you can see what the policy sees.

    conda activate so_arm101
    python learning/explore_dataset.py                      # default dataset
    python learning/explore_dataset.py --repo-id <user>/<dataset> --frame 500
"""

import argparse
from pathlib import Path

from PIL import Image

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
parser.add_argument("--frame", type=int, default=0, help="global frame index to inspect")
parser.add_argument("--out", default="learning/outputs/frames")
args = parser.parse_args()

# --- Metadata: cheap, no video download ---
meta = LeRobotDatasetMetadata(args.repo_id)
print(f"\n=== {args.repo_id} ===")
print(f"robot_type={meta.robot_type}  fps={meta.fps}  episodes={meta.total_episodes}  frames={meta.total_frames}")
print(f"tasks: {list(meta.tasks.index) if hasattr(meta.tasks, 'index') else meta.tasks}")

print("\n--- features (what's stored per frame) ---")
for key, ft in meta.features.items():
    print(f"  {key:32s} dtype={ft['dtype']:8s} shape={tuple(ft['shape'])}  names={ft.get('names')}")

print("\n--- normalization stats for the joint state (used by the preprocessor) ---")
for stat in ("mean", "std", "min", "max"):
    print(f"  {stat:4s}: {[round(float(v), 1) for v in meta.stats['observation.state'][stat]]}")

# --- One sample, requested the way ACT requests it: current obs + the next 100 actions ---
chunk = 100
delta_timestamps = {"action": [i / meta.fps for i in range(chunk)]}
ds = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
sample = ds[args.frame]

print(f"\n--- sample #{args.frame} (episode {int(sample['episode_index'])}, frame {int(sample['frame_index'])}) ---")
for key, val in sample.items():
    shape = tuple(val.shape) if hasattr(val, "shape") else ""
    print(f"  {key:32s} {type(val).__name__:8s} {shape}")
print(f"\n  observation.state      = {[round(float(v), 1) for v in sample['observation.state']]}")
print(f"  action[0]  (now)       = {[round(float(v), 1) for v in sample['action'][0]]}")
print(f"  action[99] (+{chunk / meta.fps:.1f}s)     = {[round(float(v), 1) for v in sample['action'][-1]]}")
print(f"  action_is_pad (chunk runs past episode end): {int(sample['action_is_pad'].sum())} of {chunk} padded")

# --- Save camera images (C,H,W float 0..1 -> JPEG) ---
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for key in meta.camera_keys:
    img = (sample[key].permute(1, 2, 0).numpy() * 255).astype("uint8")
    path = out / f"frame{args.frame}_{key.split('.')[-1]}.jpg"
    Image.fromarray(img).save(path)
    print(f"  saved {path}")

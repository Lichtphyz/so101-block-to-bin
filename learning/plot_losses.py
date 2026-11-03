"""Exercise 3: compare loss curves of the pretrained-backbone vs scratch-backbone ACT runs.

    python learning/plot_losses.py   ->  learning/outputs/loss_compare.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {
    "ResNet18 ImageNet-pretrained": Path("learning/outputs/act_imagenet_backbone/loss.csv"),
    "ResNet18 random init": Path("learning/outputs/act_scratch_backbone/loss.csv"),
}


def smooth(xs, k=25):
    return [sum(xs[max(0, i - k + 1) : i + 1]) / len(xs[max(0, i - k + 1) : i + 1]) for i in range(len(xs))]


fig, ax = plt.subplots(figsize=(8, 4.5))
for label, path in runs.items():
    if not path.exists():
        print(f"skip (not run yet): {path}")
        continue
    rows = list(csv.DictReader(open(path)))
    steps = [int(r["step"]) for r in rows]
    l1 = [float(r["l1_loss"]) for r in rows]
    ax.plot(steps, smooth(l1), label=label)

ax.set_xlabel("training step")
ax.set_ylabel("L1 action loss (normalized, 25-step moving avg)")
ax.set_title("ACT on lerobot/svla_so101_pickplace: effect of pretrained vision backbone")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = Path("learning/outputs/loss_compare.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")

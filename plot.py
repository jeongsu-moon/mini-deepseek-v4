"""Overlay loss curves from one or more out/<name>/log.json files.

Usage:
  python plot.py out/small/log.json
  python plot.py out/a/log.json out/b/log.json --labels baseline moe --out cmp.png

Per ROADMAP §5 protocol: rename each run's log into its own folder and overlay
before/after to compare swaps under identical data+seed.
"""
from __future__ import annotations

import argparse
import json
import os


def _load(path: str):
    with open(path) as f:
        blob = json.load(f)
    hist = blob["metrics"]["history"] if "metrics" in blob else blob["history"]
    steps = [h["step"] for h in hist]
    return steps, [h["train_loss"] for h in hist], [h["val_loss"] for h in hist]


def main():
    p = argparse.ArgumentParser(description="overlay loss curves")
    p.add_argument("logs", nargs="+")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", default="out/loss_curves.png")
    a = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed: pip install matplotlib")

    labels = a.labels or [os.path.basename(os.path.dirname(p)) or p for p in a.logs]
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, lab in zip(a.logs, labels):
        steps, tr, val = _load(path)
        line, = ax.plot(steps, val, marker="o", label=f"{lab} (val)")
        ax.plot(steps, tr, linestyle="--", alpha=0.5, color=line.get_color(),
                label=f"{lab} (train)")
    ax.set(xlabel="step", ylabel="loss (nats/char)", title="loss curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

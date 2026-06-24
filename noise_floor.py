"""Step 1 (A): empirical noise floor (ROADMAP.md §1).

Replaces the fatal "1 run = 1 data point" assumption. Trains the baseline across
multiple seeds on TWO independent axes:
  - init-seed sub-sweep : vary init_seed, fix data_seed  (init / RNG variance)
  - data-seed sub-sweep : fix init_seed, vary data_seed   (sampling-order variance)

Reports, per axis and combined, the val-loss sigma_seed and two thresholds that
every later step's claims must clear:
  - per-run real-effect threshold   = 2 * sigma_seed
  - minimal detectable mean diff(n) ~ 2 * sigma_seed / sqrt(n)

Default uses the 'small' preset so 5+ seeds finish fast; pass --config gpu3090 for
the real ~85M floor (budget a day for >=5 seeds, per the roadmap).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats

from config import get_config, clone
from train import train_once


def _summary(label: str, losses: list[float]) -> dict:
    n = len(losses)
    mean = stats.fmean(losses)
    sd = stats.stdev(losses) if n > 1 else float("nan")
    ci95 = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "label": label, "n": n, "losses": losses,
        "mean": mean, "std": sd,
        "ci95_halfwidth": ci95,
        "per_run_real_effect_threshold_2sigma": 2 * sd,
        "min_detectable_mean_diff_2sigma_over_sqrt_n": 2 * sd / math.sqrt(n) if n > 1 else float("nan"),
    }


def run(config_name: str, n_seeds: int, base_seed: int, data_path, out_dir, max_steps):
    print(f"=== noise floor: config={config_name} n_seeds={n_seeds} ===\n")

    init_losses, data_losses = [], []

    print("--- init-seed sub-sweep (vary init_seed, fix data_seed=0) ---")
    for s in range(base_seed, base_seed + n_seeds):
        cfg = clone(get_config(config_name))
        if data_path: cfg.data_path = data_path
        if out_dir: cfg.out_dir = out_dir
        if max_steps: cfg.train.max_steps = max_steps
        cfg.init_seed, cfg.data_seed = s, 0
        m = train_once(cfg, log=True, verbose=False)
        init_losses.append(m["final_val_loss"])
        print(f"  init_seed={s:2d}  val={m['final_val_loss']:.4f}  ({m['wall_clock_s']:.0f}s)")

    print("\n--- data-seed sub-sweep (fix init_seed=0, vary data_seed) ---")
    for s in range(base_seed, base_seed + n_seeds):
        cfg = clone(get_config(config_name))
        if data_path: cfg.data_path = data_path
        if out_dir: cfg.out_dir = out_dir
        if max_steps: cfg.train.max_steps = max_steps
        cfg.init_seed, cfg.data_seed = 0, s
        m = train_once(cfg, log=True, verbose=False)
        data_losses.append(m["final_val_loss"])
        print(f"  data_seed={s:2d}  val={m['final_val_loss']:.4f}  ({m['wall_clock_s']:.0f}s)")

    report = {
        "config": config_name,
        "init_seed_sweep": _summary("init_seed", init_losses),
        "data_seed_sweep": _summary("data_seed", data_losses),
        "combined": _summary("combined", init_losses + data_losses),
    }

    out = out_dir or get_config(config_name).out_dir
    os.makedirs(os.path.join(out, config_name), exist_ok=True)
    path = os.path.join(out, config_name, "noise_floor.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== NOISE FLOOR REPORT ===")
    for key in ("init_seed_sweep", "data_seed_sweep", "combined"):
        s = report[key]
        print(f"\n[{s['label']}] n={s['n']}  mean={s['mean']:.4f}  sigma_seed={s['std']:.4f}")
        print(f"    95% CI half-width        : {s['ci95_halfwidth']:.4f}")
        print(f"    real-effect threshold 2σ : {s['per_run_real_effect_threshold_2sigma']:.4f}")
        print(f"    min detectable Δmean      : {s['min_detectable_mean_diff_2sigma_over_sqrt_n']:.4f}")
    print(f"\nwrote {path}")
    print("\nGate: if sigma_seed is so large no component effect can clear 2σ, declare the\n"
          "toy scale UNDERPOWERED and raise model/token budget before continuing (ROADMAP §1).")
    return report


def main():
    p = argparse.ArgumentParser(description="empirical noise floor (Step 1A)")
    p.add_argument("--config", default="small")
    p.add_argument("--n_seeds", type=int, default=5, help=">=5 recommended (ROADMAP §1)")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--data_path", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    a = p.parse_args()
    run(a.config, a.n_seeds, a.base_seed, a.data_path, a.out_dir, a.max_steps)


if __name__ == "__main__":
    main()

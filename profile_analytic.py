"""Step 1 (C): analytic system profile — NO weights loaded (ROADMAP.md §1 / §3a).

A 3090 cannot hold V4-Pro (1.6T ~ 3.2TB) or V4-Flash (284B ~ 568GB), and cannot run
a 1M-token context. So the "top-down" system budget is computed ANALYTICALLY from the
verified config (DeepSeek_V4.pdf §4.2.1), not measured. This reproduces the SHAPE of
the report's headline (V4-Pro @1M: 27% single-token FLOPs / 10% KV cache vs V3.2).

FIRST-ORDER MODEL — explicit simplifications (refine against PDF Fig.1 in Step 3a):
  * Per-KV-entry storage bytes and per-entry attention cost are held CONSTANT across
    models, so the curves isolate the mechanism that actually changes: sequence
    compression (1/m, 1/m') + top-k sparsity. Absolute cross-model numbers also depend
    on head dims / MLA latent / params, which this model deliberately holds fixed.
  * "single-token FLOPs" = cost of attention for ONE new decode query at context L.
  * indexer cost uses indexer dim c_I; core-attention cost uses d_attn = n_h * c.
Counts (entries) and per-entry costs are reported in abstract units; only RATIOS vs
V3.2 are meaningful.
"""
from __future__ import annotations

import argparse
import csv

# --- verified configs (DeepSeek_V4.pdf §4.2.1; V3.2 from arXiv:2512.02556) ---
# d_attn = n_h * c  (core attention cost unit) ; c_I = indexer query heads * dim
V4_PRO = dict(
    name="V4-Pro", layers=61, bootstrap=("hca", "hca"),
    n_h=128, c=512, c_I=64 * 128, m=4, m_prime=128, window=128, k_csa=1024,
)
V4_FLASH = dict(
    name="V4-Flash", layers=43, bootstrap=("sliding", "sliding"),
    n_h=64, c=512, c_I=64 * 128, m=4, m_prime=128, window=128, k_csa=512,
)
# V3.2: DeepSeek Sparse Attention — raw KV kept, indexer scans all L, attend top-k=2048.
V3_2 = dict(
    name="V3.2", layers=61, mode="dsa",
    n_h=128, c=512, c_I=64 * 128, k_dsa=2048, window=0,
)

SEQ_LENS = [4_096, 16_384, 65_536, 131_072, 262_144, 524_288, 1_048_576]


def _layer_schedule(cfg) -> list[str]:
    """Per-layer attention mode. Bootstrap first 2, then interleave CSA/HCA."""
    if cfg.get("mode") == "dsa":
        return ["dsa"] * cfg["layers"]
    modes = list(cfg["bootstrap"])
    for i in range(cfg["layers"] - 2):
        modes.append("csa" if i % 2 == 0 else "hca")
    return modes


def _layer_cost(mode: str, L: int, cfg) -> tuple[float, float]:
    """Return (kv_entries, single_token_attn_flop_units) for one layer at context L."""
    d_attn, c_I, w = cfg["n_h"] * cfg["c"], cfg["c_I"], cfg.get("window", 0)
    if mode == "dsa":                              # V3.2: raw KV, indexer over all L
        kv = L
        flops = L * c_I + min(L, cfg["k_dsa"]) * d_attn
    elif mode == "csa":                            # compress 1/m, top-k over compressed + window
        comp = L / cfg["m"]
        kv = comp + w
        flops = comp * c_I + min(comp, cfg["k_csa"]) * d_attn + w * d_attn
    elif mode == "hca":                            # compress 1/m', dense over compressed + window
        comp = L / cfg["m_prime"]
        kv = comp + w
        flops = comp * d_attn + w * d_attn
    elif mode == "sliding":                        # bootstrap: local window only
        kv = w
        flops = w * d_attn
    else:
        raise ValueError(mode)
    return kv, flops


def model_totals(cfg, L: int) -> tuple[float, float]:
    sched = _layer_schedule(cfg)
    kv = sum(_layer_cost(m, L, cfg)[0] for m in sched)
    flops = sum(_layer_cost(m, L, cfg)[1] for m in sched)
    return kv, flops


def run(plot_path: str | None, csv_path: str | None):
    rows = []
    print("Analytic single-token attention FLOPs & accumulated KV (abstract units).")
    print("Only ratios vs V3.2 are meaningful — see module docstring.\n")
    header = f"{'seqlen':>10} | {'V3.2 FLOP':>12} {'Pro FLOP':>12} {'Flash FLOP':>12} | " \
             f"{'Pro/ V3.2':>9} {'Flash/V3.2':>10} || {'Pro KV%':>8} {'Flash KV%':>9}"
    print(header)
    print("-" * len(header))
    for L in SEQ_LENS:
        kv3, f3 = model_totals(V3_2, L)
        kvp, fp = model_totals(V4_PRO, L)
        kvf, ff = model_totals(V4_FLASH, L)
        row = dict(seqlen=L,
                   flop_v32=f3, flop_pro=fp, flop_flash=ff,
                   kv_v32=kv3, kv_pro=kvp, kv_flash=kvf,
                   flop_ratio_pro=fp / f3, flop_ratio_flash=ff / f3,
                   kv_ratio_pro=kvp / kv3, kv_ratio_flash=kvf / kv3)
        rows.append(row)
        print(f"{L:>10} | {f3:>12.3e} {fp:>12.3e} {ff:>12.3e} | "
              f"{fp/f3:>8.1%} {ff/f3:>9.1%} || {kvp/kv3:>7.1%} {kvf/kv3:>8.1%}")

    last = rows[-1]
    print(f"\n@1M context (vs V3.2):")
    print(f"  V4-Pro   : {last['flop_ratio_pro']:.0%} FLOPs / {last['kv_ratio_pro']:.0%} KV"
          f"   (report: 27% / 10%)")
    print(f"  V4-Flash : {last['flop_ratio_flash']:.0%} FLOPs / {last['kv_ratio_flash']:.0%} KV"
          f"   (report: 10% / 7%)")
    print("\nFirst-order model -> expect the SHAPE, not exact %. Refine per-entry "
          "byte/dim accounting against PDF Fig.1 in Step 3a.")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {csv_path}")

    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not installed -> skipping plot")
            return rows
        xs = [r["seqlen"] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for key, lab in [("flop_v32", "V3.2"), ("flop_pro", "V4-Pro"), ("flop_flash", "V4-Flash")]:
            a1.plot(xs, [r[key] for r in rows], marker="o", label=lab)
        a1.set(xscale="log", yscale="log", xlabel="context length (tokens)",
               ylabel="single-token attn FLOPs (units)", title="Attention FLOPs vs context")
        a1.legend(); a1.grid(True, which="both", alpha=0.3)
        for key, lab in [("kv_v32", "V3.2"), ("kv_pro", "V4-Pro"), ("kv_flash", "V4-Flash")]:
            a2.plot(xs, [r[key] for r in rows], marker="o", label=lab)
        a2.set(xscale="log", yscale="log", xlabel="context length (tokens)",
               ylabel="accumulated KV entries (units)", title="KV cache vs context")
        a2.legend(); a2.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(plot_path, dpi=120)
        print(f"wrote {plot_path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="analytic KV/FLOPs profile (Step 1C, no weights)")
    p.add_argument("--plot", default="out/analytic_profile.png")
    p.add_argument("--csv", default="out/analytic_profile.csv")
    a = p.parse_args()
    import os
    os.makedirs("out", exist_ok=True)
    run(a.plot, a.csv)


if __name__ == "__main__":
    main()

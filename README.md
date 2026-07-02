# mini-deepseek-v4

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-bf16%20%7C%20SDPA-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.19348-b31b1b.svg)](https://arxiv.org/abs/2606.19348)
[![Roadmap](https://img.shields.io/badge/roadmap-8%20%2F%208%20complete-success.svg)](ROADMAP.md)

> **A measurement-first research project that reproduces DeepSeek-V4's core
> components from scratch in PyTorch, "one mechanism at a time," at toy scale on a
> single RTX 3090.**

The goal is **not to beat a baseline**. It is to reproduce the *qualitative behavior*
of each architectural innovation in the DeepSeek-V4 report
([arXiv:2606.19348](https://arxiv.org/abs/2606.19348)) — CSA/HCA compressed-KV
attention, mHC, DeepSeekMoE deltas, Muon, MTP, FP4 QAT — by deliberately building the
regime in which each mechanism is supposed to matter, and to explain *why* it behaves
that way. Success is not "the swap has lower loss than the baseline"; it is
**"reproducing each mechanism's characteristic observable (expert-load CV, residual
spectral norm, Sinkhorn convergence, KV/FLOPs curves …) and understanding WHY."**

Two design principles:

- **baseline-always-runs** — a standard dense transformer (RMSNorm · RoPE · SwiGLU ·
  causal SDPA) always runs, and a single config flag *dispatches* to a V4 component.
  This keeps every comparison clean: exactly one variable changes.
- **2σ gate** — the "1 run = 1 data point" assumption is discarded. Every claimed
  effect must exceed 2σ of the seed noise floor (σ_seed) before it counts as *real*.

## Status: 8-step roadmap complete

All five V4 mechanisms (CSA/HCA, mHC, DeepSeekMoE, Muon + MTP, FP4 QAT) are implemented
from scratch and **parity-verified** against a pinned `transformers` `deepseek_v4`
reference (`parity.py`, 18 cases, max-abs ≤ 1e-4). The measurement track (`measure.py`)
judges each mechanism's characteristic observable through the 2σ gate, and the closing
attribution ledger (Step 8) reconciles it with the decomposition hypothesis.

**Key finding:** *structural invariants* (mHC non-expansiveness, MoE load balancing, MTP
acceptance) reproduce even at toy scale, whereas *efficiency trade-offs* (CSA/HCA, Muon,
FP4) only pay off in their target regime (long context · many steps · aggressive
quantization) — this is *regime gating*, not a broken implementation (parity is exact).

The full 8-step plan, fact corrections, and per-step methodology live in
**[`ROADMAP.md`](ROADMAP.md)** (in Korean). The report PDF is not included in the repo
for copyright reasons (`.gitignore`'d).

## Install

```bash
pip install -r requirements.txt   # torch / numpy / matplotlib / datasets / tokenizers
# parity cross-checks need a pinned transformers commit (see requirements.txt comments)
```

## Quickstart — baseline training

```bash
python train.py --config small                            # trains instantly; watch the loss drop
python train.py --config gpu3090 --data_path corpus.txt   # ~85M dense baseline on a 24GB 3090
python plot.py out/small/log.json                         # loss curve
python train.py --config gpu3090 --no_compile             # while editing modules
```

## Component dispatch (the swaps)

A config flag branches into `components/`. The baseline (`full` / `mlp` / `standard` /
`adamw`) always runs; each V4 mechanism is enabled by its own flag:

```bash
python train.py --config small
# attn_type = csa|hca   ffn_type = moe   residual_type = mhc   optimizer = muon
```

## Measurement infrastructure

Three tools underpin every claim in the project:

**(A) Noise floor** — discards the "1 run = 1 data point" assumption. Sweeps two axes
(init-seed / data-seed) to extract σ_seed and the 2σ threshold that every later effect
must clear to count as *real*.

```bash
python noise_floor.py --config small   --n_seeds 5   # quick check (minutes)
python noise_floor.py --config gpu3090 --n_seeds 5   # real ~85M floor (a day's budget)
# -> out/<cfg>/noise_floor.json
```

**(B) Parity harness** — verifies each component against an independent reference /
invariant. Baseline primitives (RMSNorm · RoPE · SwiGLU · causal SDPA) check against
hand-derived invariants; Steps 3–7 check each V4 component against a **pinned-commit**
`transformers` `deepseek_v4` submodule at max-abs < 1e-4 (a mismatch may be a library
bug — see ROADMAP §0.1).

```bash
python parity.py   # 18 cases; auto-detects deepseek_v4 if transformers is installed
```

**(C) Analytic system profile** — KV/FLOPs-vs-context curves computed analytically from
config numbers alone (no weights loaded). A 3090 cannot hold V4-Pro's 1.6T/284B params
or run a 1M context, so the report's headline shape (27% FLOPs / 10% KV @1M) is
reproduced by calculation.

```bash
python profile_analytic.py   # -> out/analytic_profile.{csv,png}, prints @1M ratios
```

## Files

```
config.py            presets (small / gpu3090) + swap flags + verified component hypers
model.py             baseline GPT (RMSNorm · RoPE · SwiGLU · SDPA) + dispatch
data.py              char-level dataset (data_seed controls sample order)
train.py             training loop (bf16 · cosine · deterministic) + train_once()
noise_floor.py       Step 1A (2σ gate)
parity.py            Step 1B + Step 3–7 component parity (18 cases)
profile_analytic.py  Step 1C (no weights)
measure.py           Step 3–8 measurement track (observables + 2σ verdicts + ledger)
plot.py              loss-curve overlays
components/          CSA/HCA · MoE · mHC · Muon · MTP · FP4 implementations
```

## Notes on the RTX 3090

- Ampere sm_86 has **bf16 tensor cores**, but **no FP8** (Hopper) or **FP4** (Blackwell),
  so FP4/FP8 in Step 7 is fake-quant (simulation, zero speedup).
- The `gpu3090` ~85M config uses ≈ 1.4 GB for weights + AdamW — 4×+ headroom in 24 GB.
  On OOM, lower batch, then block size.
- Context experiments run to 4k–16k (24 GB); the 1M point comes only from (C)'s analytic
  curve.

## Roadmap progress ([`ROADMAP.md`](ROADMAP.md))

| Step | Content | Status |
|---|---|:--:|
| 1 | Measurement infra · noise floor · system profile | ✅ |
| 2 | Tokenizer char-level → BPE (hard gate) | ✅ |
| 3 | CSA/HCA compressed-KV attention *(long pole)* | ✅ impl + parity (KV analysis ✓ / empirical parity regime-gated) |
| 4 | mHC (Manifold-Constrained Hyper-Connections) | ✅ impl + parity (op-norm = 1.000 non-expansive ✓) |
| 5 | DeepSeekMoE (V3→V4 deltas only) | ✅ impl + parity (load CV 0.63 → 0.24 ✓) |
| 6 | Muon optimizer + MTP | ✅ impl + parity (MTP acceptance 30% ✓ / Muon best-LR toy-null) |
| 7 | FP4 QAT (Ampere = simulation-only) | ✅ impl + parity (PTQ does not collapse → regime gate) |
| 8 | Closing reprofile + attribution ledger | ✅ KV attribution residual reconciled to 0 |

## License

MIT (see [`LICENSE`](LICENSE)). The referenced DeepSeek-V4 report and model
(arXiv:2606.19348) are under their own separate licenses.

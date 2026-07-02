# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mini-deepseek-v4** — a measurement-first research project that reproduces DeepSeek-V4's
architectural innovations (CSA/HCA compressed-KV attention, mHC, DeepSeekMoE deltas, Muon,
MTP, FP4 QAT) one mechanism at a time, at toy scale, from-scratch in PyTorch, on a single
RTX 3090. The goal is **NOT to beat a baseline** — it is to reproduce each mechanism's
*qualitative behavior* (expert load CV, residual spectral norm, Sinkhorn convergence,
KV/FLOPs curves) and explain *why* it behaves that way.

Reference report: DeepSeek-V4 ([arXiv:2606.19348](https://arxiv.org/abs/2606.19348)). The PDF
is **not** in the repo (copyright, `.gitignore`'d). The authoritative project plan, fact
corrections, and per-step methodology live in **`ROADMAP.md`** — read it before implementing
any V4 component.

**All 8 roadmap steps are complete.** The five V4 mechanisms (CSA/HCA, mHC, DeepSeekMoE,
Muon + MTP, FP4 QAT) are implemented from scratch and parity-verified against a pinned
`transformers deepseek_v4` reference (`parity.py`, 18 cases, max-abs ≤ 1e-4). The measurement
track (`measure.py`) judges each mechanism's characteristic observable through the 2σ gate, and
the Step 8 attribution ledger reconciles per-component KV savings with the system profile.

## Commands

```bash
pip install -r requirements.txt        # torch>=2.2, numpy, matplotlib, datasets, tokenizers
                                        # (transformers optional — pinned commit for parity cross-checks)

# Baseline training
python train.py --config small                       # seconds-to-minutes; CPU or GPU; plumbing/CI
python train.py --config gpu3090 --data_path corpus.txt   # ~85M dense baseline on a 24GB 3090
python train.py --config gpu3090 --no_compile        # disable torch.compile while editing modules
python plot.py out/small/log.json                    # overlay loss curves

# BPE corpus + tokenizer (Step 2, hard gate before Steps 3/5/6)
python build_corpus.py                               # stream/assemble the multi-domain corpus
python train_tokenizer.py                            # train the frozen 16k BPE
python tokenize_corpus.py                            # pre-tokenize to memmapped uint16 .bin shards

# Measurement infrastructure (A/B/C) + measurement track
python noise_floor.py --config small --n_seeds 5     # (A) empirical σ_seed + 2σ thresholds
python parity.py                                     # (B) numeric cross-check harness (exit 1 on failure)
python profile_analytic.py                           # (C) analytic KV/FLOPs vs context (no weights loaded)
python measure.py                                    # Steps 3-8 observables + 2σ verdicts + attribution ledger
```

There is no test runner, linter, or `pytest` suite. **`parity.py` is the test suite** — it runs
the baseline numeric cases and `raise SystemExit(1)` on any failure (CI-friendly). New components
are validated by adding cases there.

## Architecture

### The two design axes (this is the whole point — preserve it)

1. **baseline-always-runs** — The standard dense transformer (RMSNorm · RoPE · SwiGLU · causal
   SDPA · AdamW) always runs. A single config flag *dispatches* to a V4 component, so every
   comparison changes exactly one variable. Baseline code never imports `components/`.

2. **2σ gate** — Reject the "1 run = 1 data point" assumption. Every effect claim must clear
   2× the seed noise floor (σ_seed) to count as "real". `noise_floor.py` measures σ_seed on two
   independent axes (`init_seed` = weight init / RNG; `data_seed` = batch sampling order).

### Dispatch mechanism

Config flags select baseline vs. component. Baseline values are listed first and are the default:

| Flag (`config.py`) | Baseline | Swap targets | Built in |
|---|---|---|---|
| `attn_type` | `full` | `csa`, `hca` | `components/attention.py` (Step 3-4) |
| `ffn_type` | `mlp` | `moe` | `components/ffn.py` (Step 5) |
| `residual_type` | `standard` | `mhc` | `components/residual.py` (Step 4) |
| `optimizer` (TrainConfig) | `adamw` | `muon` | `components/muon.py` (Step 6) |

Dispatch happens in `model.py` (`_build_attention`, `_build_ffn`, `Block.__init__`,
`GPT.configure_optimizers`). Each non-baseline branch lazily imports its implemented component.
**To extend or re-verify a mechanism: edit the component, then run (and if needed extend) its
`parity.py` case against the pinned `transformers deepseek_v4` submodule.**

### Key files

- `config.py` — `ModelConfig` / `TrainConfig` / `RunConfig` dataclasses; `small` and `gpu3090`
  presets; swap-flag value sets and validation. Verified V4 hyper-params (n_hc=4, sinkhorn_iters=20,
  csa m=4, hca m'=128, window=128) live here as defaults; toy MoE values (8 experts) are
  deliberately tiny — do NOT use the real 384/256 expert counts.
- `model.py` — Baseline GPT. Primitives (`rms_norm`, `build_rope_cache`, `apply_rope`, `SwiGLU`,
  `CausalSelfAttention`) are written to be importable and numerically checkable in isolation by
  `parity.py`. RMSNorm upcasts bf16/fp16 to fp32 but preserves fp32/fp64 exactly (so parity runs
  at full precision).
- `data.py` — `CharDataset` (Step 1 plumbing / CI) and `BPEDataset` (Step 2). The char→BPE switch
  was a HARD GATE before Steps 3/5/6, since char-level undermines MoE specialization, sparse
  attention, and MTP acceptance (ROADMAP §2); `BPEDataset` reads memmapped uint16 `.bin` shards
  produced by the `build_corpus`/`train_tokenizer`/`tokenize_corpus` pipeline. `data_seed` controls
  sampling order independently of `init_seed`.
- `train.py` — `train_once(cfg)` is import-friendly so `noise_floor.py` drives many seeds
  in-process. bf16 autocast (Ampere), cosine LR + warmup, deterministic (`warn_only`) seeding,
  JSON logs to `out/<name>/log_i{init}_d{data}.json` plus a canonical `log.json`.
- `noise_floor.py` / `parity.py` / `profile_analytic.py` — Step 1 (A)/(B)/(C). `profile_analytic.py`
  computes KV/FLOPs curves analytically from verified config numbers (a 3090 cannot hold the
  1.6T/284B models or a 1M context), reproducing the *shape* of the report's headline ratios.
- `measure.py` — the Steps 3-8 measurement track: runs each mechanism's characteristic observable,
  applies the 2σ verdict, and builds the closing attribution ledger (component → measured
  FLOP/KV share vs the decomposition hypothesis).
- `components/*.py` — the V4 implementations (`attention.py` CSA/HCA, `ffn.py` MoE, `residual.py`
  mHC, `muon.py`, `mtp.py`, `quant.py` FP4). Each docstring is a spec sheet: verified DeepSeek-V4
  values and the pinned-commit cross-check.

## Critical conventions

- **Determinism has two strengths.** Training uses `torch.use_deterministic_algorithms(True,
  warn_only=True)` (stays runnable). `parity.py` uses the STRICT form for exact cross-checks. Both
  set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Parity cross-checks must pin a `transformers` commit.** The `deepseek_v4` module is actively
  churning (the CSA mask had a "mask collapse" bug fixed 2026-05-13). Trust the `.py` source over
  rendered docs (the docs' `mistralai/DeepseekV4-8x7B-v0.1` example is a fake Mixtral-template
  remnant). A mismatch may be the *library's* bug — suspect that first. See ROADMAP §0.1.
- **Don't use V3.2 as a parity target** — only `DeepseekV32Config` was merged (a "hack" PR), not
  a full DSA implementation. The dense baseline reference is `deepseek_v3`.
- **Several "V4 innovations" are inherited, not invented by V4** (DeepSeekMoE, aux-loss-free
  balancing, MTP, Muon's existence). Study them from their real V2/V3/V3.2 sources, not as V4
  firsts. ROADMAP §0.2 lists each.
- **Hardware scope (Ampere sm_86):** bf16 tensor cores yes; FP8 (Hopper) / FP4 (Blackwell) no.
  FP4 in Step 7 is fake-quant simulation only (zero speedup) — measure the QAT-vs-PTQ accuracy
  gap, not throughput. Context experiments cap at ~4k–16k (24GB); 1M is analytic-only.
- **Success criteria are mechanism observables, not loss wins.** When implementing a step, measure
  the behavior the ROADMAP names for it (load CV, spectral norm, Sinkhorn doubly-stochastic
  invariants, acceptance rate), and only claim effects that clear the 2σ noise floor.

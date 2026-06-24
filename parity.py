"""Step 1 (B): parity / cross-check harness (ROADMAP.md §1).

This is the *framework* every later step plugs into: a deterministic numeric
comparator (max-abs / mean-abs diff) plus a CASES registry. Step 1 ships working
cases that check the baseline primitives (RMSNorm, RoPE, SwiGLU, causal SDPA core)
against independent references / invariants — proving the harness end-to-end today.

V4-component cases (CSA, HCA, MoE, mHC) are registered as PENDING: they construct
the stub (NotImplementedError) and will be filled in Steps 3-7, where each is
checked against the transformers `deepseek_v4` submodule at a PINNED commit
(max-abs < 1e-4). Mismatches may be the library's bug, not yours (ROADMAP §0.1).
"""
from __future__ import annotations

import math
import os

import torch

import model as M
from config import ModelConfig


# ---------------------------------------------------------------------------
# Determinism + comparator
# ---------------------------------------------------------------------------
def set_strict_deterministic(seed: int = 0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)        # STRICT (unlike training's warn_only)
    torch.backends.cudnn.benchmark = False


def compare(name: str, a: torch.Tensor, b: torch.Tensor, atol: float) -> dict:
    a, b = a.detach().float(), b.detach().float()
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    return {"name": name, "max_abs": max_abs, "mean_abs": mean_abs,
            "atol": atol, "passed": max_abs <= atol}


# ---------------------------------------------------------------------------
# Baseline cases (real checks that run today)
# ---------------------------------------------------------------------------
def case_rms_norm() -> dict:
    """rms_norm vs an explicit fp64 reference."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, 64, dtype=torch.float64)
    w = torch.randn(64, dtype=torch.float64)
    got = M.rms_norm(x, w, eps=1e-5)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    return compare("rms_norm vs explicit (fp64)", got, ref, atol=1e-10)


def case_rope_consistency() -> dict:
    """apply_rope (uses build_rope_cache) vs an independent angle-based reference."""
    torch.manual_seed(0)
    B, nh, T, hd, theta = 2, 4, 32, 16, 10000.0
    x = torch.randn(B, nh, T, hd, dtype=torch.float64)
    cos, sin = M.build_rope_cache(T, hd, theta, device="cpu", dtype=torch.float64)
    got = M.apply_rope(x, cos, sin)

    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2).double() / hd))
    ang = torch.outer(torch.arange(T).double(), inv_freq)          # (T, hd/2)
    rc, rs = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    ref = torch.cat([x1 * rc - x2 * rs, x1 * rs + x2 * rc], dim=-1)
    return compare("apply_rope vs angle reference (fp64)", got, ref, atol=1e-10)


def case_rope_norm_invariant() -> dict:
    """RoPE is a rotation -> per-(token,head) vector norm must be preserved."""
    torch.manual_seed(0)
    B, nh, T, hd = 2, 4, 32, 16
    x = torch.randn(B, nh, T, hd, dtype=torch.float64)
    cos, sin = M.build_rope_cache(T, hd, 10000.0, device="cpu", dtype=torch.float64)
    y = M.apply_rope(x, cos, sin)
    return compare("rope norm invariant (fp64)", x.norm(dim=-1), y.norm(dim=-1), atol=1e-10)


def case_swiglu() -> dict:
    """SwiGLU module vs explicit formula."""
    torch.manual_seed(0)
    mlp = M.SwiGLU(32, 64).double()
    x = torch.randn(3, 8, 32, dtype=torch.float64)
    got = mlp(x)
    ref = mlp.down(torch.nn.functional.silu(mlp.gate(x)) * mlp.up(x))
    return compare("swiglu vs explicit (fp64)", got, ref, atol=1e-12)


def case_sdpa_core() -> dict:
    """The op the model trusts: causal SDPA vs a naive softmax-attention reference."""
    torch.manual_seed(0)
    B, nh, T, hd = 2, 4, 48, 32
    q = torch.randn(B, nh, T, hd)
    k = torch.randn(B, nh, T, hd)
    v = torch.randn(B, nh, T, hd)
    got = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    ref = att @ v
    return compare("causal SDPA vs naive softmax (fp32)", got, ref, atol=1e-4)


def case_forward_determinism() -> dict:
    """Same seed -> identical logits (strict deterministic mode)."""
    def run():
        set_strict_deterministic(0)
        cfg = ModelConfig(vocab_size=37, n_layer=2, n_head=2, n_embd=32, block_size=16)
        net = M.GPT(cfg)
        idx = torch.randint(0, 37, (2, 16), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            logits, _ = net(idx)
        return logits
    a, b = run(), run()
    return compare("GPT forward determinism", a, b, atol=1e-9)


BASELINE_CASES = [
    case_rms_norm, case_rope_consistency, case_rope_norm_invariant,
    case_swiglu, case_sdpa_core, case_forward_determinism,
]


# ---------------------------------------------------------------------------
# V4-component cases (PENDING — wired now, filled in Steps 3-7)
# ---------------------------------------------------------------------------
PENDING_CASES = {
    "csa":  lambda: ModelConfig(attn_type="csa"),
    "hca":  lambda: ModelConfig(attn_type="hca"),
    "moe":  lambda: ModelConfig(ffn_type="moe"),
    "mhc":  lambda: ModelConfig(residual_type="mhc"),
}


def probe_transformers() -> str:
    try:
        import transformers
        ok = "yes"
        try:
            from transformers.models import deepseek_v4  # noqa: F401
            ok = "deepseek_v4 present"
        except Exception:
            ok = "installed, but deepseek_v4 module NOT importable"
        return f"transformers {transformers.__version__} ({ok})"
    except ImportError:
        return "transformers NOT installed (V4 cross-checks will be skipped)"


def main():
    print("=== parity harness (Step 1B) ===")
    print(f"reference: {probe_transformers()}")
    print("  -> for Steps 3-7, PIN a commit that includes deepseek_v4 and trust the\n"
          "     .py source over rendered docs (Mixtral-template example is fake). ROADMAP §0.1\n")

    set_strict_deterministic(0)
    print("--- baseline cases ---")
    failed = 0
    for fn in BASELINE_CASES:
        try:
            r = fn()
            tag = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                failed += 1
            print(f"  [{tag}] {r['name']:42s} max_abs={r['max_abs']:.2e} (atol {r['atol']:.0e})")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  [ERR ] {fn.__name__}: {e}")

    print("\n--- V4 component cases (pending) ---")
    for key, make_cfg in PENDING_CASES.items():
        try:
            M.GPT(make_cfg()) if key in ("moe", "mhc") else M._build_attention(make_cfg())
            print(f"  [????] {key}: unexpectedly constructed (did you implement it?)")
        except NotImplementedError:
            print(f"  [PEND] {key}: stub raises NotImplementedError (fill in its roadmap step)")
        except Exception as e:                                # noqa: BLE001
            print(f"  [ERR ] {key}: {type(e).__name__}: {e}")

    print(f"\n{'ALL BASELINE CASES PASS' if failed == 0 else f'{failed} FAILURE(S)'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

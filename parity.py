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
# V4 attention parity (Step 3): our CSA/HCA compressors + Lightning Indexer vs the
# transformers deepseek_v4 reference (pinned 9ded3dbbfc). Copy the reference's
# weights into our module, feed identical inputs, require max-abs < 1e-4. These run
# only when deepseek_v4 is importable (otherwise skipped, not failed).
# ---------------------------------------------------------------------------
def _v4_aligned():
    """Tiny DeepseekV4Config + matching ModelConfig + shared random inputs."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    import components.attention as A
    HID, HD, NH, QLORA = 64, 32, 2, 48
    M_CSA, M_HCA, IDX_H, IDX_D, IDX_K = 4, 8, 4, 16, 8
    ref_cfg = DeepseekV4Config(
        vocab_size=256, hidden_size=HID, num_hidden_layers=2, num_attention_heads=NH,
        num_key_value_heads=1, head_dim=HD, q_lora_rank=QLORA, index_n_heads=IDX_H,
        index_head_dim=IDX_D, index_topk=IDX_K, sliding_window=16, partial_rotary_factor=0.5,
        rms_norm_eps=1e-6, max_position_embeddings=512,
        compress_rates={"compressed_sparse_attention": M_CSA, "heavily_compressed_attention": M_HCA},
    )
    mc = ModelConfig(vocab_size=256, n_layer=2, n_head=NH, n_embd=HID, head_dim=HD,
                     rms_eps=1e-6, csa_compress_m=M_CSA, hca_compress_m=M_HCA,
                     index_n_heads=IDX_H, index_head_dim=IDX_D, index_topk=IDX_K,
                     compress_rope_theta=160000.0, partial_rotary_factor=0.5)
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, 64, HID, generator=g)
    q_res = torch.randn(1, 64, QLORA, generator=g)
    return ref_cfg, mc, A, hidden, q_res, torch.arange(64)[None]


def _randomize(mod, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in mod.parameters():
        p.data = torch.randn(p.shape, generator=g)


def _copy(ours, ref, names):
    for n in names:
        ours.get_parameter(n).data.copy_(ref.get_parameter(n))
    ours.rope.inv_freq.copy_(ref.rotary_emb.compress_inv_freq.float())


_POOL_PARAMS = ["kv_proj.weight", "gate_proj.weight", "position_bias", "kv_norm.weight"]


def case_hca_compressor() -> dict:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor
    ref_cfg, mc, A, hidden, q_res, pos = _v4_aligned()
    ref = DeepseekV4HCACompressor(ref_cfg).float().eval(); _randomize(ref)
    ours = A.HCACompressor(mc).float().eval(); _copy(ours, ref, _POOL_PARAMS)
    with torch.no_grad():
        r, _ = ref(hidden, q_res, pos, None, 0)
        o, _ = ours(hidden, pos)
    return compare("HCA compressor vs deepseek_v4", o, r, atol=1e-4)


def case_csa_compressor() -> dict:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor
    ref_cfg, mc, A, hidden, q_res, pos = _v4_aligned()
    ref = DeepseekV4CSACompressor(ref_cfg).float().eval(); _randomize(ref)
    ours = A.CSACompressor(mc).float().eval(); _copy(ours, ref, _POOL_PARAMS)
    with torch.no_grad():
        r, _ = ref(hidden, q_res, pos, None, 0)         # compare pooling (compressed_kv) only
        o, _ = ours(hidden, hidden, pos)
    return compare("CSA compressor vs deepseek_v4", o, r, atol=1e-4)


def case_lightning_indexer() -> dict:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
    ref_cfg, mc, A, hidden, q_res, pos = _v4_aligned()
    ref = DeepseekV4Indexer(ref_cfg).float().eval(); _randomize(ref)
    ours = A.LightningIndexer(mc, q_src_dim=48).float().eval()
    _copy(ours, ref, _POOL_PARAMS + ["q_b_proj.weight", "weights_proj.weight"])
    with torch.no_grad():
        r = ref(hidden, q_res, pos, None, 0)
        o = ours(hidden, q_res, pos, 64 // mc.csa_compress_m)
    frac = (r.sort(-1).values == o.sort(-1).values).float().mean().item()   # exact top-k set match
    return {"name": "Lightning Indexer top-k vs deepseek_v4", "max_abs": 1.0 - frac,
            "mean_abs": 1.0 - frac, "atol": 0.0, "passed": frac == 1.0}


V4_ATTN_CASES = [case_hca_compressor, case_csa_compressor, case_lightning_indexer]


# ---------------------------------------------------------------------------
# V4 residual parity (Step 4): our mHC HyperConnection vs the transformers
# deepseek_v4 reference. Weight-copy fn/base/scale, feed identical streams, require
# post/comb/collapsed max-abs < 1e-4 AND the doubly-stochastic invariant on comb
# (row & col sums = 1 +/- 1e-5 after the 20-iter Sinkhorn-Knopp projection).
# ---------------------------------------------------------------------------
def case_hyper_connection() -> dict:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
    import components.residual as R
    HID, HC, ITERS, EPS = 64, 4, 20, 1e-6
    ref_cfg = DeepseekV4Config(hidden_size=HID, hc_mult=HC, hc_sinkhorn_iters=ITERS,
                               hc_eps=EPS, rms_norm_eps=1e-6, num_hidden_layers=1)
    mc = ModelConfig(n_embd=HID, n_head=2, n_hc=HC, sinkhorn_iters=ITERS, hc_eps=EPS, rms_eps=1e-6)
    ref = DeepseekV4HyperConnection(ref_cfg).float().eval(); _randomize(ref)
    ours = R.HyperConnection(mc, sinkhorn_iters=ITERS).float().eval()
    for n in ("fn", "base", "scale"):
        ours.get_parameter(n).data.copy_(ref.get_parameter(n))

    g = torch.Generator().manual_seed(0)
    streams = torch.randn(1, 32, HC, HID, generator=g)
    with torch.no_grad():
        r_post, r_comb, r_coll = ref(streams)
        o_post, o_comb, o_coll = ours(streams)

    got = torch.cat([o_post.flatten(), o_comb.flatten(), o_coll.flatten()])
    ref_t = torch.cat([r_post.flatten(), r_comb.flatten(), r_coll.flatten()])
    res = compare("mHC HyperConnection vs deepseek_v4", got, ref_t, atol=1e-4)

    # Doubly-stochastic invariant (Birkhoff manifold): row & col sums -> 1. Checked on a
    # FRESH module at the real init scale (std 0.02), not the std-1 _randomize weights used
    # for the weight-copy above — those make the logits pathologically peaky so 20 Sinkhorn
    # iters can't converge to 1e-5 (the reference can't either, hence parity still matches).
    fresh = R.HyperConnection(mc, sinkhorn_iters=ITERS).float().eval()
    with torch.no_grad():
        _, c, _ = fresh(streams)
    ds = max((c.sum(-1) - 1).abs().max().item(), (c.sum(-2) - 1).abs().max().item())
    res["name"] = f"mHC (post/comb/collapsed) vs deepseek_v4 [doubly-stoch {ds:.0e}]"
    res["passed"] = res["passed"] and ds < 1e-5
    return res


V4_RESIDUAL_CASES = [case_hyper_connection]


# ---------------------------------------------------------------------------
# V4 MoE parity (Step 5): our DeepSeekMoE block (router + clamped-SwiGLU experts +
# shared expert) vs the transformers DeepseekV4SparseMoeBlock. Weight-copy the gate /
# experts / shared weights (and the hash table / bias buffers), feed identical input,
# require max-abs < 1e-4. Covers both layer kinds: learned top-k and frozen hash.
# ---------------------------------------------------------------------------
def _moe_aligned(layer_idx: int):
    """Tiny DeepseekV4Config + matching ModelConfig for one MoE layer (hash if layer_idx<3)."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    import components.ffn as Fmod
    HID, NEXP, TOPK, INTER, VOCAB = 64, 8, 2, 32, 50
    ref_cfg = DeepseekV4Config(
        vocab_size=VOCAB, hidden_size=HID, num_hidden_layers=4, n_routed_experts=NEXP,
        num_experts_per_tok=TOPK, n_shared_experts=1, moe_intermediate_size=INTER,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, norm_topk_prob=True,
        swiglu_limit=10.0, hidden_act="silu", mlp_bias=False, rms_norm_eps=1e-6)
    mc = ModelConfig(vocab_size=VOCAB, n_embd=HID, n_head=2, n_routed_experts=NEXP,
                     n_active_experts=TOPK, n_shared_experts=1, moe_intermediate_size=INTER,
                     scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, norm_topk_prob=True,
                     swiglu_limit=10.0, n_hash_layers=3, rms_eps=1e-6)
    return ref_cfg, mc, Fmod, HID, NEXP, TOPK, VOCAB


def _copy_moe(ours, ref):
    ours.gate.weight.data.copy_(ref.gate.weight)
    ours.experts.gate_up_proj.data.copy_(ref.experts.gate_up_proj)
    ours.experts.down_proj.data.copy_(ref.experts.down_proj)
    ours.shared_experts.gate.weight.data.copy_(ref.shared_experts.gate_proj.weight)
    ours.shared_experts.up.weight.data.copy_(ref.shared_experts.up_proj.weight)
    ours.shared_experts.down.weight.data.copy_(ref.shared_experts.down_proj.weight)


def case_moe_topk() -> dict:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
    ref_cfg, mc, Fmod, HID, NEXP, TOPK, VOCAB = _moe_aligned(layer_idx=3)
    ref = DeepseekV4SparseMoeBlock(ref_cfg, layer_idx=3).float().eval(); _randomize(ref)
    ours = Fmod.DeepSeekMoE(mc, layer_idx=3).float().eval(); _copy_moe(ours, ref)
    # exercise the aux-loss-free path: same non-zero selection bias in both (selection only).
    bias = torch.randn(NEXP, generator=torch.Generator().manual_seed(7))
    ref.gate.e_score_correction_bias.copy_(bias); ours.gate.e_score_correction_bias.copy_(bias)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 16, HID, generator=g)
    with torch.no_grad():
        r = ref(x); o = ours(x)
    return compare("MoE top-k block vs deepseek_v4", o, r, atol=1e-4)


def case_moe_hash() -> dict:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
    ref_cfg, mc, Fmod, HID, NEXP, TOPK, VOCAB = _moe_aligned(layer_idx=0)
    ref = DeepseekV4SparseMoeBlock(ref_cfg, layer_idx=0).float().eval(); _randomize(ref)
    ours = Fmod.DeepSeekMoE(mc, layer_idx=0).float().eval(); _copy_moe(ours, ref)
    table = torch.randint(0, NEXP, (VOCAB, TOPK), generator=torch.Generator().manual_seed(3))
    ref.gate.tid2eid.copy_(table); ours.gate.tid2eid.copy_(table)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 16, HID, generator=g)
    ids = torch.randint(0, VOCAB, (2, 16), generator=g)
    with torch.no_grad():
        r = ref(x, ids); o = ours(x, ids)
    return compare("MoE hash block vs deepseek_v4", o, r, atol=1e-4)


V4_MOE_CASES = [case_moe_topk, case_moe_hash]


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

    print("\n--- V4 attention parity (Step 3: CSA/HCA/indexer vs deepseek_v4) ---")
    if "deepseek_v4 present" in probe_transformers():
        for fn in V4_ATTN_CASES:
            try:
                r = fn()
                tag = "PASS" if r["passed"] else "FAIL"
                if not r["passed"]:
                    failed += 1
                print(f"  [{tag}] {r['name']:42s} max_abs={r['max_abs']:.2e} (atol {r['atol']:.0e})")
            except Exception as e:                            # noqa: BLE001
                failed += 1
                print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
    else:
        print("  [SKIP] deepseek_v4 not importable — pin transformers per requirements.txt")

    print("\n--- V4 residual parity (Step 4: mHC HyperConnection vs deepseek_v4) ---")
    if "deepseek_v4 present" in probe_transformers():
        for fn in V4_RESIDUAL_CASES:
            try:
                r = fn()
                tag = "PASS" if r["passed"] else "FAIL"
                if not r["passed"]:
                    failed += 1
                print(f"  [{tag}] {r['name']:42s} max_abs={r['max_abs']:.2e} (atol {r['atol']:.0e})")
            except Exception as e:                            # noqa: BLE001
                failed += 1
                print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
    else:
        print("  [SKIP] deepseek_v4 not importable — pin transformers per requirements.txt")

    print("\n--- V4 MoE parity (Step 5: DeepSeekMoE block vs deepseek_v4) ---")
    if "deepseek_v4 present" in probe_transformers():
        for fn in V4_MOE_CASES:
            try:
                r = fn()
                tag = "PASS" if r["passed"] else "FAIL"
                if not r["passed"]:
                    failed += 1
                print(f"  [{tag}] {r['name']:42s} max_abs={r['max_abs']:.2e} (atol {r['atol']:.0e})")
            except Exception as e:                            # noqa: BLE001
                failed += 1
                print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
    else:
        print("  [SKIP] deepseek_v4 not importable — pin transformers per requirements.txt")

    print(f"\n{'ALL BASELINE CASES PASS' if failed == 0 else f'{failed} FAILURE(S)'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

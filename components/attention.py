"""CSA + HCA compressed-KV attention — Step 3 (ROADMAP.md §3).

Toy from-scratch reproduction of DeepSeek-V4's two compressed-KV attention types,
swappable for the baseline `full` attention via `attn_type` (model.py dispatch).

Design (preserve the "change one variable" axis): vs the baseline these change ONLY
the attention *pattern* — a compressed long-range KV stream + a 128-token sliding
window + a per-head learnable sink, under shared-KV MQA — reusing toy-simple Q/K/V
projections (NOT V4's LoRA q_a/q_b + grouped output, which are orthogonal to the
compression mechanism). The pieces that ARE the mechanism (window pooling, the CSA
2m-overlap two-series scheme, the Lightning Indexer top-k) mirror the reference
exactly so parity.py can weight-copy and cross-check them against transformers
`deepseek_v4` (pinned 9ded3dbbfc) at max-abs < 1e-4.

Reference math (DeepSeek_V4.pdf §2.3; transformers DeepseekV4{HCA,CSA}Compressor /
DeepseekV4Indexer):
  HCA: non-overlapping windows of m'=128 -> one entry each, dense over entries.
  CSA: 2m-overlap windows (net 1/m, m=4) -> Lightning Indexer picks top-k entries.
  Both: + 128 sliding-window local branch + per-head sink; single softmax over
  [local|compressed|sink], sink column dropped (a denominator-only "attend to none").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# V4 primitives: interleaved partial RoPE + weighted RMSNorm
# ---------------------------------------------------------------------------
def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """V4 uses *interleaved* RoPE (pairs of consecutive channels), unlike the
    baseline's half-split. For pairs (x0,x1),(x2,x3),... return (-x1,x0,-x3,x2,...)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_v4_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the *trailing* `rope_dim` channels of x (partial rotary), V4
    interleaved. x: (B, H, T, Dx); cos/sin: (B, T, rope_dim//2) (half-sized, one per
    pair). Leading "nope" channels pass through. Rotation math runs in fp32."""
    cos = cos.repeat_interleave(2, dim=-1)[:, None, :, :]   # (B,1,T,rope_dim)
    sin = sin.repeat_interleave(2, dim=-1)[:, None, :, :]
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (rope.float() * cos + _rotate_half_interleaved(rope).float() * sin).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class V4Rope(nn.Module):
    """Per-call cos/sin for arbitrary integer positions (compressed entries sit at
    irregular positions w*m, so we can't precompute a fixed table). Interleaved =>
    only rope_dim//2 unique frequencies."""
    def __init__(self, rope_dim: int, theta: float):
        super().__init__()
        assert rope_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)   # (rope_dim//2,)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (B, T) int -> cos/sin: (B, T, rope_dim//2)
        freqs = positions[..., None].float() * self.inv_freq[None, None, :]
        return freqs.cos(), freqs.sin()


class V4RMSNorm(nn.Module):
    """Weighted RMSNorm matching DeepseekV3RMSNorm (upcast to fp32, normalize, scale)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def _rope_dim(config: ModelConfig) -> int:
    rd = int(config.head_dim * config.partial_rotary_factor)
    return rd - (rd % 2)            # force even


# ---------------------------------------------------------------------------
# Compressors (mirror the reference for parity)
# ---------------------------------------------------------------------------
class HCACompressor(nn.Module):
    """Non-overlapping window pooling (HCA, §2.3.2). Each window of m'=compress_rate
    tokens -> one entry: C_i = kv_norm( Σ_{j∈win} softmax(Z_j + B)_j ⊙ (W^KV h)_j ),
    then RoPE at absolute position i*m'. Stateless full-sequence forward."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.m = config.hca_compress_m
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.m, self.head_dim))
        self.kv_norm = V4RMSNorm(self.head_dim, config.rms_eps)
        self.rope = V4Rope(_rope_dim(config), config.compress_rope_theta)

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        """hidden: (B,S,n_embd). Returns (compressed_kv (B,1,T,D), block_bias (B,1,S,T))."""
        B, S, _ = hidden.shape
        kv = self.kv_proj(hidden)
        gate = self.gate_proj(hidden)
        usable = (S // self.m) * self.m
        T = usable // self.m
        if T == 0:
            return hidden.new_zeros(B, 1, 0, self.head_dim), None
        kv = kv[:, :usable].view(B, T, self.m, self.head_dim)
        gate = gate[:, :usable].view(B, T, self.m, self.head_dim) + self.position_bias
        w = gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)       # over the m'-token axis
        compressed = self.kv_norm((kv * w).sum(dim=2))                  # (B,T,D)
        positions = (torch.arange(T, device=hidden.device) * self.m)[None].expand(B, -1)
        cos, sin = self.rope(positions)
        compressed = apply_v4_rope(compressed.unsqueeze(1), cos, sin).squeeze(1)   # (B,T,D)
        block_bias = _compressed_causal_bias(position_ids, T, self.m, compressed)
        return compressed.unsqueeze(1), block_bias                     # (B,1,T,D)


def _two_series_pool(chunk_kv, chunk_gate, m, head_dim):
    """CSA 2m-overlap pooling shared by the compressor and the indexer. chunk_* are
    (B, n_win, m, 2*head_dim): Ca=[...,:head_dim] feeds the NEXT window, Cb=[...,head_dim:]
    the CURRENT one. Entry w = softmax-gated combo of window (w-1)'s Ca with window w's Cb
    over 2m slots. Window 0's first half stays zero-kv / -inf-gate (weight 0)."""
    B, n_win = chunk_kv.shape[:2]
    new_kv = chunk_kv.new_zeros((B, n_win, 2 * m, head_dim))
    new_gate = chunk_gate.new_full((B, n_win, 2 * m, head_dim), float("-inf"))
    new_kv[:, :, m:] = chunk_kv[..., head_dim:]                         # Cb -> second half (current)
    new_gate[:, :, m:] = chunk_gate[..., head_dim:]
    if n_win > 1:
        new_kv[:, 1:, :m] = chunk_kv[:, :-1, :, :head_dim]             # Ca(prev) -> first half (w+1)
        new_gate[:, 1:, :m] = chunk_gate[:, :-1, :, :head_dim]
    w = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
    return (new_kv * w).sum(dim=2)                                     # (B, n_win, head_dim)


def _compressed_causal_bias(position_ids, T, m, ref):
    """(B,1,S,T) additive bias: query t may see entry w only if w < (t+1)//m."""
    B, S = position_ids.shape
    entry = torch.arange(T, device=ref.device)
    thresh = (position_ids + 1) // m                                   # (B,S)
    future = entry.view(1, 1, 1, -1) >= thresh.unsqueeze(1).unsqueeze(-1)
    return ref.new_zeros(B, 1, S, T).masked_fill(future, float("-inf"))


class LightningIndexer(nn.Module):
    """Scores queries against a scaled-down compressed-key stream and keeps top-k
    (§2.3.1). I_{t,s} = Σ_h w_{t,h}·ReLU(q^I_{t,h}·k^I_s)·c^{-1/2}. The keys use the
    SAME 2m-overlap pooling as CSA, at index_head_dim. Scoring runs in fp32."""
    def __init__(self, config: ModelConfig, q_src_dim: int):
        super().__init__()
        self.m = config.csa_compress_m
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.weights_scaling = self.n_heads ** -0.5
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.n_embd, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.m, 2 * self.head_dim))
        self.kv_norm = V4RMSNorm(self.head_dim, config.rms_eps)
        self.q_b_proj = nn.Linear(q_src_dim, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.n_embd, self.n_heads, bias=False)
        self.rope = V4Rope(_rope_dim(config), config.compress_rope_theta)

    def forward(self, hidden, q_src, position_ids, T):
        B, S, _ = hidden.shape
        usable = T * self.m
        chunk_kv = self.kv_proj(hidden)[:, :usable].view(B, T, self.m, 2 * self.head_dim)
        chunk_gate = self.gate_proj(hidden)[:, :usable].view(B, T, self.m, 2 * self.head_dim) + self.position_bias
        keys = self.kv_norm(_two_series_pool(chunk_kv, chunk_gate, self.m, self.head_dim))   # (B,T,c)
        positions = (torch.arange(T, device=hidden.device) * self.m)[None].expand(B, -1)
        cos, sin = self.rope(positions)
        keys = apply_v4_rope(keys.unsqueeze(1), cos, sin).squeeze(1)                         # (B,T,c)

        cos_q, sin_q = self.rope(position_ids)
        q = self.q_b_proj(q_src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)     # (B,H,S,c)
        q = apply_v4_rope(q, cos_q, sin_q).transpose(1, 2)                                   # (B,S,H,c)

        scores = F.relu(torch.matmul(q.float(), keys.transpose(-1, -2).float()[:, None]))    # (B,S,H,T)
        scores = scores * self.softmax_scale
        weights = self.weights_proj(hidden).float() * self.weights_scaling                   # (B,S,H)
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)                           # (B,S,T)

        top_k = min(self.topk, T)
        thresh = (position_ids + 1) // self.m                                                # (B,S)
        entry = torch.arange(T, device=hidden.device)
        future = entry.view(1, 1, -1) >= thresh.unsqueeze(-1)                                # (B,S,T)
        index_scores = index_scores.masked_fill(future, float("-inf"))
        idx = index_scores.topk(top_k, dim=-1).indices                                       # (B,S,k)
        invalid = idx >= thresh.unsqueeze(-1)
        return torch.where(invalid, torch.full_like(idx, -1), idx)


class CSACompressor(nn.Module):
    """2m-overlap two-series pooling (CSA, §2.3.1) + Lightning Indexer top-k.
    kv_proj/gate_proj/position_bias project to 2*head_dim (Ca|Cb)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.m = config.csa_compress_m
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.n_embd, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.m, 2 * self.head_dim))
        self.kv_norm = V4RMSNorm(self.head_dim, config.rms_eps)
        self.rope = V4Rope(_rope_dim(config), config.compress_rope_theta)
        self.indexer = LightningIndexer(config, q_src_dim=config.n_embd)

    def forward(self, hidden, q_src, position_ids):
        B, S, _ = hidden.shape
        kv = self.kv_proj(hidden)
        gate = self.gate_proj(hidden)
        usable = (S // self.m) * self.m
        T = usable // self.m
        if T == 0:
            return hidden.new_zeros(B, 1, 0, self.head_dim), None
        chunk_kv = kv[:, :usable].view(B, T, self.m, 2 * self.head_dim)
        chunk_gate = gate[:, :usable].view(B, T, self.m, 2 * self.head_dim) + self.position_bias
        compressed = self.kv_norm(_two_series_pool(chunk_kv, chunk_gate, self.m, self.head_dim))
        positions = (torch.arange(T, device=hidden.device) * self.m)[None].expand(B, -1)
        cos, sin = self.rope(positions)
        compressed = apply_v4_rope(compressed.unsqueeze(1), cos, sin).squeeze(1)   # (B,T,D)

        # Lightning Indexer -> top-k entry indices per query (-1 = invalid/causal-pad).
        top_k = self.indexer(hidden, q_src, position_ids, T)           # (B,S,k)
        valid = top_k >= 0
        safe = torch.where(valid, top_k, torch.full_like(top_k, T))    # park invalid at T
        block_bias = compressed.new_full((B, 1, S, T + 1), float("-inf"))
        block_bias.scatter_(-1, safe.unsqueeze(1), 0.0)                # 0 only at chosen valid entries
        return compressed.unsqueeze(1), block_bias[..., :T]


# ---------------------------------------------------------------------------
# Attention assembly (toy MQA: compressed stream + sliding window + sink)
# ---------------------------------------------------------------------------
def _sliding_causal(S, window, device, dtype):
    """(S,S) additive mask: 0 where j<=i and i-j<window, else -inf."""
    i = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    keep = (j <= i) & (i - j < window)
    return torch.zeros(S, S, device=device, dtype=dtype).masked_fill(~keep, float("-inf"))


class _CompressedAttentionBase(nn.Module):
    """Shared assembly: MQA Q/K/V (K==V, shared single KV head) + a compressed
    long-range branch + a sliding-window local branch + per-head learnable sink,
    fused in a single softmax over [local | compressed | sink]."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.n_embd, self.head_dim, bias=False)       # MQA: 1 KV head
        self.kv_norm = V4RMSNorm(self.head_dim, config.rms_eps)
        self.proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.n_head))
        self.rope = V4Rope(_rope_dim(config), config.compress_rope_theta)

    def _compress(self, x, position_ids):                                        # overridden
        raise NotImplementedError

    def forward(self, x, cos, sin):
        # cos/sin from the baseline (full rope) are ignored: CSA/HCA layers use the
        # compress rope (θ=160000, partial, interleaved) for q/kv too.
        B, S, _ = x.shape
        position_ids = torch.arange(S, device=x.device)[None].expand(B, -1)
        cos_m, sin_m = self.rope(position_ids)

        q = self.q_proj(x).view(B, S, self.n_head, self.head_dim).transpose(1, 2)   # (B,H,S,D)
        q = apply_v4_rope(q, cos_m, sin_m)
        kv = self.kv_norm(self.kv_proj(x)).view(B, S, 1, self.head_dim).transpose(1, 2)  # (B,1,S,D)
        kv = apply_v4_rope(kv, cos_m, sin_m)

        compressed_kv, block_bias = self._compress(x, position_ids)              # (B,1,T,D)
        full_kv = torch.cat([kv, compressed_kv], dim=2)                          # (B,1,S+T,D)
        T = compressed_kv.shape[2]

        # additive mask (B,1,S,S+T): sliding-window causal on local cols, block_bias on compressed.
        local = _sliding_causal(S, self.window, x.device, torch.float32)         # (S,S)
        mask = local[None, None].expand(B, 1, S, S)
        if T > 0:
            comp = block_bias if block_bias is not None else x.new_zeros(B, 1, S, T)
            mask = torch.cat([mask, comp.to(torch.float32)], dim=-1)             # (B,1,S,S+T)

        logits = torch.matmul(q, full_kv.transpose(-1, -2)) * self.scaling       # (B,H,S,S+T)
        logits = logits.float() + mask
        sink = self.sinks.view(1, self.n_head, 1, 1).float().expand(B, self.n_head, S, 1)
        combined = torch.cat([logits, sink], dim=-1)                             # (B,H,S,S+T+1)
        combined = combined - combined.amax(dim=-1, keepdim=True)
        probs = combined.softmax(dim=-1).to(q.dtype)
        attn = torch.matmul(probs[..., :-1], full_kv)                            # drop sink col; (B,H,S,D)

        attn = apply_v4_rope(attn, cos_m, -sin_m)                                # undo V-side rope (K==V)
        attn = attn.transpose(1, 2).reshape(B, S, self.n_head * self.head_dim)
        return self.proj(attn)


class HeavilyCompressedAttention(_CompressedAttentionBase):
    """HCA: dense attention over ALL compressed entries (no indexer/top-k)."""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.compressor = HCACompressor(config)

    def _compress(self, x, position_ids):
        return self.compressor(x, position_ids)


class CompressedSparseAttention(_CompressedAttentionBase):
    """CSA: indexer picks top-k compressed entries per query."""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.compressor = CSACompressor(config)

    def _compress(self, x, position_ids):
        return self.compressor(x, x, position_ids)

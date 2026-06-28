"""Baseline GPT for mini-deepseek-v4: RMSNorm + RoPE + SwiGLU + causal SDPA.

This is the "꼭대기" — the baseline that always runs. Non-baseline config flags
dispatch to components/* stubs that raise NotImplementedError (the swap targets).

The submodules (rms_norm, apply_rope, SwiGLU, CausalSelfAttention) are written so
parity.py can import and numerically check them in isolation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


# ---------------------------------------------------------------------------
# Primitives (also imported by parity.py)
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm. Upcasts low-precision inputs (bf16/fp16) to fp32 for a stable
    reduction, but preserves fp32/fp64 inputs exactly (so parity tests can run
    at full precision)."""
    in_dtype = x.dtype
    compute_dtype = torch.float32 if in_dtype in (torch.float16, torch.bfloat16) else in_dtype
    x = x.to(compute_dtype)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x.to(in_dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32):
    """Returns (cos, sin) each shaped (seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)                       # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding (GPT-NeoX half-split). x: (B, n_head, T, head_dim)."""
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x))."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with RoPE and PyTorch SDPA (flash on Ampere)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.n_embd, 3 * self.n_head * self.head_dim, bias=False)
        self.proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                          # each (B, T, nh, hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (B, nh, T, hd)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.proj(y)


# ---------------------------------------------------------------------------
# Dispatch: build the attention / ffn / residual the config asks for
# ---------------------------------------------------------------------------
def _build_attention(config: ModelConfig) -> nn.Module:
    if config.attn_type == "full":
        return CausalSelfAttention(config)
    if config.attn_type == "csa":
        from components.attention import CompressedSparseAttention
        return CompressedSparseAttention(config)
    if config.attn_type == "hca":
        from components.attention import HeavilyCompressedAttention
        return HeavilyCompressedAttention(config)
    raise ValueError(config.attn_type)


def _build_ffn(config: ModelConfig, layer_idx: int = 0) -> nn.Module:
    if config.ffn_type == "mlp":
        hidden = int(config.mlp_ratio * config.n_embd)
        hidden = 64 * ((hidden + 63) // 64)                  # round to a nice multiple
        return SwiGLU(config.n_embd, hidden)
    if config.ffn_type == "moe":
        from components.ffn import DeepSeekMoE
        return DeepSeekMoE(config, layer_idx)                # layer_idx selects hash vs top-k
    raise ValueError(config.ffn_type)


class Block(nn.Module):
    """Pre-norm transformer block.

    residual_type='standard' is the plain single-stream residual. 'hc'/'mhc' keep the
    residual as n_hc PARALLEL STREAMS ([B,S,H,D]) and mix them in/out of each sublayer
    via a HyperConnection (Step 4, ROADMAP §4); 'mhc' Sinkhorn-projects the stream mixer
    onto the doubly-stochastic manifold, 'hc' (sinkhorn_iters=0) leaves it unconstrained.
    """
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.residual_type = config.residual_type
        self.is_moe = config.ffn_type == "moe"
        self.attn_norm = RMSNorm(config.n_embd, config.rms_eps)
        self.attn = _build_attention(config)
        self.ffn_norm = RMSNorm(config.n_embd, config.rms_eps)
        self.ffn = _build_ffn(config, layer_idx)
        if config.residual_type in ("hc", "mhc"):
            from components.residual import HyperConnection
            iters = config.sinkhorn_iters if config.residual_type == "mhc" else 0
            self.attn_hc = HyperConnection(config, sinkhorn_iters=iters)
            self.ffn_hc = HyperConnection(config, sinkhorn_iters=iters)

    def _apply_ffn(self, h, input_ids):
        # MoE hash layers need the token ids (frozen tid2eid lookup); MLP/top-k ignore them.
        return self.ffn(h, input_ids) if self.is_moe else self.ffn(h)

    def forward(self, x, cos, sin, input_ids=None):
        if self.residual_type == "standard":
            x = x + self.attn(self.attn_norm(x), cos, sin)
            x = x + self._apply_ffn(self.ffn_norm(x), input_ids)
            return x
        return self._hc_forward(x, cos, sin, input_ids)

    def _hc_forward(self, x, cos, sin, input_ids):
        # x: [B, S, H, D] parallel residual streams. Per sublayer: collapse streams (pre),
        # run the sublayer on the single sequence, then place its output back (post) and
        # mix the streams (comb). comb is consumed TRANSPOSED — sum_j comb[j,k]*x[j] — since
        # Sinkhorn yields a doubly-stochastic but non-symmetric matrix (direction matters).
        dtype = x.dtype
        post, comb, collapsed = self.attn_hc(x)
        out = self.attn(self.attn_norm(collapsed), cos, sin)
        x = post.to(dtype).unsqueeze(-1) * out.unsqueeze(-2) \
            + torch.matmul(comb.to(dtype).transpose(-1, -2), x)
        post, comb, collapsed = self.ffn_hc(x)
        out = self._apply_ffn(self.ffn_norm(collapsed), input_ids)
        x = post.to(dtype).unsqueeze(-1) * out.unsqueeze(-2) \
            + torch.matmul(comb.to(dtype).transpose(-1, -2), x)
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.residual_type = config.residual_type
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        if config.residual_type in ("hc", "mhc"):
            from components.residual import HyperHead
            self.hc_head = HyperHead(config)               # collapse streams -> one sequence
        self.mtp_depth = config.mtp_depth
        if config.mtp_depth > 0:
            from components.mtp import MTP
            self.mtp = MTP(config)                         # multi-token prediction (Step 6 +MTP)
        self.norm_f = RMSNorm(config.n_embd, config.rms_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(config.block_size, config.head_dim, config.rope_theta,
                                    device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.grad_checkpoint = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.config.block_size}")
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        emb = self.tok_emb(idx)
        x = emb
        if self.residual_type in ("hc", "mhc"):            # expand into n_hc parallel streams
            x = x.unsqueeze(2).expand(-1, -1, self.config.n_hc, -1).contiguous()
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, cos, sin, idx, use_reentrant=False)
            else:
                x = block(x, cos, sin, idx)
        if self.residual_type in ("hc", "mhc"):            # collapse streams -> [B, T, D]
            x = self.hc_head(x)
        h0 = x                                             # trunk hidden (pre-final-norm) for MTP
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.config.ffn_type == "moe":             # + sequence-wise balance loss (Step 5)
                loss = loss + self._moe_balance_loss(B)
            if self.mtp_depth > 0:                         # + multi-token prediction loss (Step 6)
                loss = loss + self.mtp.loss(emb, h0, idx, cos, sin, self.norm_f, self.lm_head)
        return logits, loss

    def _moe_balance_loss(self, batch_seq: int) -> torch.Tensor:
        """Sum the sequence-wise balance loss over MoE blocks, scaled by balance_loss_coef.
        The aux-loss-free bias controller is separate — it updates inside the routers, not
        through this gradient. Hash layers route statically, so only top-k layers contribute
        a meaningful gradient, but every MoE block reports its term for measurement."""
        total = 0.0
        for block in self.blocks:
            moe = block.ffn
            total = total + moe.balance_loss_coef * moe.balance_loss(batch_seq)
        return total

    @torch.no_grad()
    def _trunk(self, idx):
        """Run the trunk and return (emb, h0, logits, cos, sin) — forward() without the
        loss, reused by self-speculative decoding. h0 is the pre-final-norm hidden."""
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        emb = self.tok_emb(idx)
        x = emb
        if self.residual_type in ("hc", "mhc"):
            x = x.unsqueeze(2).expand(-1, -1, self.config.n_hc, -1).contiguous()
        for block in self.blocks:
            x = block(x, cos, sin, idx)
        if self.residual_type in ("hc", "mhc"):
            x = self.hc_head(x)
        h0 = x
        logits = self.lm_head(self.norm_f(x))
        return emb, h0, logits, cos, sin

    @torch.no_grad()
    def generate_speculative(self, idx, max_new_tokens):
        """Depth-1 self-speculative greedy decoding (Step 6 +MTP). Each step: the trunk emits
        the main greedy token g0, the MTP head DRAFTS the next-next token d1, and one verify
        pass checks d1 against the trunk's own argmax. The emitted sequence is IDENTICAL to
        plain greedy (drafts are only accepted when they match the target's argmax); returns
        (idx, accepted, drafted) so acceptance rate can be measured. Use batch size 1."""
        assert self.mtp_depth > 0, "generate_speculative needs mtp_depth > 0"
        start = idx.shape[1]
        accepted = drafted = 0
        while idx.shape[1] - start < max_new_tokens:
            emb, h0, logits, cos, sin = self._trunk(idx)
            g0 = logits[:, -1].argmax(-1, keepdim=True)               # main greedy next token
            d1 = self.mtp.draft_next(h0[:, -1:], self.tok_emb(g0),
                                     cos[-1:], sin[-1:], self.norm_f, self.lm_head).unsqueeze(1)
            _, _, vlogits, _, _ = self._trunk(torch.cat([idx, g0, d1], dim=1))
            v1 = vlogits[:, -2].argmax(-1, keepdim=True)              # true token following g0
            drafted += 1
            if torch.equal(d1, v1):                                   # draft matched -> 2 tokens
                accepted += 1
                idx = torch.cat([idx, g0, v1], dim=1)
            else:                                                     # reject draft, keep verified
                idx = torch.cat([idx, g0, v1], dim=1)
        return idx, accepted, drafted

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def configure_optimizers(self, train_cfg):
        """AdamW with decay/no-decay groups. optimizer='muon' dispatches to Step-6 stub."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if train_cfg.optimizer == "adamw":
            return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                     betas=(train_cfg.beta1, train_cfg.beta2))
        if train_cfg.optimizer == "muon":
            from components.muon import Muon
            # Route ONLY true 2D matrices (not the embedding/tied head) through Newton-Schulz;
            # 1D scalars, embeddings, and 3D MoE expert stacks stay on AdamW (ROADMAP §6: a
            # mixed routing would corrupt the AdamW-vs-Muon comparison).
            muon_p, adamw_decay, adamw_nodecay = [], [], []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                is_emb_or_head = name.endswith("tok_emb.weight") or name.endswith("lm_head.weight")
                if p.dim() == 2 and not is_emb_or_head:
                    muon_p.append(p)
                elif p.dim() >= 2:
                    adamw_decay.append(p)
                else:
                    adamw_nodecay.append(p)
            return Muon(muon_p, adamw_decay, adamw_nodecay, lr=train_cfg.lr,
                        weight_decay=train_cfg.weight_decay,
                        adamw_betas=(train_cfg.beta1, train_cfg.beta2))
        raise ValueError(train_cfg.optimizer)

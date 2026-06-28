"""MTP — Multi-Token Prediction head — Step 6 deliverable ("+ MTP", ROADMAP.md §6).

V3-IDENTICAL (NOT a V4 invention — ROADMAP §0.2). Each MTP depth k owns a small module
that predicts the token at offset (k+1) so the model can DRAFT several future tokens and
verify them in one pass (self-speculative decoding). Per depth k (DeepSeek-V3 §3.2):

    combined_i = M_k · [ RMSNorm(h^{k-1}_i) ; RMSNorm(emb(t_{i+k})) ]   (concat, 2D -> D)
    h^k        = TransformerBlock_k(combined)
    logits^k_i = SharedHead(RMSNorm_f(h^k_i))            predicts token t_{i+1+k}

The output head and token embedding are SHARED with the main model (passed in); only the
per-depth (norm_h, norm_e, proj, block) are owned here. The MTP block is a plain
standard/MLP block (residual_type='standard', ffn_type='mlp') so MTP composes cleanly with
mHC / MoE on the main trunk without threading streams or input_ids into the draft head
(one-variable-at-a-time; a faithful toy simplification — see ROADMAP §6).

Verification (parity.py): MTP gives a finite training loss, AND greedy self-speculative
decoding yields EXACTLY the same sequence as plain greedy autoregressive decoding (the
correctness invariant of speculative decoding — drafts are accepted only when they match
the target model's own argmax). Acceptance RATE is meaningful only on BPE (Step 2) with a
trained head — that is the measurement track.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class MTPDepth(nn.Module):
    """One MTP depth: combine the previous hidden with the next token's embedding, run a
    transformer block. The shared output head/norm is applied by the caller."""

    def __init__(self, config):
        super().__init__()
        from model import RMSNorm, Block
        d = config.n_embd
        self.norm_h = RMSNorm(d, config.rms_eps)
        self.norm_e = RMSNorm(d, config.rms_eps)
        self.proj = nn.Linear(2 * d, d, bias=False)
        block_cfg = copy.copy(config)              # plain block: no mHC streams, no MoE/input_ids
        block_cfg.residual_type = "standard"
        block_cfg.ffn_type = "mlp"
        self.block = Block(block_cfg, layer_idx=0)

    def forward(self, h_prev, emb_next, cos, sin):
        combined = self.proj(torch.cat([self.norm_h(h_prev), self.norm_e(emb_next)], dim=-1))
        return self.block(combined, cos, sin)


class MTP(nn.Module):
    """Stack of `mtp_depth` MTP modules. Shares the trunk's token embedding, final norm, and
    output head (all passed in at call time) — MTP owns only the per-depth combine+block."""

    def __init__(self, config):
        super().__init__()
        self.depth = config.mtp_depth
        self.loss_coef = config.mtp_loss_coef
        self.depths = nn.ModuleList([MTPDepth(config) for _ in range(self.depth)])

    def loss(self, emb, h0, idx, cos, sin, norm_f, head) -> torch.Tensor:
        """Mean cross-entropy over depths. Depth k consumes emb(t_{i+k}) and predicts
        t_{i+1+k}; sequences are rolled (and the wrapped tail masked via [:valid])."""
        B, T, _ = h0.shape
        V = head.weight.shape[0]
        h_prev, losses = h0, []
        for k in range(1, self.depth + 1):
            emb_next = torch.roll(emb, shifts=-k, dims=1)          # emb_next[:,i] = emb[:,i+k]
            h_k = self.depths[k - 1](h_prev, emb_next, cos, sin)
            logits = head(norm_f(h_k))                             # predicts t_{i+1+k}
            tgt = torch.roll(idx, shifts=-(k + 1), dims=1)
            valid = T - (k + 1)
            if valid > 0:
                losses.append(torch.nn.functional.cross_entropy(
                    logits[:, :valid].reshape(-1, V), tgt[:, :valid].reshape(-1)))
            h_prev = h_k
        if not losses:
            return h0.new_zeros(())
        return self.loss_coef * torch.stack(losses).mean()

    @torch.no_grad()
    def draft_next(self, h_last, emb_next, cos, sin, norm_f, head) -> torch.Tensor:
        """Depth-1 draft: from the last trunk hidden + the just-emitted token's embedding,
        propose the next-next token (greedy argmax). Used by self-speculative decoding."""
        h1 = self.depths[0](h_last, emb_next, cos, sin)
        return head(norm_f(h1))[:, -1].argmax(dim=-1)              # [B]

"""mHC (Manifold-Constrained Hyper-Connections) — Step 4 deliverable (ROADMAP.md §4).

Verified V4 spec (DeepSeek_V4.pdf §3.3 / §4.2.1; transformers DeepseekV4HyperConnection,
pinned 9ded3dbbfc):
  - expands the residual into n_hc=4 PARALLEL STREAMS, shape [B, S, 4, D]
    (this is STREAM mixing, NOT hidden-channel mixing — common misreading).
  - each sublayer site owns one HyperConnection producing a (pre, post, comb) triplet
    from an unweighted-RMSNorm'd flatten of the streams:
        pre  = sigmoid(.) + eps     stream-collapse weights -> one sequence into the sublayer
        post = 2*sigmoid(.)         placement of the sublayer output back across streams
        comb = Sinkhorn(softmax(.)) HxH stream mixer (the residual map B_l)
  - comb is projected onto the doubly-stochastic (Birkhoff) manifold by Sinkhorn-Knopp:
    softmax-positive start, then alternate row/col normalizations for t_max=20 steps.
  - doubly-stochastic => operator 2-norm <= 1 (non-expansive) => stable signal.
  - constrained extension of naive Hyper-Connections (Zhu et al. 2025, ByteDance 2409.19606).

The arms (ROADMAP §4 wants all three, HP-matched, residual scheme the only change):
  - residual_type='standard'  plain single-stream residual (baseline, model.py Block).
  - residual_type='hc'        UNCONSTRAINED HC: sinkhorn_iters=0 -> comb stays row-stochastic
                              (softmax only), no doubly-stochastic projection.
  - residual_type='mhc'       sinkhorn_iters=20 -> comb projected doubly-stochastic.
  The 0-vs-20 toggle is the pre-registered sweep that isolates "the CONSTRAINT causes
  stability" from "the extra parameters cause it" — both arms have identical param counts.

Implementation notes (ROADMAP.md §4):
  - 20 Sinkhorn iters => UNROLLED autograd is correct (no implicit diff needed).
  - keep comb / sinks in fp32 (bf16 self-destabilizes — mirrors transformers' fix):
    the (pre, post, comb) mapping runs in fp32 here regardless of autocast.
  - mHC static bias/gating -> AdamW, not Muon (handled by the optimizer step, not here).
  - measure it on a DELIBERATELY UNSTABLE deep baseline, else there's nothing to fix.

Parity (parity.py case_hyper_connection): weight-copy fn/base/scale from the transformers
DeepseekV4HyperConnection, feed identical streams, require post/comb/collapsed max-abs < 1e-4
and the doubly-stochastic invariant (row/col sums = 1 +/- 1e-5).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with NO learnable scale (mirrors DeepseekV4UnweightedRMSNorm). The
    reduction runs in fp32; the result is cast back to the input dtype."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class HyperConnection(nn.Module):
    """One (pre, post, comb) mixing triplet for a single sublayer site (attn or ffn).

    Mirrors transformers ``DeepseekV4HyperConnection`` (pinned 9ded3dbbfc) EXACTLY when
    ``sinkhorn_iters >= 1`` so parity.py can weight-copy and cross-check. With
    ``sinkhorn_iters == 0`` the doubly-stochastic projection is skipped (comb is left
    row-stochastic, the unconstrained-HC arm). The (pre, post, comb) mapping is always
    computed in fp32 (bf16 self-destabilizes the Sinkhorn projection).
    """

    def __init__(self, config, sinkhorn_iters: int | None = None):
        super().__init__()
        self.hc = config.n_hc
        self.eps = config.hc_eps
        self.rms_eps = config.rms_eps
        self.sinkhorn_iters = config.sinkhorn_iters if sinkhorn_iters is None else sinkhorn_iters
        mix = (2 + self.hc) * self.hc                      # pre(H) + post(H) + comb(H*H)
        self.fn = nn.Parameter(torch.empty(mix, self.hc * config.n_embd))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))          # one learned scale per output
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, streams: torch.Tensor):
        """streams: [B, S, H, D]. Returns (post [B,S,H], comb [B,S,H,H], collapsed [B,S,D])."""
        hc = self.hc
        flat = _unweighted_rms_norm(streams.flatten(start_dim=2).float(), self.rms_eps)
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.eps
        if self.sinkhorn_iters >= 1:
            # Sinkhorn-Knopp: 1 col-norm, then (iters-1) x (row-norm, col-norm). The final
            # op is a col-norm, so columns are exactly ~1 and rows converge to ~1 — the
            # doubly-stochastic (Birkhoff) projection. Matches the reference op-for-op.
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
            for _ in range(self.sinkhorn_iters - 1):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        # else (sinkhorn_iters == 0): unconstrained HC — comb stays merely row-stochastic.

        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
        return post, comb, collapsed


# mHC is a HyperConnection with the doubly-stochastic projection on (sinkhorn_iters>=1).
# Kept as an explicit name for the ROADMAP/spec vocabulary and external imports.
ManifoldConstrainedHyperConnection = HyperConnection


class HyperHead(nn.Module):
    """Final stream-collapse before the shared output RMSNorm (mirrors
    DeepseekV4HyperHead). One more `pre`-style weighted sum over the stream axis,
    [B, S, H, D] -> [B, S, D]. Runs the mapping in fp32."""

    def __init__(self, config):
        super().__init__()
        self.hc = config.n_hc
        self.eps = config.hc_eps
        self.rms_eps = config.rms_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc, self.hc * config.n_embd))
        self.hc_base = nn.Parameter(torch.empty(self.hc))
        self.hc_scale = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.hc_fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = _unweighted_rms_norm(x.flatten(2).float(), self.rms_eps)
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)

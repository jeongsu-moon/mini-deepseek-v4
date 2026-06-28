"""Muon optimizer — Step 6 deliverable (ROADMAP.md §6).

Verified V4 spec (DeepSeek_V4.pdf §4.2.2):
  - hybrid Newton-Schulz orthogonalization of the (2D) momentum matrix:
    10 iterations over 2 stages — first 8 steps (a,b,c)=(3.4445, -4.7750, 2.0315),
    final 2 steps (a,b,c)=(2, -1.5, 0.5). M0 = M / ||M||_F.
  - Nesterov trick; RMS-rescale the update to reuse AdamW hyper-params.
  - 2D weights -> Muon; embeddings / heads / RMSNorm / mHC static bias|gating -> AdamW.
  - NOTE: V4 did NOT invent Muon, nor was it 'first at 1.6T' (Moonlight/Kimi-K2
    predate it). Study the orthogonalization MECHANISM + LR-transfer, not a 'first'.

Methodology (ROADMAP.md §6):
  - unit-test newton_schulz() FIRST (orthogonalize random matrices, singular values
    -> 1) before wrapping the optimizer. parity.py asserts max|svdvals(NS(G)) - 1| small.
  - compare AdamW vs Muon at BEST-LR-vs-BEST-LR (independent LR grids). A shared LR
    grid is the #1 confound — Muon's effective LR scale differs. (measurement track)
"""
import torch


def newton_schulz(G, steps_ab=((3.4445, -4.7750, 2.0315), 8),
                  steps_cd=((2.0, -1.5, 0.5), 2)):
    """Approximately orthogonalize a 2D matrix G -> U V^T (the polar factor) via the
    verified V4 two-stage quintic Newton-Schulz schedule.

    Iterates X <- a*X + (b*A + c*A^2)*X with A = X X^T, starting from X = G/||G||_F.
    The first 8 steps use the aggressive (3.4445, -4.7750, 2.0315) coefficients (fast but
    overshoots), the final 2 use the gentle (2, -1.5, 0.5) fixed-point coefficients that
    pull the singular values onto 1. Runs in fp32. Operates on the fat orientation
    (transpose if rows > cols) so A is the smaller Gram matrix, then transposes back."""
    assert G.ndim == 2, "newton_schulz expects a 2D matrix"
    X = G.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-2, -1)
    X = X / (X.norm() + 1e-7)
    for (a, b, c), n in (steps_ab, steps_cd):
        for _ in range(n):
            A = X @ X.transpose(-2, -1)
            X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Hybrid optimizer: Newton-Schulz-orthogonalized momentum update for 2D matrices,
    standard AdamW for everything else (embeddings, tied head, RMSNorm scales, mHC static
    bias/gating, and 3D MoE expert stacks). One Optimizer with per-group `use_muon`, so the
    train.py cosine scheduler (which writes group['lr']) just works.

    Routing is decided by GPT.configure_optimizers, NOT here: only true 2D weights that are
    not the embedding/head reach the muon group (sending 1D / embedding params through
    Newton-Schulz would corrupt the AdamW-vs-Muon comparison — ROADMAP §6 confound).

    `lr_ratio` lets the AdamW group run at a fraction of the scheduled lr (Muon matrices
    typically want a larger lr); default 1.0 keeps a shared schedule. Per-optimizer best-LR
    tuning is the measurement track, not wired here.
    """

    def __init__(self, muon_params, adamw_decay=(), adamw_nodecay=(), *, lr=3e-4,
                 momentum=0.95, nesterov=True, weight_decay=0.0,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_lr_ratio=1.0):
        groups = []
        muon_params = list(muon_params)
        if muon_params:
            groups.append(dict(params=muon_params, use_muon=True, lr=lr, lr_ratio=1.0,
                               momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        if list(adamw_decay):
            groups.append(dict(params=list(adamw_decay), use_muon=False, lr=lr,
                               lr_ratio=adamw_lr_ratio, betas=adamw_betas, eps=adamw_eps,
                               weight_decay=weight_decay))
        if list(adamw_nodecay):
            groups.append(dict(params=list(adamw_nodecay), use_muon=False, lr=lr,
                               lr_ratio=adamw_lr_ratio, betas=adamw_betas, eps=adamw_eps,
                               weight_decay=0.0))
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"] * group["lr_ratio"]
            if group["use_muon"]:
                self._muon_step(group, lr)
            else:
                self._adamw_step(group, lr)
        return loss

    def _muon_step(self, group, lr):
        mu, nesterov, wd = group["momentum"], group["nesterov"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(mu).add_(g)
            update = g.add(buf, alpha=mu) if nesterov else buf      # Nesterov-style lookahead
            update = newton_schulz(update)
            # RMS-rescale by aspect ratio so the orthogonal update's per-element magnitude is
            # comparable across shapes (reuses AdamW-scale LRs — verified V4 note).
            scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
            if wd != 0:
                p.mul_(1 - lr * wd)                                  # decoupled weight decay
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group, lr):
        beta1, beta2 = group["betas"]
        eps, wd = group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bc1 = 1 - beta1 ** state["step"]
            bc2 = 1 - beta2 ** state["step"]
            denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
            if wd != 0:
                p.mul_(1 - lr * wd)                                  # decoupled weight decay
            p.addcdiv_(m, denom, value=-lr / bc1)

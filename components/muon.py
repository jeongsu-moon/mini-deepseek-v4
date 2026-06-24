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
    -> 1) before wrapping the optimizer.
  - compare AdamW vs Muon at BEST-LR-vs-BEST-LR (independent LR grids). A shared LR
    grid is the #1 confound — Muon's effective LR scale differs.
"""
import torch


def newton_schulz(G, steps_ab=((3.4445, -4.7750, 2.0315), 8),
                  steps_cd=((2.0, -1.5, 0.5), 2)):
    """Approximately orthogonalize a 2D matrix G -> U V^T (Step 6).

    The two-stage hybrid schedule above is the verified V4 recipe. Implement and
    unit-test here before building the optimizer.
    """
    raise NotImplementedError(
        "newton_schulz() is a Step 6 deliverable — implement the verified 8+2 step "
        "schedule and unit-test orthogonality (singular values -> 1)."
    )


class Muon(torch.optim.Optimizer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Muon (optimizer='muon') is a Step 6 deliverable — not yet implemented.\n"
            "Build on newton_schulz() per ROADMAP.md §6. Route only 2D weights here;\n"
            "embeddings/heads/scalars stay on AdamW."
        )

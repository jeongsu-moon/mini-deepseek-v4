"""FP4 fake-quantization — Step 7 deliverable (ROADMAP.md §7). SIMULATION-ONLY.

HARDWARE REALITY (read first): an RTX 3090 is Ampere sm_86 — it has NO FP4 (sm_100/
Blackwell) or FP8 (sm_90/Hopper) tensor cores. Every "FP4" path here is FAKE-QUANT
(quantize -> dequantize, all arithmetic still in bf16/fp32). There is ZERO memory or
wall-clock benefit on this card — that is reported as a measured invariant, not hidden.
What this DOES let us study is the *numerics* of FP4 QAT: the E2M1 grid, scale choice,
the straight-through estimator, and how much accuracy QAT recovers over naive PTQ. The
report's "FP4xFP8 ~1/3 efficiency" is a future-hardware citation, not reproducible here.

FP4 = E2M1 (1 sign, 2 exponent, 1 mantissa). Positive representable magnitudes:
    {0, 0.5, 1, 1.5, 2, 3, 4, 6}   (max 6.0)
A per-(expert) scale maps the weight range onto this grid: scale = amax / 6.

Modes (config.quant_mode), applied to the routed-expert weights (the dominant param mass):
  'none' : full precision (bf16/fp32) — the baseline.
  'qat'  : quantization-AWARE training — fake-quant in the forward at train AND eval, with
           a straight-through estimator so gradients flow to the underlying fp weights.
  'ptq'  : post-training quantization — train full precision, fake-quant ONLY at eval
           (self.training is False). No STE needed (no training through the quantizer).

The QAT-recovers-the-PTQ-gap study (seeds>=3, 2sigma) is the measurement track; here we
implement + verify the mechanism (grid snapping, STE identity gradient, mode behaviour).
"""
from __future__ import annotations

import torch

# E2M1 positive grid (magnitudes). Registered lazily per device/dtype in _grid().
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX = 6.0


def _grid(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_E2M1, device=x.device, dtype=x.dtype)


def quantize_fp4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize x to the E2M1 grid under `scale`, then dequantize (round-trip). Values are
    scaled into the grid range, snapped to the nearest representable magnitude (ties-to-grid
    via argmin), sign restored, and rescaled. No STE here — see fp4_fake_quant."""
    g = _grid(x)
    xs = (x / scale)
    mag = xs.abs().clamp(max=FP4_MAX)
    idx = (mag.unsqueeze(-1) - g).abs().argmin(dim=-1)        # nearest grid index
    q = g[idx] * xs.sign()
    return q * scale


def fp4_fake_quant(w: torch.Tensor, ste: bool = True, per_channel: bool = True) -> torch.Tensor:
    """Fake-quantize a weight tensor to FP4 (E2M1). Per-channel scale (amax over all but the
    leading dim — e.g. per-expert for a [E, ...] stack) when per_channel and w is >=2D, else
    per-tensor. With ste=True the forward returns the quantized values but the gradient is the
    identity (straight-through estimator: w + (q - w).detach())."""
    if per_channel and w.dim() >= 2:
        amax = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True)
    else:
        amax = w.detach().abs().amax()
    scale = (amax / FP4_MAX).clamp_min(1e-12)
    q = quantize_fp4(w, scale)
    return w + (q - w).detach() if ste else q

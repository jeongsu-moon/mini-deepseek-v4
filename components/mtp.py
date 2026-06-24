"""MTP (Multi-Token Prediction) heads — Step 6 add-on (ROADMAP.md §6).

Verified V4 spec (DeepSeek_V4.pdf §4.2.1): MTP depth = 1, and the report states it
is "as DeepSeek-V3" — i.e. IDENTICAL to V3. So study V3's MTP (arXiv:2412.19437);
it is NOT a V4 invention. Acceptance / self-speculative decoding is only meaningful
on BPE tokens (char-level inflates acceptance toward chance — Step 2 gate).
"""
import torch.nn as nn


class MTPHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(
            "MTP heads are a Step 6 add-on — not yet implemented.\n"
            "Build per ROADMAP.md §6 (V3-identical, depth=1); measure speculative\n"
            "acceptance + tokens/sec on BPE, ablate on char-level to show dependence."
        )

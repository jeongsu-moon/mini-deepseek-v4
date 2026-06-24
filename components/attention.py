"""CSA + HCA — Step 3-4 deliverables (ROADMAP.md §3).

Verified V4 spec (DeepSeek_V4.pdf §4.2.1):
  CSA: 2m-overlap token compression -> net seq length 1/m (m=4); Lightning Indexer
       top-k over the *compressed* stream (V4-Pro=1024, V4-Flash=512); plus a
       128-token sliding-window branch with a per-head learnable sink.
  HCA: non-overlapping compression m'=128; *dense* attention over compressed
       entries (no indexer / no top-k); same 128 sliding window + sink.
  Layout: first 2 layers are bootstrap (Pro=HCA, Flash=pure sliding-window),
          the rest interleave CSA/HCA.

Cross-check submodules against transformers `deepseek_v4`:
  'compressed_sparse_attention' / 'heavily_compressed_attention' layer_types
  at a PINNED commit (the CSA mask had a 'mask collapse' bug fixed 2026-05-13 —
  a mismatch may be the library's bug, not yours). See ROADMAP.md §0.1.
"""
import torch.nn as nn

_PENDING = (
    "{name} (attn_type={flag!r}) is a Step 3-4 deliverable — not yet implemented.\n"
    "Build it in components/attention.py per ROADMAP.md §3, then parity-check the\n"
    "submodule against transformers deepseek_v4 (pinned commit) via parity.py."
)


class CompressedSparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(_PENDING.format(name="CSA", flag="csa"))


class HeavilyCompressedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(_PENDING.format(name="HCA", flag="hca"))

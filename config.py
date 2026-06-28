"""Configuration for mini-deepseek-v4.

Baseline = (attn_type='full', ffn_type='mlp', residual_type='standard', optimizer='adamw').
Each non-baseline value dispatches to a component in components/ that raises
NotImplementedError until its roadmap step lands.

Verified DeepSeek-V4 spec constants (from DeepSeek_V4.pdf §4.2.1, arXiv:2606.19348)
live in profile_analytic.py — NOT here, because we never instantiate the real
1.6T / 284B models on a 3090. See ROADMAP.md §0.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

# --- swap-flag value sets (baseline value listed first) ---
ATTN_TYPES = ("full", "csa", "hca")
FFN_TYPES = ("mlp", "moe")
RESIDUAL_TYPES = ("standard", "hc", "mhc")   # 'hc' = unconstrained HC (sinkhorn_iters=0)
OPTIM_TYPES = ("adamw", "muon")
SCORING_FUNCS = ("sqrtsoftplus", "sigmoid")   # V4 delta vs V3 (A/B for the routing study)
DATA_FORMATS = ("char", "bpe")     # char-level (Step 1) vs BPE multi-domain (Step 2+)


@dataclass
class ModelConfig:
    vocab_size: int = 256          # set at runtime from the dataset
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    head_dim: Optional[int] = None  # defaults to n_embd // n_head
    mlp_ratio: float = 8 / 3        # SwiGLU intermediate ≈ mlp_ratio * n_embd
    block_size: int = 1024
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True

    # --- swap flags (dispatch happens in model.py) ---
    attn_type: str = "full"
    ffn_type: str = "mlp"
    residual_type: str = "standard"

    # --- component hyper-params (only used when the flag is non-baseline) ---
    n_hc: int = 4                   # mHC residual streams (verified V4 value: 4)
    sinkhorn_iters: int = 20        # verified V4 value: 20 ('hc' arm forces 0)
    hc_eps: float = 1e-6            # Sinkhorn-Knopp numerical floor (verified V4 value)
    n_routed_experts: int = 8       # toy value (V4-Pro=384 / Flash=256 — do NOT use here)
    n_active_experts: int = 2       # toy value (V4=6); = num_experts_per_tok (top-k)
    n_shared_experts: int = 1
    # MoE (Step 5) — verified V4 deltas. scoring sqrtsoftplus (vs V3 sigmoid); first
    # n_hash_layers MoE layers use FROZEN hash routing; aux-loss-free bias controller
    # (load-based, NOT gradient) + a sequence-wise balance loss. Toy intermediate size.
    moe_intermediate_size: int = 256   # per-expert SwiGLU hidden (V4: 2048)
    scoring_func: str = "sqrtsoftplus"  # 'sqrtsoftplus' (V4) or 'sigmoid' (V3) — A/B knob
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0          # clamped-SwiGLU expert activation bound
    norm_topk_prob: bool = True
    n_hash_layers: int = 3              # first N MoE layers route via frozen tid2eid
    balance_loss_coef: float = 1e-3     # sequence-wise balance loss weight
    bias_update_rate: float = 1e-3      # aux-loss-free controller step (sign rule, no grad)
    # MTP (Step 6, "+MTP") — V3-identical multi-token prediction (NOT a V4 invention).
    # mtp_depth = num_nextn_predict_layers (0 = off); each depth k predicts token t+1+k for
    # self-speculative decoding. Acceptance is meaningful on BPE only (ROADMAP §6).
    mtp_depth: int = 0
    mtp_loss_coef: float = 0.3
    csa_compress_m: int = 4         # verified V4 value: 4
    hca_compress_m: int = 128       # verified V4 value (m'): 128
    sliding_window: int = 128       # verified V4 value (n_win): 128
    # CSA Lightning Indexer (toy values; verified V4: topk Pro=1024/Flash=512,
    # index_n_heads=64, index_head_dim=128). compress_rope_theta/partial_rotary are
    # the verified V4 rope settings for the compressed branches.
    index_topk: int = 64
    index_n_heads: int = 4
    index_head_dim: int = 32
    compress_rope_theta: float = 160000.0
    partial_rotary_factor: float = 0.5   # fraction of head_dim that gets RoPE (V4: 64/512)

    def __post_init__(self):
        if self.attn_type not in ATTN_TYPES:
            raise ValueError(f"attn_type must be one of {ATTN_TYPES}, got {self.attn_type!r}")
        if self.ffn_type not in FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {FFN_TYPES}, got {self.ffn_type!r}")
        if self.residual_type not in RESIDUAL_TYPES:
            raise ValueError(f"residual_type must be one of {RESIDUAL_TYPES}, got {self.residual_type!r}")
        if self.scoring_func not in SCORING_FUNCS:
            raise ValueError(f"scoring_func must be one of {SCORING_FUNCS}, got {self.scoring_func!r}")
        if self.head_dim is None:
            if self.n_embd % self.n_head != 0:
                raise ValueError("n_embd must be divisible by n_head when head_dim is None")
            self.head_dim = self.n_embd // self.n_head


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    min_lr_ratio: float = 0.1       # cosine floor = lr * min_lr_ratio
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 16
    eval_interval: int = 100
    eval_steps: int = 50
    grad_checkpoint: bool = False
    compile: bool = False           # torch.compile; off while editing modules
    amp_dtype: str = "bfloat16"     # Ampere (sm_86) has bf16 tensor cores
    deterministic: bool = True      # warn_only in training; strict in parity.py

    def __post_init__(self):
        if self.optimizer not in OPTIM_TYPES:
            raise ValueError(f"optimizer must be one of {OPTIM_TYPES}, got {self.optimizer!r}")


@dataclass
class RunConfig:
    name: str = "small"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_path: str = "sample.txt"
    data_format: str = "char"       # 'char' -> CharDataset; 'bpe' -> BPEDataset(manifest.json)
    out_dir: str = "out"
    init_seed: int = 0              # seeds weight init + dropout + CUDA RNG
    data_seed: int = 0              # seeds batch sampling order (independent axis)
    device: str = "cuda"

    def __post_init__(self):
        if self.data_format not in DATA_FORMATS:
            raise ValueError(f"data_format must be one of {DATA_FORMATS}, got {self.data_format!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# 'small'   : seconds-to-minutes; CPU or GPU; for plumbing / CI / quick checks.
# 'gpu3090' : ~85M dense baseline; fits a 24GB 3090 with ~4x headroom
#             (weights+AdamW ≈ 1.4GB, activations a few GB). See ROADMAP.md spec note.

def _small() -> RunConfig:
    return RunConfig(
        name="small",
        model=ModelConfig(n_layer=4, n_head=4, n_embd=128, block_size=256),
        train=TrainConfig(max_steps=500, warmup_steps=50, batch_size=32,
                          eval_interval=100, eval_steps=20, compile=False),
        device="cuda",
    )


def _gpu3090() -> RunConfig:
    # n_embd 768 / 12 layers / SwiGLU ≈ 2048  ->  ~85M params (tied embeddings).
    return RunConfig(
        name="gpu3090",
        model=ModelConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024),
        train=TrainConfig(max_steps=3000, warmup_steps=150, batch_size=24,
                          eval_interval=200, eval_steps=50, compile=True,
                          grad_checkpoint=False),
        device="cuda",
    )


def _gpu3090_bpe() -> RunConfig:
    # Step 2: same ~85M dense baseline, but on the BPE multi-domain corpus.
    # vocab_size is set at runtime from data/manifest.json (16k). Retrain the
    # baseline here once the tokenizer is frozen (ROADMAP §2).
    cfg = _gpu3090()
    cfg.name = "gpu3090_bpe"
    cfg.data_path = "data/manifest.json"
    cfg.data_format = "bpe"
    return cfg


_PRESETS = {"small": _small, "gpu3090": _gpu3090, "gpu3090_bpe": _gpu3090_bpe}


def get_config(name: str = "small") -> RunConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(_PRESETS)}")
    return _PRESETS[name]()


def clone(cfg: RunConfig) -> RunConfig:
    return copy.deepcopy(cfg)

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
RESIDUAL_TYPES = ("standard", "mhc")
OPTIM_TYPES = ("adamw", "muon")


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
    sinkhorn_iters: int = 20        # verified V4 value: 20
    n_routed_experts: int = 8       # toy value (V4-Pro=384 / Flash=256 — do NOT use here)
    n_active_experts: int = 2       # toy value (V4=6)
    n_shared_experts: int = 1
    csa_compress_m: int = 4         # verified V4 value: 4
    hca_compress_m: int = 128       # verified V4 value (m'): 128
    sliding_window: int = 128       # verified V4 value (n_win): 128

    def __post_init__(self):
        if self.attn_type not in ATTN_TYPES:
            raise ValueError(f"attn_type must be one of {ATTN_TYPES}, got {self.attn_type!r}")
        if self.ffn_type not in FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {FFN_TYPES}, got {self.ffn_type!r}")
        if self.residual_type not in RESIDUAL_TYPES:
            raise ValueError(f"residual_type must be one of {RESIDUAL_TYPES}, got {self.residual_type!r}")
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
    out_dir: str = "out"
    init_seed: int = 0              # seeds weight init + dropout + CUDA RNG
    data_seed: int = 0              # seeds batch sampling order (independent axis)
    device: str = "cuda"

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


_PRESETS = {"small": _small, "gpu3090": _gpu3090}


def get_config(name: str = "small") -> RunConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(_PRESETS)}")
    return _PRESETS[name]()


def clone(cfg: RunConfig) -> RunConfig:
    return copy.deepcopy(cfg)

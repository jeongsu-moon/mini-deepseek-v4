"""Char-level in-memory dataset with a SEED-CONTROLLED sampling order.

The data-order seed is a separate noise axis from the init seed (ROADMAP.md §1):
init_seed governs weight init / dropout; data_seed governs which random offsets
get sampled each step. noise_floor.py sweeps each axis independently.

Char-level is deliberate for Step 1 plumbing only — it is a HARD GATE to switch to
BPE before Steps 3/5/6 (ROADMAP.md §2), because char-level undermines MoE
specialization, long-range sparse attention, and MTP acceptance.
"""
from __future__ import annotations

import torch


class CharDataset:
    def __init__(self, path: str, block_size: int, split_frac: float = 0.9, device: str = "cuda"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.device = device

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        # each split must be able to yield at least one (block_size+1) window
        for name, d in (("train", self.train_data), ("val", self.val_data)):
            if len(d) <= block_size + 1:
                raise ValueError(
                    f"{name} split has {len(d)} chars <= block_size+1 ({block_size + 1}). "
                    f"Use a larger corpus via --data_path, or the 'small' preset. "
                    f"(bundled sample.txt targets the 'small' preset.)")

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, t: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in t)

    def get_batch(self, split: str, batch_size: int, generator: torch.Generator):
        """Sample a batch. `generator` (CPU) makes the order reproducible per data_seed."""
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        ix = torch.randint(max_start, (batch_size,), generator=generator)
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + 1 + self.block_size] for i in ix])
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

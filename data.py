"""Char-level in-memory dataset with a SEED-CONTROLLED sampling order.

The data-order seed is a separate noise axis from the init seed (ROADMAP.md §1):
init_seed governs weight init / dropout; data_seed governs which random offsets
get sampled each step. noise_floor.py sweeps each axis independently.

Char-level is deliberate for Step 1 plumbing only — it is a HARD GATE to switch to
BPE before Steps 3/5/6 (ROADMAP.md §2), because char-level undermines MoE
specialization, long-range sparse attention, and MTP acceptance.
"""
from __future__ import annotations

import json

import numpy as np
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


class BPEDataset:
    """Multi-domain BPE dataset over memmapped uint16 .bin shards (Step 2, ROADMAP §2).

    Replaces char-level once Steps 3/5/6 need real subword structure. Each domain's
    pre-tokenized shard (tokenize_corpus.py) is split 90/10; get_batch samples a domain
    (equal probability by default, which REBALANCES the unequal domain sizes — dialogue
    is ~5M vs 12M others) then a random window, preserving the data_seed reproducibility
    contract. Domain ids are tracked so Step 5 can measure per-domain expert routing;
    pass domain= to draw a single-domain batch for those per-domain histograms.
    """

    def __init__(self, manifest_path: str, block_size: int, split_frac: float = 0.9,
                 device: str = "cuda", domain_weights: dict | None = None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        self.vocab_size = man["vocab_size"]
        self.block_size = block_size
        self.device = device
        self.domains = list(man["domains"])
        self.domain_to_id = {d: i for i, d in enumerate(self.domains)}

        self.train_shards, self.val_shards = [], []
        for d in self.domains:
            arr = np.memmap(man["domains"][d]["bin"], dtype=np.uint16, mode="r")
            n = int(split_frac * len(arr))
            for split_name, lo, hi in (("train", 0, n), ("val", n, len(arr))):
                if (hi - lo) <= block_size + 1:
                    raise ValueError(
                        f"domain {d!r} {split_name} split has {hi - lo} tokens "
                        f"<= block_size+1 ({block_size + 1}). Lower block_size or "
                        f"raise --target_tokens in tokenize_corpus.py.")
            self.train_shards.append(arr[:n])
            self.val_shards.append(arr[n:])

        if domain_weights is None:
            w = np.ones(len(self.domains), dtype=np.float64)
        else:
            w = np.asarray([domain_weights.get(d, 0.0) for d in self.domains], dtype=np.float64)
        self.domain_probs = torch.tensor(w / w.sum(), dtype=torch.float)

    def get_batch(self, split: str, batch_size: int, generator: torch.Generator,
                  domain: str | None = None):
        """Sample a batch -> (x, y), drop-in compatible with CharDataset. `generator`
        (CPU) makes domain choice + offsets reproducible per data_seed. domain=<name>
        forces a single-domain batch (Step-5 per-domain routing probes know the domain
        because they pass it in)."""
        shards = self.train_shards if split == "train" else self.val_shards
        if domain is not None:
            dom_idx = torch.full((batch_size,), self.domain_to_id[domain], dtype=torch.long)
        else:
            dom_idx = torch.multinomial(self.domain_probs, batch_size,
                                        replacement=True, generator=generator)

        xs, ys = [], []
        for di in dom_idx.tolist():
            data = shards[di]
            max_start = len(data) - self.block_size - 1
            i = int(torch.randint(max_start, (1,), generator=generator).item())
            chunk = torch.from_numpy(data[i: i + self.block_size + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x, y = torch.stack(xs), torch.stack(ys)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

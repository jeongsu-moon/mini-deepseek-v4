"""Training loop + logging for mini-deepseek-v4.

`train_once(run_cfg)` is import-friendly so noise_floor.py can drive many seeds
in-process. The CLI runs a single config. bf16 autocast (Ampere), cosine LR with
warmup, deterministic (warn_only) seeding, JSON logging to out/<name>/log.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from config import RunConfig, get_config
from data import CharDataset
from model import GPT


def set_seed(seed: int, deterministic: bool = True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only: keep training runnable even if an op lacks a deterministic kernel
        # (parity.py uses the STRICT form for exact cross-checks).
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def cosine_lr(step: int, base_lr: float, warmup: int, max_steps: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * coeff)


@torch.no_grad()
def evaluate(model, dataset, batch_size, steps, generator, autocast_ctx):
    model.eval()
    losses = torch.zeros(steps)
    for i in range(steps):
        x, y = dataset.get_batch("val", batch_size, generator)
        with autocast_ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> falling back to CPU (use --config small).")
        return "cpu"
    return requested


def train_once(cfg: RunConfig, log: bool = True, verbose: bool = True) -> dict:
    """Train one model to completion. Returns a metrics dict (final/best val loss)."""
    device = _resolve_device(cfg.device)
    cfg.device = device
    set_seed(cfg.init_seed, cfg.train.deterministic)

    dataset = CharDataset(cfg.data_path, cfg.model.block_size, device=device)
    cfg.model.vocab_size = dataset.vocab_size

    model = GPT(cfg.model).to(device)
    if cfg.train.grad_checkpoint:
        model.grad_checkpoint = True
    raw_model = model
    if cfg.train.compile and device.startswith("cuda"):
        model = torch.compile(model)

    optimizer = raw_model.configure_optimizers(cfg.train)

    # init_seed drives weight init + CUDA RNG; data_seed drives sampling order.
    data_gen = torch.Generator().manual_seed(cfg.data_seed)
    eval_gen = torch.Generator().manual_seed(cfg.data_seed + 10_000)

    use_amp = device.startswith("cuda")
    amp_dtype = getattr(torch, cfg.train.amp_dtype)
    autocast_ctx = (torch.autocast("cuda", dtype=amp_dtype) if use_amp
                    else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False))

    if verbose:
        print(f"[{cfg.name}] params={raw_model.num_params()/1e6:.1f}M "
              f"vocab={dataset.vocab_size} device={device} "
              f"init_seed={cfg.init_seed} data_seed={cfg.data_seed}")

    history = []
    best_val = float("inf")
    t0 = time.time()
    model.train()
    for step in range(cfg.train.max_steps):
        lr = cosine_lr(step, cfg.train.lr, cfg.train.warmup_steps,
                       cfg.train.max_steps, cfg.train.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = dataset.get_batch("train", cfg.train.batch_size, data_gen)
        with autocast_ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps - 1:
            val = evaluate(model, dataset, cfg.train.batch_size, cfg.train.eval_steps,
                           eval_gen, autocast_ctx)
            best_val = min(best_val, val)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val, "lr": lr})
            if verbose:
                print(f"  step {step:5d} | train {loss.item():.4f} | val {val:.4f} | lr {lr:.2e}")

    final_val = history[-1]["val_loss"]
    metrics = {
        "name": cfg.name,
        "init_seed": cfg.init_seed,
        "data_seed": cfg.data_seed,
        "params_M": raw_model.num_params() / 1e6,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "final_val_bpc": final_val / math.log(2),     # bits per char
        "wall_clock_s": time.time() - t0,
        "history": history,
    }

    if log:
        out = os.path.join(cfg.out_dir, cfg.name)
        os.makedirs(out, exist_ok=True)
        tag = f"i{cfg.init_seed}_d{cfg.data_seed}"
        with open(os.path.join(out, f"log_{tag}.json"), "w") as f:
            json.dump({"config": cfg.to_dict(), "metrics": metrics}, f, indent=2)
        # also write the canonical log.json (latest run) for plot.py convenience
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump({"config": cfg.to_dict(), "metrics": metrics}, f, indent=2)

    # free for the next in-process run (noise_floor)
    del model, raw_model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description="mini-deepseek-v4 baseline trainer")
    p.add_argument("--config", default="small", help="preset: small | gpu3090")
    p.add_argument("--data_path", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--data_seed", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    a = _parse_args()
    cfg = get_config(a.config)
    if a.data_path is not None: cfg.data_path = a.data_path
    if a.out_dir is not None: cfg.out_dir = a.out_dir
    if a.init_seed is not None: cfg.init_seed = a.init_seed
    if a.data_seed is not None: cfg.data_seed = a.data_seed
    if a.max_steps is not None: cfg.train.max_steps = a.max_steps
    if a.batch_size is not None: cfg.train.batch_size = a.batch_size
    if a.no_compile: cfg.train.compile = False
    if a.device is not None: cfg.device = a.device
    m = train_once(cfg)
    print(f"\nfinal val loss {m['final_val_loss']:.4f} "
          f"({m['final_val_bpc']:.4f} bpc) | best {m['best_val_loss']:.4f} "
          f"| {m['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()

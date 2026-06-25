"""Step 2 (groundwork): pre-tokenize each domain to a uint16 .bin shard (ROADMAP.md §2).

Replaces Step-1's per-run char loop (a Python loop over every char, reloaded each
seed) with a one-time tokenize -> memmap-able binary. vocab 16k < 65536 so token ids
fit uint16 (2 bytes/token). Each domain is trimmed to a per-domain TOKEN budget;
domains with fewer tokens (e.g. dialogue) keep all they have — training balances by
sampling, not by truncation.

Emits data/<domain>.bin + data/manifest.json (token counts, dtype, tokenizer hash).
Also reports chars-per-token per domain == the §2 "long-range inflation" measurement
(char-level inflates token distance ~4-5x vs BPE).

  python tokenize_corpus.py                       # 12M tokens/domain cap
  python tokenize_corpus.py --target_tokens 8000000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer

RAW_DIR = "data/raw"
OUT_DIR = "data"
DOMAINS = ["code", "prose", "math", "dialogue"]


def main():
    p = argparse.ArgumentParser(description="Step 2 corpus pre-tokenizer")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--target_tokens", type=int, default=12_000_000,
                   help="per-domain token cap; domains with fewer keep all")
    p.add_argument("--domains", nargs="*", default=DOMAINS)
    a = p.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    vocab = tok.get_vocab_size()
    assert vocab <= 65536, f"vocab {vocab} exceeds uint16 range"

    manifest = {"tokenizer": a.tokenizer, "vocab_size": vocab,
                "target_tokens": a.target_tokens, "dtype": "uint16", "domains": {}}
    print(f"tokenizer vocab={vocab}, per-domain cap={a.target_tokens:,}\n")

    for name in a.domains:
        raw = os.path.join(RAW_DIR, f"{name}.txt")
        if not os.path.exists(raw):
            print(f"  [warn] {raw} missing -> skip")
            continue
        with open(raw, "r", encoding="utf-8") as f:
            text = f.read()
        n_chars = len(text)
        ids = tok.encode(text).ids
        n_full = len(ids)
        if n_full > a.target_tokens:
            ids = ids[:a.target_tokens]
        arr = np.asarray(ids, dtype=np.uint16)
        out = os.path.join(OUT_DIR, f"{name}.bin")
        arr.tofile(out)

        ch_per_tok = n_chars / max(1, n_full)
        manifest["domains"][name] = {
            "bin": out, "n_tokens": int(arr.size), "n_tokens_available": n_full,
            "n_chars": n_chars, "chars_per_token": round(ch_per_tok, 3),
            "capped": n_full > a.target_tokens,
        }
        print(f"  [{name:9s}] {arr.size:>10,} toks "
              f"({'capped from '+format(n_full, ',') if n_full > a.target_tokens else 'all available'}) "
              f"| {ch_per_tok:.2f} ch/tok -> {out}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(d["n_tokens"] for d in manifest["domains"].values())
    print(f"\n  TOTAL {total:,} tokens across {len(manifest['domains'])} domains")
    print(f"  wrote {os.path.join(OUT_DIR, 'manifest.json')}")
    print("\n§2 long-range inflation: BPE ch/tok above ~ how many chars one token spans;\n"
          "char-level would be 1.0 ch/tok, so token *distances* shrink by that factor under BPE.")


if __name__ == "__main__":
    main()

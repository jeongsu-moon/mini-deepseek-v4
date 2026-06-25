"""Step 2 (groundwork): train ONE BPE tokenizer on the multi-domain corpus (ROADMAP.md §2).

§2 rule: the tokenizer is trained ONCE and frozen — reused unchanged in every
downstream step (retraining it per experiment makes runs incomparable). vocab is
fixed at 16k so the embedding/head table stays small at toy width (16k * n_embd,
tied) and leaves 24GB headroom.

ByteLevel BPE (GPT-2 style): no UNK, every byte is representable, so code / LaTeX /
dialogue punctuation all round-trip. Trained on a BALANCED sample across the 4
domains so no single domain dominates the merge table.

  python train_tokenizer.py                 # -> tokenizer.json (vocab 16000)
  python train_tokenizer.py --vocab 32000   # ROADMAP upper bound
"""
from __future__ import annotations

import argparse
import os

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

RAW_DIR = "data/raw"
EOT = "<|endoftext|>"


def _balanced_iter(domains, per_domain_bytes: int, chunk: int = 1 << 20):
    """Yield ~per_domain_bytes of text from each domain (balanced training set)."""
    for name in domains:
        path = os.path.join(RAW_DIR, f"{name}.txt")
        if not os.path.exists(path):
            print(f"  [warn] {path} missing -> skipping in tokenizer training")
            continue
        read = 0
        with open(path, "r", encoding="utf-8") as f:
            while read < per_domain_bytes:
                block = f.read(chunk)
                if not block:
                    break
                read += len(block)
                yield block
        print(f"  [{name}] fed {read/1e6:.1f} MB to trainer")


def main():
    p = argparse.ArgumentParser(description="Step 2 BPE tokenizer trainer")
    p.add_argument("--vocab", type=int, default=16000)
    p.add_argument("--out", default="tokenizer.json")
    p.add_argument("--domains", nargs="*", default=["code", "prose", "math", "dialogue"])
    p.add_argument("--per_domain_mb", type=float, default=24.0,
                   help="balanced training bytes per domain (enough for 16k merges, fast)")
    a = p.parse_args()

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"training BPE vocab={a.vocab} on domains={a.domains} "
          f"({a.per_domain_mb:.0f} MB each) ...")
    tok.train_from_iterator(_balanced_iter(a.domains, int(a.per_domain_mb * 1e6)),
                            trainer=trainer)
    tok.save(a.out)

    # quick round-trip sanity on one snippet per domain
    print(f"\nsaved {a.out}  (vocab={tok.get_vocab_size()}, EOT id={tok.token_to_id(EOT)})")
    for name in a.domains:
        path = os.path.join(RAW_DIR, f"{name}.txt")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(300)
        ids = tok.encode(sample).ids
        ok = tok.decode(ids) == sample
        print(f"  [{name}] 300 chars -> {len(ids)} toks ({len(sample)/max(1,len(ids)):.2f} ch/tok) "
              f"round-trip={'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()

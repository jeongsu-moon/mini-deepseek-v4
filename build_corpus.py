"""Step 2 (groundwork): assemble the multi-domain corpus (ROADMAP.md §2).

char-level (Step 1) destroys the very signals later steps measure — MoE expert
specialization, sparse long-range attention, MTP acceptance. §2 therefore mandates
a BPE switch on a corpus of *visually distinct* domains so per-domain expert routing
(Step 5) is measurable. enwik8/char data is explicitly banned here.

This script streams 4 domains and writes balanced raw text to data/raw/<domain>.txt.
A domain is capped by BYTES (not tokens) because the BPE tokenizer does not exist yet
(chicken-and-egg); tokenize_corpus.py trims to an exact per-domain token budget later.

  code     codeparrot/codeparrot-clean-valid   field 'content'   (Python files)
  prose    HuggingFaceFW/fineweb-edu           field 'text'      (edu web prose)
  math     open-web-math/open-web-math         field 'text'      (real LaTeX)
           -> fallback ccdv/arxiv-summarization field 'article'
  dialogue OpenAssistant/oasst1                field 'text'      (en, role-tagged)

Docs are separated by a blank line. Run is resumable per-domain (skips a domain
whose output already meets the byte cap).
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset

RAW_DIR = "data/raw"
DOC_SEP = "\n\n"


def _iter_code(cap):
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    for ex in ds:
        c = ex.get("content")
        if c:
            yield c


def _iter_prose(cap):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t


def _iter_math(cap):
    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        field = "text"
        print("  [math] using open-web-math (real LaTeX)")
    except Exception as e:                                            # noqa: BLE001
        print(f"  [math] open-web-math failed ({type(e).__name__}); falling back to arxiv-summarization")
        ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
        field = "article"
    for ex in ds:
        t = ex.get(field)
        if t:
            yield t


def _iter_dialogue(cap):
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    for ex in ds:
        if ex.get("lang") != "en" or ex.get("deleted"):
            continue
        role, text = ex.get("role"), ex.get("text")
        if text:
            yield f"{role}: {text}"


DOMAINS = {
    "code": _iter_code,
    "prose": _iter_prose,
    "math": _iter_math,
    "dialogue": _iter_dialogue,
}


def build_domain(name: str, cap_bytes: int) -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.txt")
    if os.path.exists(path) and os.path.getsize(path) >= cap_bytes:
        print(f"[{name}] already >= cap ({os.path.getsize(path)/1e6:.1f} MB) -> skip")
        return os.path.getsize(path)

    written = 0
    print(f"[{name}] target {cap_bytes/1e6:.0f} MB ...")
    with open(path, "w", encoding="utf-8") as f:
        for doc in DOMAINS[name](cap_bytes):
            doc = doc.strip()
            if not doc:
                continue
            f.write(doc)
            f.write(DOC_SEP)
            written += len(doc.encode("utf-8")) + len(DOC_SEP)
            if written >= cap_bytes:
                break
    print(f"[{name}] wrote {written/1e6:.1f} MB -> {path}")
    return written


def main():
    p = argparse.ArgumentParser(description="Step 2 multi-domain corpus builder")
    p.add_argument("--cap_mb", type=float, default=64.0,
                   help="per-domain raw byte cap (over-provision; trimmed to token budget later)")
    p.add_argument("--domains", nargs="*", default=list(DOMAINS),
                   help="subset of domains to build")
    a = p.parse_args()
    cap = int(a.cap_mb * 1e6)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    totals = {}
    for name in a.domains:
        if name not in DOMAINS:
            raise SystemExit(f"unknown domain {name!r}; choose from {list(DOMAINS)}")
        totals[name] = build_domain(name, cap)
    print("\n=== raw corpus summary ===")
    for name, b in totals.items():
        print(f"  {name:9s} {b/1e6:7.1f} MB")
    print(f"  {'TOTAL':9s} {sum(totals.values())/1e6:7.1f} MB")


if __name__ == "__main__":
    main()

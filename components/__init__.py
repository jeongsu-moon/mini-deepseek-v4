"""V4 components. Each module raises NotImplementedError until its roadmap step.

The point of these stubs is the *dispatch contract*: model.py / train.py construct
the right object from a config flag, and the stub tells you exactly what to build
and which pinned transformers source to cross-check against. Baseline never imports
these, so the baseline always runs (ROADMAP.md "측정 철학").
"""

"""stf.sentiment - fine-tune PhoBERT and produce 3-class sentiment probabilities.

Validity constraints (per docs/research-gap-analysis-and-development-flow.md):
  - 3 classes NEGATIVE / NEUTRAL / POSITIVE.
  - macro-F1 is the primary metric (class-imbalanced corpus).
  - The fine-tune split separates TIME from the forecast evaluation (no leakage).
  - PhoBERT-base: 256 tokens max.
  - Reproducibility: fixed seed, save config + checkpoint + manifest.
"""

from stf.sentiment.labels import ID2LABEL, LABEL2ID, LABELS

__all__ = ["LABELS", "LABEL2ID", "ID2LABEL"]

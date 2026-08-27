"""The 3-class sentiment label definitions, shared across the whole pipeline."""

from __future__ import annotations

# Fixed order: index = label id. Do not reorder after training.
LABELS: tuple[str, ...] = ("NEGATIVE", "NEUTRAL", "POSITIVE")

LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(LABELS)}

NUM_LABELS = len(LABELS)

"""Định nghĩa nhãn cảm xúc 3 lớp dùng thống nhất toàn pipeline."""

from __future__ import annotations

# Thứ tự cố định: index = id nhãn. Không đổi thứ tự này sau khi đã train.
LABELS: tuple[str, ...] = ("NEGATIVE", "NEUTRAL", "POSITIVE")

LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(LABELS)}

NUM_LABELS = len(LABELS)

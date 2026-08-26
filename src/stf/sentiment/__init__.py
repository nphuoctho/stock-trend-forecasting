"""stf.sentiment - fine-tune PhoBERT và sinh xác suất cảm xúc 3 lớp.

Ràng buộc validity (theo docs/research-gap-analysis-and-development-flow.md):
  - 3 lớp NEGATIVE / NEUTRAL / POSITIVE.
  - macro-F1 là metric chính (corpus lệch lớp).
  - Split fine-tune tách THỜI GIAN khỏi phần đánh giá dự báo (chống rò rỉ).
  - PhoBERT-base: tối đa 256 token.
  - Tái lập: seed cố định, lưu config + checkpoint + manifest.
"""

from stf.sentiment.labels import ID2LABEL, LABEL2ID, LABELS

__all__ = ["LABELS", "LABEL2ID", "ID2LABEL"]

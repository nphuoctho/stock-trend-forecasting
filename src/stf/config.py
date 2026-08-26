"""Cấu hình tập trung cho toàn pipeline.

Mọi hằng số dùng chung (danh sách mã, cửa sổ thời gian, đường dẫn dữ liệu) đặt ở
đây để một chỗ sửa là mọi module theo. Tránh mỗi script tự khai báo lệch nhau
như ở giai đoạn spike.
"""

from __future__ import annotations

from pathlib import Path

# --- Phạm vi đề tài -------------------------------------------------------

# 10 mã VN30 thanh khoản cao, mật độ tin phù hợp (chốt ở plan.md).
TICKERS: tuple[str, ...] = (
    "FPT", "GAS", "HPG", "MBB", "MWG",
    "TCB", "VCB", "VHM", "VIC", "VNM",
)

# Cửa sổ dữ liệu: 01/2020 -> 31/03/2026 (mở rộng từ 2020-2025 theo yêu cầu).
DATE_START = "2020-01-01"
DATE_END = "2026-03-31"

# Cutoff phiên HOSE dùng ở Phase 3 (tin sau giờ này dồn sang phiên kế tiếp).
SESSION_CUTOFF = "15:00"
TIMEZONE = "Asia/Ho_Chi_Minh"

# --- Đường dẫn ------------------------------------------------------------

# Gốc repo = thư mục chứa pyproject.toml (src/stf/config.py -> parents[2]).
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

PRICES_DIR = RAW / "prices"
NEWS_DIR = RAW / "news"
NEWS_HTML_DIR = NEWS_DIR / "html"

LISTINGS_PQ = NEWS_DIR / "listings.parquet"
ARTICLES_PQ = NEWS_DIR / "articles.parquet"

# Nơi lưu checkpoint/artifact mô hình sentiment.
MODELS = ROOT / "models"
SENTIMENT_DIR = MODELS / "sentiment"


def ensure_dirs() -> None:
    """Tạo mọi thư mục dữ liệu nếu chưa có (idempotent)."""
    for d in (RAW, INTERIM, PROCESSED, PRICES_DIR, NEWS_DIR, NEWS_HTML_DIR, MODELS):
        d.mkdir(parents=True, exist_ok=True)

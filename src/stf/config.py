"""Shared study settings and filesystem paths."""

from __future__ import annotations

from pathlib import Path

# --- Study scope ----------------------------------------------------------

# Symbols in the current study scope.
TICKERS: tuple[str, ...] = (
    "FPT",
    "GAS",
    "HPG",
    "MBB",
    "MWG",
    "TCB",
    "VCB",
    "VHM",
    "VIC",
    "VNM",
)

# Data window: Jan 2020 -> Mar 31 2026
DATE_START = "2020-01-01"
DATE_END = "2026-03-31"

# HOSE session cutoff
SESSION_CUTOFF = "15:00"
TIMEZONE = "Asia/Ho_Chi_Minh"

# --- Paths ----------------------------------------------------------------

# Repo root = the dir holding pyproject.toml (src/stf/config.py -> parents[2]).
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

# Where sentiment model checkpoints/artifacts are stored.
MODELS = ROOT / "models"
SENTIMENT_DIR = MODELS / "sentiment"


def ensure_dirs() -> None:
    """Create all data directories if missing (idempotent)."""
    for d in (RAW, INTERIM, PROCESSED, PRICES_DIR, NEWS_DIR, NEWS_HTML_DIR, MODELS):
        d.mkdir(parents=True, exist_ok=True)

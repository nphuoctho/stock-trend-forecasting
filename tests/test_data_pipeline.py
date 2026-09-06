"""Data pipeline tests - run offline, no network access.

uv run pytest tests/test_data_pipeline.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from stf.data import news, prices
from stf.sentiment import model
from stf.sentiment.dataset import make_split, normalize_labels
from stf.sentiment.metrics import classification_metrics, fleiss_kappa


def test_extract_body_from_vst_detail():
    """Extract content correctly from the articleBody/vst_detail block and strip HTML."""
    html = (
        "<html><body>"
        '<div itemprop="articleBody" id="vst_detail">'
        "<p class=pTitle>Tiêu đề bài</p>"
        "<p class=pBody>Hòa Phát bán 628,000 tấn thép trong tháng 8.</p>"
        "</div></body></html>"
    )
    body = news.extract_body(html)
    assert body is not None
    assert "Hòa Phát" in body
    assert "<p" not in body  # tags stripped
    assert "628,000 tấn thép" in body


def test_extract_body_missing_returns_none():
    """An article with no vst_detail block and no og:description returns None, doesn't crash."""
    assert news.extract_body("<html><body>không có body</body></html>") is None


def test_extract_body_fallback_og_description():
    """A longform article without vst_detail falls back to og:description."""
    html = (
        "<html><head>"
        '<meta property="og:description" content="Chuyên đề cuộc đua lợi nhuận VN30 quý 3.">'
        "</head><body>longform layout không có vst_detail</body></html>"
    )
    body = news.extract_body(html)
    assert body is not None
    assert "cuộc đua lợi nhuận VN30" in body


def test_parse_article_fields():
    """parse_article returns all 3 keys: timestamp, title, body."""
    html = (
        '<meta property="og:title" content="Sản lượng thép Hòa Phát">'
        '<span itemprop="datePublished">08/09/2022 16:35</span>'
        '<div itemprop="articleBody" id="vst_detail"><p>Nội dung.</p></div>'
    )
    parsed = news.parse_article(html)
    assert parsed["published_at_str"] == "08/09/2022 16:35"
    assert parsed["title"] == "Sản lượng thép Hòa Phát"
    assert "Nội dung" in parsed["body"]


def test_to_iso_parsing():
    """Invalid timestamps are rejected; valid ones keep local timezone."""
    assert news._to_iso("08/09/2022 16:35") == "2022-09-08T16:35:00+07:00"
    assert news._to_iso(None) is None
    assert news._to_iso("không hợp lệ") is None


def test_old_stored_timestamp_is_upgraded():
    assert (
        news._normalize_stored_timestamp("2022-09-08T16:35:00")
        == "2022-09-08T16:35:00+07:00"
    )


def test_has_body_handles_nan():
    """Treat missing and whitespace-only bodies as empty."""
    import numpy as np

    assert news._has_body("nội dung thật") is True
    assert news._has_body(None) is False
    assert news._has_body(float("nan")) is False
    assert news._has_body(np.nan) is False
    assert news._has_body("") is False
    assert news._has_body("   ") is False


def test_extract_body_handles_nested_markup_and_attribute_order():
    html = (
        '<div id="vst_detail" itemprop="articleBody">'
        "<div><p>Nội dung <strong>lồng nhau</strong>.</p></div>"
        "</div>"
    )
    assert news.extract_body(html) == "Nội dung lồng nhau ."



def test_normalize_labels_accepts_numeric_strings():
    normalized = normalize_labels(pd.DataFrame({"label": ["0", "NEUTRAL", 2]}))
    assert normalized["label_id"].tolist() == [0, 1, 2]


def test_normalize_labels_rejects_unknown_ids():

    with pytest.raises(ValueError, match="Invalid labels"):
        normalize_labels(pd.DataFrame({"label": [3]}))


def test_time_split_keeps_chronological_order():

    frame = pd.DataFrame(
        {
            "text": [f"tin {i}" for i in range(30)],
            "label_id": [i % 3 for i in range(30)],
            "date": pd.date_range("2024-01-01", periods=30),
        }
    )
    split = make_split(frame, time_aware=True)
    assert split.train["date"].max() < split.val["date"].min()
    assert split.val["date"].max() < split.test["date"].min()


def test_price_normalize_sorts_rows_and_adds_ticker():
    frame = pd.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-01"],
            "open": [2, 1],
            "high": [2, 1],
            "low": [2, 1],
            "close": [2, 1],
            "volume": [20, 10],
        }
    )
    normalized = prices._normalize(frame, "FPT")
    assert normalized["time"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
    ]
    assert normalized["ticker"].tolist() == ["FPT", "FPT"]


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        classification_metrics([], [])


def test_fleiss_kappa_requires_consistent_rater_counts():
    assert fleiss_kappa(pd.DataFrame([[2, 0, 0], [0, 2, 0]]).to_numpy()) == 1.0
    with pytest.raises(ValueError, match="same number"):
        fleiss_kappa(pd.DataFrame([[2, 0, 0], [1, 0, 0]]).to_numpy())


def test_predict_proba_handles_empty_input_without_loading_model():
    result = model.predict_proba([])
    assert result.shape == (0, 3)

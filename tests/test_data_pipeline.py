"""Test pipeline dữ liệu — chạy offline, không truy cập mạng.

    uv run pytest tests/test_data_pipeline.py -q
"""

from __future__ import annotations

from stf.data import news


def test_extract_body_from_vst_detail():
    """Trích đúng nội dung từ khối articleBody/vst_detail và strip HTML."""
    html = (
        '<html><body>'
        '<div itemprop="articleBody" id="vst_detail">'
        '<p class=pTitle>Tiêu đề bài</p>'
        '<p class=pBody>Hòa Phát bán 628,000 tấn thép trong tháng 8.</p>'
        '</div></body></html>'
    )
    body = news.extract_body(html)
    assert body is not None
    assert "Hòa Phát" in body
    assert "<p" not in body  # đã strip tag
    assert "628,000 tấn thép" in body


def test_extract_body_missing_returns_none():
    """Bài không có khối vst_detail (vd Longform) trả None, không crash."""
    assert news.extract_body("<html><body>không có body</body></html>") is None


def test_parse_article_fields():
    """parse_article trả đủ 3 khóa: timestamp, title, body."""
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
    """Timestamp dd/mm/yyyy hh:mm -> ISO; sai định dạng -> None."""
    assert news._to_iso("08/09/2022 16:35") == "2022-09-08T16:35:00"
    assert news._to_iso(None) is None
    assert news._to_iso("không hợp lệ") is None


def test_has_body_handles_nan():
    """_has_body phân biệt đúng body thật với None/NaN/rỗng (chống bug NaN truthy)."""
    import numpy as np

    assert news._has_body("nội dung thật") is True
    assert news._has_body(None) is False
    assert news._has_body(float("nan")) is False
    assert news._has_body(np.nan) is False
    assert news._has_body("") is False
    assert news._has_body("   ") is False

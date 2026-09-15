"""Data pipeline tests - run offline, no network access.

uv run pytest tests/test_data_pipeline.py -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stf.data import news, prices
from stf.sentiment import model
from stf.sentiment.dataset import (
    build_input_text,
    file_fingerprint,
    frame_fingerprint,
    make_split,
    normalize_labels,
    prepare_model_input,
    reject_preliminary_labels,
    resolve_time_column,
)
from stf.sentiment.experiments import make_stratified_folds
from stf.sentiment.metrics import classification_metrics



def test_build_input_text_selects_title_context_variants():
    frame = pd.DataFrame(
        {
            "title": ["Tiêu đề tốt", "Chỉ có tiêu đề"],
            "body": ["Nội dung dài", ""],
            "label": ["POSITIVE", "NEUTRAL"],
        }
    )
    assert build_input_text(frame, "title")["text"].tolist() == [
        "Tiêu đề tốt",
        "Chỉ có tiêu đề",
    ]
    assert build_input_text(frame, "context")["text"].tolist() == [
        "Nội dung dài",
        "Chỉ có tiêu đề",
    ]
    assert build_input_text(frame, "title_context")["text"].tolist() == [
        "Tiêu đề tốt\n\nNội dung dài",
        "Chỉ có tiêu đề",
    ]


def test_text_dataset_truncates_before_adding_special_tokens():
    class FakeTokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(8))}

        def prepare_for_model(self, token_ids, **_kwargs):
            ids = [101, *token_ids, 102]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    ds = model._TextDataset(
        ["dữ liệu dài"],
        [2],
        FakeTokenizer(),
        max_len=6,
        truncation_strategy="head_tail",
    )
    assert ds[0]["input_ids"] == [101, 0, 1, 6, 7, 102]


def test_truncation_strategies_preserve_requested_regions():
    tokens = list(range(10))
    assert model.truncate_token_ids(tokens, 4, "head") == [0, 1, 2, 3]
    assert model.truncate_token_ids(tokens, 4, "tail") == [6, 7, 8, 9]
    assert model.truncate_token_ids(tokens, 4, "head_tail") == [0, 1, 8, 9]


def test_resolve_truncation_strategy_reads_checkpoint_manifest(tmp_path):
    out_dir = tmp_path / "run"
    model_dir = out_dir / "best"
    model_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"config": {"truncation_strategy": "tail"}}), encoding="utf-8"
    )
    assert model.resolve_truncation_strategy(model_dir) == "tail"


def test_resolve_truncation_strategy_override_wins_over_manifest(tmp_path):
    out_dir = tmp_path / "run"
    model_dir = out_dir / "best"
    model_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"config": {"truncation_strategy": "tail"}}), encoding="utf-8"
    )
    assert model.resolve_truncation_strategy(model_dir, "head_tail") == "head_tail"


def test_resolve_truncation_strategy_defaults_to_head_without_manifest(tmp_path):
    model_dir = tmp_path / "best"
    model_dir.mkdir()
    assert model.resolve_truncation_strategy(model_dir) == "head"


def test_resolve_truncation_strategy_rejects_invalid_override(tmp_path):
    model_dir = tmp_path / "best"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="Unknown truncation strategy"):
        model.resolve_truncation_strategy(model_dir, "middle")


def test_bounded_warmup_steps_scales_with_ratio():
    # 100 examples, batch 10 -> 10 steps/epoch * 5 epochs = 50 total steps.
    assert model.bounded_warmup_steps(100, 10, 5, 0.1) == 5


def test_bounded_warmup_steps_caps_at_total_when_ratio_is_one():
    # 5 examples, batch 2 -> 3 steps/epoch * 1 epoch = 3 total steps.
    assert model.bounded_warmup_steps(5, 2, 1, 1.0) == 3


def test_bounded_warmup_steps_rejects_out_of_range_ratio():
    with pytest.raises(ValueError, match="warmup_ratio"):
        model.bounded_warmup_steps(100, 10, 1, 1.5)
def test_compute_class_weights_uses_training_class_frequencies():
    weights = model.compute_class_weights([0, 0, 1, 2, 2, 2])
    assert weights.tolist() == pytest.approx([1.0, 2.0, 2.0 / 3.0])


def test_compute_class_weights_rejects_missing_class():
    with pytest.raises(ValueError, match="Every class"):
        model.compute_class_weights([0, 0, 1, 1])



def test_stratified_folds_are_disjoint_and_cover_every_row():
    frame = pd.DataFrame({"label_id": [0, 1, 2] * 5})
    folds = make_stratified_folds(frame, n_splits=5, seed=7)
    holdouts = [set(holdout) for _, holdout in folds]
    assert set.union(*holdouts) == set(range(len(frame)))
    assert sum(len(part) for part in holdouts) == len(frame)
    for train, holdout in folds:
        assert set(train).isdisjoint(holdout)

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



def test_listing_rows_pair_dates_with_nearest_article_url():
    html = """
    <div class="item">08/09/2022 <a href="//vietstock.vn/2022/09/first-1.htm">Một</a></div>
    <div class="item">09/09/2022 <a href="//vietstock.vn/2022/09/second-2.htm">Hai</a></div>
    """
    assert news._listing_rows(html) == [
        ("//vietstock.vn/2022/09/first-1.htm", "08/09/2022"),
        ("//vietstock.vn/2022/09/second-2.htm", "09/09/2022"),
    ]

def test_to_iso_parsing():
    """Invalid timestamps are rejected; valid ones keep local timezone."""
    assert news._to_iso("08/09/2022 16:35") == "2022-09-08T16:35:00+07:00"
    assert news._to_iso(None) is None
    assert news._to_iso("không hợp lệ") is None


def test_join_listings_articles_preserves_many_to_many_ticker_links():
    listings = pd.DataFrame(
        {
            "ticker": ["FPT", "VNM", "FPT"],
            "url": ["u1", "u1", "u1"],
        }
    )
    articles = pd.DataFrame(
        {
            "url": ["u1", "u1"],
            "published_at": ["2024-01-01T10:00:00+07:00"] * 2,
            "title": ["Tin"] * 2,
            "body": ["Nội dung"] * 2,
        }
    )
    joined = news.join_listings_articles(listings, articles)
    assert joined[["ticker", "url"]].drop_duplicates().to_dict("records") == [
        {"ticker": "FPT", "url": "u1"},
        {"ticker": "VNM", "url": "u1"},
    ]


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


def test_clean_html_strips_style_and_script_blocks():
    html = (
        "<style>.pTitle{color:red}</style>"
        "<script>trackClick();</script>"
        "<p>N\u1ed9i dung s\u1ea1ch.</p>"
    )
    assert news.clean_html(html) == "N\u1ed9i dung s\u1ea1ch."


def test_clean_html_returns_none_for_markup_only_input():
    assert news.clean_html("<style>.a{color:red}</style>") is None
    assert news.clean_html(None) is None


def test_extract_body_drops_inline_style_block_from_vst_detail():
    html = (
        "<html><body>"
        '<div itemprop="articleBody" id="vst_detail">'
        "<style>.pBody{font-size:14px}</style>"
        "<p>Tin t\u1ee9c kh\u00f4ng d\u00ednh CSS.</p>"
        "</div></body></html>"
    )
    assert news.extract_body(html) == "Tin t\u1ee9c kh\u00f4ng d\u00ednh CSS."



def test_normalize_labels_accepts_numeric_strings():
    normalized = normalize_labels(pd.DataFrame({"label": ["0", "NEUTRAL", 2]}))
    assert normalized["label_id"].tolist() == [0, 1, 2]


def test_normalize_labels_rejects_stale_label_id():
    frame = pd.DataFrame({"label": ["POSITIVE"], "label_id": [0]})
    with pytest.raises(ValueError, match="disagree"):
        normalize_labels(frame)


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


def test_time_aware_split_fails_loudly_without_date_or_published_at():
    frame = pd.DataFrame(
        {
            "text": [f"tin {i}" for i in range(20)],
            "label_id": [i % 3 for i in range(20)],
        }
    )
    with pytest.raises(ValueError, match="Time-aware split requires"):
        make_split(frame, time_aware=True)


def test_time_aware_split_fails_loudly_on_unparseable_dates():
    frame = pd.DataFrame(
        {
            "text": [f"tin {i}" for i in range(5)],
            "label_id": [i % 3 for i in range(5)],
            "date": ["2024-01-01", "not-a-date", "2024-01-03", "2024-01-04", "2024-01-05"],
        }
    )
    with pytest.raises(ValueError, match="valid timestamp"):
        make_split(frame, time_aware=True)


def test_resolve_time_column_normalizes_published_at_to_study_timezone():
    frame = pd.DataFrame(
        {
            "published_at": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        }
    )
    dates = resolve_time_column(frame)
    assert str(dates.dt.tz) == "Asia/Ho_Chi_Minh"


def test_resolve_time_column_interprets_naive_values_in_study_timezone():
    frame = pd.DataFrame({"published_at": ["2024-01-01T23:30:00"]})
    dates = resolve_time_column(frame)
    assert dates.iloc[0] == pd.Timestamp(
        "2024-01-01 23:30:00", tz="Asia/Ho_Chi_Minh"
    )


def test_time_aware_split_derives_date_column_from_published_at():
    frame = pd.DataFrame(
        {
            "text": [f"tin {i}" for i in range(20)],
            "label_id": [i % 3 for i in range(20)],
            "published_at": pd.date_range("2024-01-01", periods=20, tz="UTC"),
        }
    )
    split = make_split(frame, time_aware=True)
    assert "date" in split.train.columns
    assert str(split.train["date"].dt.tz) == "Asia/Ho_Chi_Minh"
    assert split.train["date"].max() < split.test["date"].min()


def test_metrics_keep_absent_classes_in_macro_scores():
    result = classification_metrics([0, 2, 0, 2], [0, 2, 0, 2])
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx(1.0)


def test_time_split_deduplicates_and_keeps_whole_dates_together():
    frame = pd.DataFrame(
        {
            "text": ["a", "a", "b", "c", "d", "e", "f", "g", "h"],
            "label_id": [0, 0, 1, 2, 0, 1, 2, 0, 1],
            "published_at": [
                "2024-01-01T10:00:00+07:00",
                "2024-01-01T12:00:00+07:00",
                "2024-01-02T10:00:00+07:00",
                "2024-01-03T10:00:00+07:00",
                "2024-01-04T10:00:00+07:00",
                "2024-01-05T10:00:00+07:00",
                "2024-01-06T10:00:00+07:00",
                "2024-01-07T10:00:00+07:00",
                "2024-01-08T10:00:00+07:00",
            ],
        }
    )
    split = make_split(frame, val_frac=0.2, test_frac=0.2)
    assert sum(len(part) for part in (split.train, split.val, split.test)) == 8
    assert set(split.train["date"]).isdisjoint(split.val["date"])
    assert set(split.val["date"]).isdisjoint(split.test["date"])


def test_resolve_time_column_accepts_mixed_offsets():
    frame = pd.DataFrame(
        {"published_at": ["2024-01-01T10:00:00+07:00", "2024-01-01T03:00:00Z"]}
    )
    dates = resolve_time_column(frame)
    assert str(dates.dt.tz) == "Asia/Ho_Chi_Minh"


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




def test_predict_proba_handles_empty_input_without_loading_model():
    result = model.predict_proba([])
    assert result.shape == (0, 3)


def test_prepare_model_input_dedupes_after_variant_not_before():
    """Dedup must run on the selected variant's text, not a stale precomputed one.

    Real in-domain files ship a precomputed ``text`` column built for a different
    variant (title_context). Two rows here share a title but differ in body and
    precomputed text; deduping before variant selection would miss the duplicate
    that the "title" variant actually produces.
    """
    frame = pd.DataFrame(
        {
            "title": ["Cùng tiêu đề", "Cùng tiêu đề"],
            "body": ["Nội dung A", "Nội dung B"],
            "text": ["Cùng tiêu đề\n\nNội dung A", "Cùng tiêu đề\n\nNội dung B"],
            "url": ["https://example.test/a", "https://example.test/b"],
            "label_id": [0, 1],
        }
    )
    out = prepare_model_input(frame, "title")
    assert out["text"].tolist() == ["Cùng tiêu đề"]


def test_reject_preliminary_labels_blocks_unreviewed_rows_by_default():
    frame = pd.DataFrame(
        {
            "text": ["a", "b"],
            "label_id": [0, 1],
            "annotation_status": ["PRELIMINARY_REVIEW_REQUIRED", "CONFIRMED"],
        }
    )
    with pytest.raises(ValueError, match="preliminary"):
        reject_preliminary_labels(frame)
    allowed = reject_preliminary_labels(frame, allow_preliminary=True)
    assert len(allowed) == 2


def test_reject_preliminary_labels_checks_annotation_source():
    frame = pd.DataFrame(
        {"text": ["a"], "label_id": [0], "annotation_source": ["assistant_prelabel"]}
    )
    with pytest.raises(ValueError, match="preliminary"):
        reject_preliminary_labels(frame)


def test_reject_preliminary_labels_accepts_public_files_without_provenance():
    frame = pd.DataFrame({"text": ["a", "b"], "label_id": [0, 1]})
    out = reject_preliminary_labels(frame)
    assert len(out) == 2


def test_resolve_inference_config_reads_manifest_max_len(tmp_path):
    out_dir = tmp_path / "run"
    model_dir = out_dir / "best"
    model_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"config": {"truncation_strategy": "tail", "max_len": 128}}),
        encoding="utf-8",
    )
    assert model.resolve_inference_config(model_dir) == ("tail", 128)


def test_resolve_inference_config_overrides_win_over_manifest(tmp_path):
    out_dir = tmp_path / "run"
    model_dir = out_dir / "best"
    model_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text(
        json.dumps({"config": {"truncation_strategy": "tail", "max_len": 128}}),
        encoding="utf-8",
    )
    assert model.resolve_inference_config(
        model_dir, truncation_strategy="head", max_len=64
    ) == ("head", 64)


def test_resolve_inference_config_defaults_without_manifest(tmp_path):
    model_dir = tmp_path / "best"
    model_dir.mkdir()
    assert model.resolve_inference_config(model_dir) == ("head", model.MAX_LEN)


def test_frame_fingerprint_is_order_invariant_and_content_sensitive():
    a = pd.DataFrame({"text": ["x", "y"], "label_id": [0, 1]})
    b = pd.DataFrame({"text": ["y", "x"], "label_id": [1, 0]})
    c = pd.DataFrame({"text": ["x", "y"], "label_id": [0, 2]})
    assert frame_fingerprint(a) == frame_fingerprint(b)
    assert frame_fingerprint(a) != frame_fingerprint(c)


def test_frame_fingerprint_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        frame_fingerprint(pd.DataFrame({"text": [], "label_id": []}))


def test_file_fingerprint_is_deterministic_and_content_sensitive(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = file_fingerprint(path)
    assert first == file_fingerprint(path)
    path.write_text("a,b\n1,3\n", encoding="utf-8")
    assert file_fingerprint(path) != first


def test_is_allowed_url_accepts_exact_vietstock_hosts_only():
    assert news._is_allowed_url("https://vietstock.vn/2021/01/a-1.htm")
    assert news._is_allowed_url("https://finance.vietstock.vn/View/PagingNewsContent")
    assert not news._is_allowed_url("http://vietstock.vn/2021/01/a-1.htm")
    assert not news._is_allowed_url("https://evil.vietstock.vn.attacker.com/x")
    assert not news._is_allowed_url("https://notvietstock.vn/x")


def test_get_rejects_disallowed_url_without_a_network_call(monkeypatch):
    calls = []
    monkeypatch.setattr(news.requests, "get", lambda *a, **k: calls.append((a, k)))
    assert news.get("https://evil.example.com/x") is None
    assert news.get("http://vietstock.vn/2021/01/a-1.htm") is None
    assert calls == []


def test_get_disables_automatic_redirects(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def __init__(self):
            self.encoding = None

    def fake_get(url, params=None, headers=None, timeout=None, allow_redirects=None):
        captured["allow_redirects"] = allow_redirects
        return FakeResponse()

    monkeypatch.setattr(news.requests, "get", fake_get)
    result = news.get("https://vietstock.vn/2021/01/a-1.htm")
    assert result is not None
    assert captured["allow_redirects"] is False


def test_price_normalize_rejects_non_positive_close():
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [0],
            "volume": [10],
        }
    )
    with pytest.raises(ValueError, match="non-positive close"):
        prices._normalize(frame, "FPT")


def test_price_normalize_rejects_non_positive_volume():
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [0],
        }
    )
    with pytest.raises(ValueError, match="non-positive volume"):
        prices._normalize(frame, "FPT")


def test_price_normalize_rejects_non_finite_values():
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [float("inf")],
            "volume": [10],
        }
    )
    with pytest.raises(ValueError, match="non-finite"):
        prices._normalize(frame, "FPT")

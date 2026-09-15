"""Data pipeline tests - run offline, no network access.

uv run pytest tests/test_data_pipeline.py -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stf.data import news, prices
from stf.sentiment.annotation import compare_raters, load_annotation_file
from stf.sentiment import model
from stf.sentiment.dataset import (
    build_input_text,
    make_split,
    normalize_labels,
    resolve_time_column,
)
from stf.sentiment.experiments import make_stratified_folds
from stf.sentiment.metrics import classification_metrics, fleiss_kappa



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


def test_fleiss_kappa_requires_consistent_rater_counts():
    assert fleiss_kappa(pd.DataFrame([[2, 0, 0], [0, 2, 0]]).to_numpy()) == 1.0
    with pytest.raises(ValueError, match="same number"):
        fleiss_kappa(pd.DataFrame([[2, 0, 0], [1, 0, 0]]).to_numpy())


def test_annotation_comparison_requires_complete_matching_raters(tmp_path):
    rater_a = pd.DataFrame({"sample_id": [1, 2, 3], "label": ["NEGATIVE", "NEUTRAL", "POSITIVE"]})
    rater_b = pd.DataFrame({"sample_id": [1, 2, 3], "label": ["NEGATIVE", "POSITIVE", "POSITIVE"]})
    path_a = tmp_path / "rater_a.csv"
    path_b = tmp_path / "rater_b.csv"
    rater_a.to_csv(path_a, index=False)
    rater_b.to_csv(path_b, index=False)

    result = compare_raters([path_a, path_b])

    assert result["n_items"] == 3
    assert result["n_disagreements"] == 1
    assert result["disagreements"]["sample_id"].tolist() == [2]


def test_annotation_loader_rejects_unlabeled_rows(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"sample_id": [1], "label": [""]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no label"):
        load_annotation_file(path)

def test_annotation_loader_rejects_empty_files(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["sample_id", "label"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_annotation_file(path)


def test_predict_proba_handles_empty_input_without_loading_model():
    result = model.predict_proba([])
    assert result.shape == (0, 3)

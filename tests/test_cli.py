"""CLI contract tests that stay local and deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stf.cli import main


def test_forecast_smoke_command_reports_partitioned_sequences(capsys):
    assert main(["forecast-smoke", "--n", "20", "--epochs", "1"]) == 0
    output = capsys.readouterr().out
    assert "sequences=train:" in output
    assert "val:" in output
    assert "test:" in output


def test_forecast_smoke_rejects_too_short_input(capsys):
    assert main(["forecast-smoke", "--n", "11"]) == 2
    assert "--n must be at least 12" in capsys.readouterr().err


def test_forecast_smoke_trains_both_price_and_fused_models(capsys):
    assert main(["forecast-smoke", "--n", "20", "--epochs", "1"]) == 0
    output = capsys.readouterr().out
    assert "price_lstm_final_loss=" in output
    assert "price_sentiment_lstm_final_loss=" in output


def test_score_news_scores_articles_without_a_real_model(monkeypatch, tmp_path):
    from stf.data import news as news_module
    from stf.sentiment import model as model_module

    articles = pd.DataFrame(
        {
            "ticker": ["FPT", "VNM"],
            "url": ["https://vietstock.vn/a.htm", "https://vietstock.vn/b.htm"],
            "published_at": [
                "2021-01-01T10:00:00+07:00",
                "2021-01-02T10:00:00+07:00",
            ],
            "title": ["Tin A", "Tin B"],
            "body": ["Nội dung A", "Nội dung B"],
        }
    )
    monkeypatch.setattr(news_module, "load_ticker_articles", lambda: articles)

    captured = {}

    def fake_predict_proba(texts, model_dir, *, batch_size=32, truncation_strategy=None):
        captured["texts"] = list(texts)
        captured["model_dir"] = model_dir
        return np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])

    monkeypatch.setattr(model_module, "predict_proba", fake_predict_proba)

    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "scored.parquet"
    rc = main(
        [
            "score-news",
            "--model-dir",
            str(model_dir),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert captured["texts"] == ["Tin A", "Tin B"]
    result = pd.read_parquet(output)
    assert result["ticker"].tolist() == ["FPT", "VNM"]
    assert result["prob_positive"].tolist() == pytest.approx([0.1, 0.7])
    manifest = pd.read_json(output.with_suffix(".manifest.json"), typ="series")
    assert manifest["output"]["rows"] == 2
    assert manifest["probabilities"]["totals"]["prob_neutral"] == pytest.approx(0.4)
    assert manifest["checkpoint"]["directory_sha256"]
    assert manifest["input"]["variant"] == "title"
    assert manifest["input"]["fingerprint"]
    assert manifest["input"]["limit"] is None

    articles = articles.iloc[::-1].reset_index(drop=True)
    reordered_output = tmp_path / "reordered.parquet"
    assert (
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(reordered_output),
            ]
        )
        == 0
    )
    reordered_manifest = pd.read_json(
        reordered_output.with_suffix(".manifest.json"), typ="series"
    )
    assert reordered_manifest["input"]["fingerprint"] == manifest["input"]["fingerprint"]

    articles.loc[0, "title"] = "Tin A đã sửa"
    changed_output = tmp_path / "changed.parquet"
    assert (
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(changed_output),
            ]
        )
        == 0
    )
    changed_manifest = pd.read_json(
        changed_output.with_suffix(".manifest.json"), typ="series"
    )
    assert changed_manifest["input"]["fingerprint"] != manifest["input"]["fingerprint"]


    monkeypatch.setattr(
        model_module,
        "predict_proba",
        lambda *_args, **_kwargs: np.array([[1.0, 0.1, -0.1], [0.0, 0.0, 1.0]]),
    )
    invalid_output = tmp_path / "invalid.parquet"
    with pytest.raises(RuntimeError, match="invalid probability vectors"):
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(invalid_output),
            ]
        )
    assert not invalid_output.exists()

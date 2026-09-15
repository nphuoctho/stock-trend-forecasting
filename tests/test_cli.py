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

    output = tmp_path / "scored.parquet"
    rc = main(
        [
            "score-news",
            "--model-dir",
            str(tmp_path / "fake-model"),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert captured["texts"] == ["Tin A", "Tin B"]
    result = pd.read_parquet(output)
    assert result["ticker"].tolist() == ["FPT", "VNM"]
    assert result["prob_positive"].tolist() == pytest.approx([0.1, 0.7])

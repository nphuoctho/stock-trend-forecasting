"""Forecasting core tests - offline, deterministic, no model downloads.

Focus: point-in-time correctness. The feature causality tests fail if any future price
value leaks into a feature column.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stf.forecasting import calendar as cal
from stf.forecasting.experiment import (
    LADDER,
    ForecastConfig,
    _date_block_bootstrap,
    compare_information_gain,
    run_experiment,
)
from stf.forecasting.features import FeatureScaler, price_feature_columns, price_features
from stf.forecasting.labels import (
    TREND_LABELS,
    add_target,
    apply_labels,
    fit_thresholds,
    label_panel,
)
from stf.forecasting.models import (
    ClassicalBaseline,
    MajorityBaseline,
    PriceLSTM,
    PriceSentimentLSTM,
    RandomBaseline,
    TemporalFusionClassifier,
    evaluate_predictions,
    fit_lstm,
    make_sequences,
    make_two_branch_sequences,
    predict_lstm,
    set_seed,
)
from stf.forecasting.panel import assemble, build_panel
from stf.forecasting.sentiment_agg import (
    ROLLING_SENTIMENT_COLUMNS,
    add_rolling_sentiment,
    daily_sentiment,
)
from stf.forecasting.split import chronological_split, walk_forward_windows

# --- fixtures -------------------------------------------------------------------------


def _prices(ticker: str = "FPT", n: int = 40, start: str = "2021-01-04") -> pd.DataFrame:
    """Business-day OHLCV with a deterministic drifting close."""
    dates = pd.bdate_range(start=start, periods=n)
    close = 100.0 + np.arange(n, dtype=float) + np.sin(np.arange(n)) * 2.0
    return pd.DataFrame(
        {
            "ticker": ticker,
            "time": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000 + np.arange(n) * 1_000,
        }
    )


# --- calendar / cutoff mapping --------------------------------------------------------


def test_before_cutoff_maps_to_same_session():
    prices = _prices(n=10)  # starts Mon 2021-01-04
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "published_at": ["2021-01-05T14:00:00+07:00"],  # Tuesday, before 15:00
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    )
    aligned = cal.align_news_to_sessions(news, prices)
    assert aligned.loc[0, "mapping_status"] == "same_session"
    assert aligned.loc[0, "observation_date"] == pd.Timestamp("2021-01-05")


def test_after_cutoff_rolls_to_next_session():
    prices = _prices(n=10)
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "published_at": ["2021-01-05T16:00:00+07:00"],  # Tuesday, after 15:00
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    )
    aligned = cal.align_news_to_sessions(news, prices)
    assert aligned.loc[0, "mapping_status"] == "next_session"
    assert aligned.loc[0, "observation_date"] == pd.Timestamp("2021-01-06")  # Wednesday


def test_weekend_news_maps_to_next_trading_session():
    prices = _prices(n=10)
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "published_at": ["2021-01-09T10:00:00+07:00"],  # Saturday
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    )
    aligned = cal.align_news_to_sessions(news, prices)
    assert aligned.loc[0, "mapping_status"] == "next_session"
    assert aligned.loc[0, "observation_date"] == pd.Timestamp("2021-01-11")  # Monday



def test_stale_news_before_calendar_is_dropped_explicitly():
    prices = _prices(n=5)
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "published_at": ["2019-01-01T09:00:00+07:00"],
        }
    )
    aligned = cal.align_news_to_sessions(news, prices)
    assert aligned.loc[0, "mapping_status"] == "stale"
    assert cal.alignment_report(aligned)["dropped"] == 1

def test_out_of_calendar_and_invalid_are_reported_not_kept():
    prices = _prices(n=5)  # last session 2021-01-08
    news = pd.DataFrame(
        {
            "ticker": ["FPT", "FPT", "ZZZ"],
            "published_at": ["2021-06-01T09:00:00+07:00", "not-a-date", "2021-01-05T09:00:00+07:00"],
            "prob_negative": [0.3, 0.3, 0.3],
            "prob_neutral": [0.4, 0.4, 0.4],
            "prob_positive": [0.3, 0.3, 0.3],
        }
    )
    aligned = cal.align_news_to_sessions(news, prices)
    statuses = aligned["mapping_status"].tolist()
    assert statuses == ["unmapped", "invalid", "no_calendar"]
    report = cal.alignment_report(aligned)
    assert report["mapped"] == 0
    assert report["dropped"] == 3
    # None of the dropped rows carry an observation date.
    assert aligned["observation_date"].isna().all()


def test_assemble_retains_alignment_report():
    prices = _prices(n=5)
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "published_at": ["2021-01-05T14:00:00+07:00"],
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    )
    panel = assemble(prices, news)
    assert panel.attrs["alignment_report"]["mapped"] == 1


# --- price feature causality (leakage guard) ------------------------------------------


def test_rolling_features_use_only_current_and_past_rows():
    prices = _prices(n=20)
    base = price_features(prices, ma_window=5, vol_window=5)

    # Mutate ONLY the last close far into the future direction.
    tampered = prices.copy()
    tampered.loc[tampered.index[-1], "close"] = 10_000.0
    after = price_features(tampered, ma_window=5, vol_window=5)

    feat_cols = price_feature_columns(5, 5)
    # Every row except the final one must be untouched: a future close cannot change them.
    head_base = base.iloc[:-1][feat_cols].to_numpy()
    head_after = after.iloc[:-1][feat_cols].to_numpy()
    np.testing.assert_allclose(np.nan_to_num(head_base), np.nan_to_num(head_after))
    # The final row's features DO change (it legitimately depends on that close).
    assert not np.allclose(
        np.nan_to_num(base.iloc[-1][feat_cols].to_numpy(dtype=float)),
        np.nan_to_num(after.iloc[-1][feat_cols].to_numpy(dtype=float)),
    )


def test_early_rows_lack_history_and_are_nan():
    prices = _prices(n=10)
    feats = price_features(prices, ma_window=5, vol_window=5)
    # First row has no prior close -> return/log/vol_change NaN; rolling needs 5 rows.
    assert np.isnan(feats.iloc[0]["ret_1d"])
    assert np.isnan(feats.iloc[0]["log_ret_1d"])
    assert np.isnan(feats.iloc[3][f"ma_ratio_5"])
    assert np.isfinite(feats.iloc[4][f"ma_ratio_5"])



def test_feature_scaler_standardizes_constant_and_integer_columns():
    frame = pd.DataFrame({"count": [0, 1, 2], "constant": [5.0, 5.0, 5.0]})
    scaler = FeatureScaler.fit(frame, ["count", "constant"])
    transformed = scaler.transform(frame)
    assert transformed[["count", "constant"]].dtypes.tolist() == [
        np.dtype("float64"),
        np.dtype("float64"),
    ]
    assert transformed["count"].tolist() == pytest.approx([-1.22474487, 0.0, 1.22474487])
    assert transformed["constant"].tolist() == pytest.approx([0.0, 0.0, 0.0])

def test_target_return_matches_next_session_close_ratio():
    prices = _prices(n=6)
    feats = price_features(prices)
    with_target = add_target(feats)
    close = prices.sort_values("time")["close"].to_numpy()
    expected = close[1] / close[0] - 1.0
    assert with_target.iloc[0]["target_return"] == pytest.approx(expected)
    assert with_target.iloc[0]["target_date"] == pd.Timestamp(
        prices.sort_values("time")["time"].iloc[1]
    ).normalize()
    # Last row has no next session.
    assert np.isnan(with_target.iloc[-1]["target_return"])


# --- train-only thresholds ------------------------------------------------------------


def test_thresholds_fit_on_train_only_and_apply_without_refit():
    train_returns = np.linspace(-0.05, 0.05, 61)  # symmetric, deterministic
    q_low, q_high = fit_thresholds(train_returns, low_q=1 / 3, high_q=2 / 3)
    assert q_low == pytest.approx(np.quantile(train_returns, 1 / 3))
    assert q_high == pytest.approx(np.quantile(train_returns, 2 / 3))

    # Applying to unrelated test returns must reuse the fitted thresholds, not refit.
    test_returns = np.array([q_low - 1.0, 0.0, q_high + 1.0, np.nan])
    labels = apply_labels(test_returns, (q_low, q_high))
    assert labels[0] == "DOWN"
    assert labels[1] == "FLAT"
    assert labels[2] == "UP"
    assert labels[3] is None


def test_label_panel_keeps_missing_target_as_na():
    prices = _prices(n=8)
    panel = build_panel(prices)
    thresholds = fit_thresholds(panel["target_return"].dropna().to_numpy())
    labeled = label_panel(panel, thresholds)
    # Last row per ticker has NaN target_return -> label stays NA.
    assert pd.isna(labeled.iloc[-1]["target_label"])
    assert labeled["target_label"].notna().sum() == panel["target_return"].notna().sum()


# --- sentiment aggregation / no-news distinction --------------------------------------


def test_no_news_day_is_distinguishable_from_neutral_news_day():
    prices = _prices(n=12)
    # One neutral-news article on 2021-01-06 (before cutoff).
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "published_at": ["2021-01-06T09:00:00+07:00"],
            "prob_negative": [0.2],
            "prob_neutral": [0.6],
            "prob_positive": [0.2],
        }
    )
    panel = assemble(prices, news)
    news_row = panel[panel["observation_date"] == pd.Timestamp("2021-01-06")].iloc[0]
    quiet_row = panel[panel["observation_date"] == pd.Timestamp("2021-01-07")].iloc[0]

    assert news_row["has_news"] == 1 and news_row["news_count"] == 1
    assert news_row["sent_prob_neutral_mean"] == pytest.approx(0.6)

    # No-news day: neutral prior, zeroed counts, has_news 0 -> clearly distinct.
    assert quiet_row["has_news"] == 0 and quiet_row["news_count"] == 0
    assert quiet_row["sent_prob_neutral_mean"] == pytest.approx(1.0)
    assert quiet_row["sent_prob_positive_mean"] == pytest.approx(0.0)


def test_daily_sentiment_drops_unmapped_and_invalid_rows():
    aligned = pd.DataFrame(
        {
            "ticker": ["FPT", "FPT"],
            "observation_date": [pd.NaT, pd.Timestamp("2021-01-05")],
            "mapping_status": ["unmapped", "same_session"],
            "prob_negative": [0.1, 0.1],
            "prob_neutral": [0.1, 0.2],
            "prob_positive": [0.8, 0.7],
        }
    )
    daily = daily_sentiment(aligned)
    assert len(daily) == 1
    assert daily.iloc[0]["observation_date"] == pd.Timestamp("2021-01-05")
    assert daily.iloc[0]["sent_pos_ratio"] == pytest.approx(1.0)


# --- chronological / walk-forward splits ----------------------------------------------


def test_chronological_split_is_ordered_and_non_overlapping():
    prices = _prices(n=30)
    panel = build_panel(prices)
    split = chronological_split(panel, val_frac=0.2, test_frac=0.2)
    train, val, test = split.frames(panel)
    assert train["target_date"].max() < val["target_date"].min()
    assert val["target_date"].max() < test["target_date"].min()
    # Only rows with a realized target belong to an evaluation partition.
    idx = np.concatenate([split.train, split.val, split.test])
    assert len(idx) == len(np.unique(idx)) == panel["target_date"].notna().sum()


def test_split_keeps_same_target_day_rows_together():
    prices = pd.concat([_prices("FPT", n=20), _prices("VNM", n=20)], ignore_index=True)
    panel = build_panel(prices)
    split = chronological_split(panel, val_frac=0.2, test_frac=0.2)
    tagged = split.assign_split(panel).dropna(subset=["target_date"])
    per_date = tagged.groupby("target_date")["split"].nunique()
    assert (per_date == 1).all()


def test_walk_forward_windows_are_chronological_and_expanding():
    prices = _prices(n=40)
    panel = build_panel(prices)
    windows = walk_forward_windows(
        panel, n_windows=3, test_size=4, val_size=2, min_train=10, expanding=True
    )
    assert len(windows) == 3
    prev_train = -1
    for w in windows:
        tr, va, te = w.frames(panel)
        assert tr["observation_date"].max() < va["observation_date"].min()
        assert va["observation_date"].max() < te["observation_date"].min()
        # Expanding: each window's train set is at least as large as the previous.
        assert len(tr) >= prev_train
        prev_train = len(tr)



def test_walk_forward_windows_rejects_partial_request():
    panel = build_panel(_prices(n=20))
    with pytest.raises(ValueError, match="build 5 walk-forward windows"):
        walk_forward_windows(panel, n_windows=5, test_size=4, val_size=2, min_train=10)

# --- models ---------------------------------------------------------------------------


def _labeled_panel(n: int = 40) -> pd.DataFrame:
    prices = _prices(n=n)
    panel = build_panel(prices)
    thresholds = fit_thresholds(panel["target_return"].dropna().to_numpy())
    return label_panel(panel, thresholds)


def test_make_sequences_shapes_and_no_cross_ticker_mixing():
    prices = pd.concat([_prices("FPT", n=30), _prices("VNM", n=30)], ignore_index=True)
    panel = build_panel(prices)
    thresholds = fit_thresholds(panel["target_return"].dropna().to_numpy())
    panel = label_panel(panel, thresholds)
    cols = price_feature_columns()
    X, y, meta = make_sequences(panel, cols, seq_len=5)
    assert X.shape[1:] == (5, len(cols))
    assert X.shape[0] == y.shape[0] == len(meta)
    assert np.isfinite(X).all()  # no NaN windows leak through
    assert set(meta["ticker"]) <= {"FPT", "VNM"}


def test_price_lstm_output_shape():
    cols = price_feature_columns()
    panel = _labeled_panel()
    X, y, _ = make_sequences(panel, cols, seq_len=5)
    model = PriceLSTM(n_features=len(cols), hidden=8)
    logits = model(__import__("torch").as_tensor(X))
    assert logits.shape == (X.shape[0], 3)


def test_two_branch_lstm_output_shape():
    import torch

    price_cols = price_feature_columns()
    sent_cols = ["sent_prob_negative_mean", "sent_prob_positive_mean", "sent_pos_minus_neg"]
    panel = _labeled_panel()
    Xp, Xs, y, meta = make_two_branch_sequences(panel, price_cols, sent_cols, seq_len=5)
    assert Xp.shape[0] == Xs.shape[0] == y.shape[0] == len(meta)
    model = PriceSentimentLSTM(n_price=len(price_cols), n_sent=len(sent_cols), hidden=8)
    logits = model(torch.as_tensor(Xp), torch.as_tensor(Xs))
    assert logits.shape == (Xp.shape[0], 3)


def test_two_branch_forward_rejects_mismatched_windows():
    import torch

    model = PriceSentimentLSTM(n_price=2, n_sent=2, hidden=4)
    price = torch.zeros((3, 5, 2))
    sent = torch.zeros((3, 4, 2))  # wrong seq_len
    with pytest.raises(ValueError, match="seq_len"):
        model(price, sent)


def test_baselines_are_deterministic():
    labels = ["UP", "UP", "UP", "FLAT", "DOWN"]
    majority = MajorityBaseline().fit(labels)
    assert majority.predict(4).tolist() == [2, 2, 2, 2]  # UP == id 2

    rand_a = RandomBaseline(seed=7).fit(labels).predict(10)
    rand_b = RandomBaseline(seed=7).fit(labels).predict(10)
    np.testing.assert_array_equal(rand_a, rand_b)  # same seed -> same draws


def test_fit_lstm_runs_real_gradient_steps():
    cols = price_feature_columns()
    panel = _labeled_panel()
    X, y, _ = make_sequences(panel, cols, seq_len=5)
    set_seed(0)
    model = PriceLSTM(n_features=len(cols), hidden=8)
    history = fit_lstm(model, X, y, epochs=6, lr=0.05, batch_size=8, seed=0)
    losses = history["loss_history"]
    assert len(losses) == 6
    assert history["epochs_run"] == 6
    assert all(np.isfinite(losses))
    # A real optimizer on separable-ish synthetic data reduces the loss.
    assert losses[-1] < losses[0]
    # Without validation data there is nothing to select on.
    assert history["best_val_macro_f1"] is None


def test_fit_lstm_restores_the_best_validation_checkpoint():
    cols = price_feature_columns()
    panel = _labeled_panel(n=90)
    X, y, meta = make_sequences(panel, cols, seq_len=5)
    cut = len(y) // 2
    set_seed(0)
    model = PriceLSTM(n_features=len(cols), hidden=8)
    history = fit_lstm(
        model,
        X[:cut],
        y[:cut],
        X_val=X[cut:],
        y_val=y[cut:],
        epochs=12,
        lr=0.05,
        batch_size=8,
        patience=3,
        seed=0,
    )
    scores = history["val_macro_f1_history"]
    assert scores, "validation data must be scored once per epoch"
    assert history["best_val_macro_f1"] == pytest.approx(max(scores))
    assert scores[history["best_epoch"] - 1] == pytest.approx(max(scores))
    # The weights left in the model must be the selected epoch's, not the last epoch's.
    restored = evaluate_predictions(y[cut:], predict_lstm(model, X[cut:]).argmax(axis=1))
    assert restored["macro_f1"] == pytest.approx(max(scores))
    # Early stopping must not run past the patience budget.
    assert history["epochs_run"] <= history["best_epoch"] + 3


def test_feature_scaler_uses_training_rows_and_preserves_schema():
    cols = price_feature_columns()
    train = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in cols})
    holdout = pd.DataFrame({col: [100.0] for col in cols})
    scaler = FeatureScaler.fit(train, cols)
    transformed = scaler.transform(holdout)
    assert transformed[cols[0]].iloc[0] > 90
    assert np.allclose(scaler.transform(train)[cols].mean().to_numpy(), 0.0)


def test_panel_without_news_has_stable_numeric_sentiment_dtypes():
    panel = build_panel(_prices(n=12))
    assert all(str(panel[col].dtype) == "float64" for col in [
        "sent_prob_negative_mean",
        "sent_prob_neutral_mean",
        "sent_prob_positive_mean",
        "sent_dispersion",
    ])


def test_session_as_of_is_timezone_aware_even_with_nat():
    values = cal.session_as_of([pd.Timestamp("2021-01-05"), pd.NaT])
    assert str(values.dt.tz) == "Asia/Ho_Chi_Minh"
    assert pd.isna(values.iloc[1])


# --- timezone-unified point-in-time alignment ------------------------------------------


def test_trading_sessions_normalizes_naive_and_aware_prices_to_same_calendar():
    naive = _prices("FPT", n=5)
    aware = naive.copy()
    aware["time"] = aware["time"].dt.tz_localize("Asia/Ho_Chi_Minh")
    naive_sessions = cal.trading_sessions(naive)
    aware_sessions = cal.trading_sessions(aware)
    np.testing.assert_array_equal(naive_sessions["FPT"], aware_sessions["FPT"])


def test_trading_sessions_rejects_duplicate_normalized_dates():
    prices = _prices("FPT", n=3)
    dup = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate normalized"):
        cal.trading_sessions(dup)


def test_price_features_rejects_duplicate_normalized_session_dates():
    prices = _prices("FPT", n=6)
    dup = pd.concat([prices, prices.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate normalized session dates"):
        price_features(dup)


# --- probability boundary validation ---------------------------------------------------


def test_daily_sentiment_rejects_out_of_bounds_and_bad_sum_probabilities():
    aligned = pd.DataFrame(
        {
            "ticker": ["FPT", "FPT", "FPT", "FPT"],
            "observation_date": [pd.Timestamp("2021-01-05")] * 4,
            "mapping_status": ["same_session"] * 4,
            "prob_negative": [1.5, -0.1, 0.5, 0.2],
            "prob_neutral": [0.2, 0.5, 0.5, 0.3],
            "prob_positive": [0.3, 0.6, 0.5, 0.5],
        }
    )
    daily = daily_sentiment(aligned)
    assert len(daily) == 1
    assert daily.iloc[0]["news_count"] == 1
    assert daily.iloc[0]["sent_prob_negative_mean"] == pytest.approx(0.2)


# --- forecasting model evaluation --------------------------------------------------------


def test_evaluate_predictions_perfect_predictions_score_one():
    result = evaluate_predictions([0, 1, 2, 1], [0, 1, 2, 1])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)


def test_evaluate_predictions_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        evaluate_predictions([0, 1], [0])


def test_evaluate_predictions_rejects_labels_outside_fixed_ids():
    with pytest.raises(ValueError, match="fixed ids"):
        evaluate_predictions([0, 5], [0, 1])


# --- trailing sentiment features ------------------------------------------------------


def test_rolling_sentiment_never_reads_a_later_session():
    """A news spike on the last session must not alter any earlier row."""
    panel = build_panel(_prices(n=20))
    quiet = add_rolling_sentiment(panel)

    spiked = panel.copy()
    last = spiked.index[-1]
    spiked.loc[last, ["sent_pos_minus_neg", "news_count", "has_news"]] = [0.9, 7, 1]
    loud = add_rolling_sentiment(spiked)

    for col in ROLLING_SENTIMENT_COLUMNS:
        np.testing.assert_allclose(
            quiet[col].to_numpy()[:-1], loud[col].to_numpy()[:-1], atol=1e-12
        )
    # The spiked session itself must react, otherwise the feature is inert.
    assert loud["sent_pos_minus_neg_roll"].iloc[-1] > quiet["sent_pos_minus_neg_roll"].iloc[-1]


def test_rolling_sentiment_decays_after_news_stops():
    """Trailing news must fade as quiet sessions accumulate, not persist flat."""
    panel = build_panel(_prices(n=20))
    panel.loc[5, ["sent_pos_minus_neg", "news_count", "has_news"]] = [1.0, 3, 1]
    rolled = add_rolling_sentiment(panel)
    ewm = rolled["sent_pos_minus_neg_ewm"].to_numpy()
    assert ewm[5] > ewm[6] > ewm[7]
    assert rolled["has_news_roll_mean"].iloc[5] > rolled["has_news_roll_mean"].iloc[-1]


# --- walk-forward experiment contract -------------------------------------------------


def test_experiment_windows_are_chronological_and_share_test_rows():
    """Every arm must be scored on the same test rows, strictly after training."""
    prices = pd.concat([_prices("FPT", n=120), _prices("VNM", n=120)], ignore_index=True)
    panel = assemble(prices)
    cfg = ForecastConfig(
        n_windows=2, test_size=6, val_size=6, epochs=2, batch_size=32, seeds=(42,)
    )
    record = run_experiment(panel, cfg=cfg)

    assert len(record["windows"]) == 2
    assert record["chance_level"] == pytest.approx(1 / 3)
    previous_test_end = None
    for window in record["windows"]:
        train_end = window["train_dates"][1]
        assert train_end < window["val_dates"][0] < window["test_dates"][0]
        if previous_test_end is not None:
            assert window["test_dates"][0] > previous_test_end
        previous_test_end = window["test_dates"][1]

        sizes = {
            arm: run["seed_runs"][0]["n"] for arm, run in window["metrics"].items()
        }
        assert len(set(sizes.values())) == 1, sizes
        assert sizes["lstm_price"] == window["sizes"]["test"]


def test_experiment_thresholds_are_refitted_per_window():
    """Train-only thresholds must move with the window, not be fitted once globally."""
    prices = pd.concat([_prices("FPT", n=140), _prices("VNM", n=140)], ignore_index=True)
    panel = assemble(prices)
    cfg = ForecastConfig(
        n_windows=3, test_size=6, val_size=6, epochs=2, batch_size=32, seeds=(42,)
    )
    record = run_experiment(panel, cfg=cfg)
    thresholds = [tuple(w["thresholds"]) for w in record["windows"]]
    assert len(set(thresholds)) > 1, thresholds
    for low, high in thresholds:
        assert low < high


def test_classical_baseline_expands_probabilities_to_all_three_classes():
    """A train block missing a class must still yield a 3-column probability matrix."""
    X = np.zeros((12, 5, 2), dtype=float)
    X[:, -1, 0] = np.arange(12, dtype=float)
    y = np.array([1, 2] * 6)  # DOWN (id 0) never appears in training
    model = ClassicalBaseline(seed=0).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (12, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    # The absent class must carry exactly zero mass, not a shifted column.
    np.testing.assert_allclose(probs[:, 0], 0.0)
    assert probs[:, 1:].sum() > 0


def test_date_block_bootstrap_brackets_a_real_difference():
    """A uniformly better arm must produce an interval strictly above zero."""
    dates = pd.to_datetime(pd.bdate_range("2024-01-01", periods=40)).repeat(5)
    truth = np.tile([0, 1, 2, 0, 1], 40)
    good = truth.copy()
    bad = np.roll(truth, 1)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "target_date": dates,
                    "seed": 42,
                    "arm": arm,
                    "y_true": [TREND_LABELS[i] for i in truth],
                    "y_pred": [TREND_LABELS[i] for i in pred],
                }
            )
            for arm, pred in (("treated", good), ("control", bad))
        ],
        ignore_index=True,
    )
    out = _date_block_bootstrap(frame, "treated", "control", samples=500, seed=1)
    assert out["macro_f1"]["n_blocks"] == 40
    assert out["macro_f1"]["mean"] > 0.3
    assert out["macro_f1"]["low"] > 0.0
    # Two arms making the same predictions must instead give a zero-width interval.
    identical = frame[frame["arm"] == "treated"]
    identical = pd.concat(
        [identical, identical.assign(arm="control")], ignore_index=True
    )
    same = _date_block_bootstrap(
        identical, "treated", "control", samples=500, seed=1
    )
    assert same["macro_f1"]["low"] <= 0.0 <= same["macro_f1"]["high"]
    assert same["macro_f1"]["mean"] == pytest.approx(0.0)


def test_experiment_is_reproducible_across_repeated_runs():
    """Two runs in one process must agree; weight init must not inherit RNG state.

    Seeding only inside the training loop leaves initialisation drawing from
    whatever global torch state earlier arms and windows left behind, which makes
    a window's result depend on how much randomness was consumed before it.
    """
    prices = pd.concat([_prices("FPT", n=120), _prices("VNM", n=120)], ignore_index=True)
    panel = assemble(prices)
    cfg = ForecastConfig(
        n_windows=2, test_size=6, val_size=6, epochs=4, batch_size=32, seeds=(42,)
    )
    first = run_experiment(panel, cfg=cfg)
    second = run_experiment(panel, cfg=cfg)

    for arm in ("lstm_price", "lstm_price_sentiment"):
        a = [w["metrics"][arm]["window_mean"] for w in first["windows"]]
        b = [w["metrics"][arm]["window_mean"] for w in second["windows"]]
        assert a == b, f"{arm} is not reproducible: {a} vs {b}"


def test_experiment_writes_every_artifact(tmp_path):
    """The reportable artifacts and both interval flavours must be produced."""
    prices = pd.concat([_prices("FPT", n=120), _prices("VNM", n=120)], ignore_index=True)
    panel = assemble(prices)
    cfg = ForecastConfig(
        n_windows=2,
        test_size=6,
        val_size=6,
        epochs=2,
        batch_size=32,
        seeds=(42, 43),
        bootstrap_samples=200,
    )
    record = run_experiment(panel, cfg=cfg, output_dir=tmp_path)

    for name in (
        "forecast_results.json",
        "forecast_metrics.csv",
        "forecast_stratified.csv",
        "forecast_predictions.csv",
    ):
        assert (tmp_path / name).exists(), name

    delta = record["ablation"]["lstm_price_sentiment__minus__lstm_price"]["macro_f1"]
    assert len(delta["per_window"]) == 2
    assert delta["window_bootstrap"]["low"] is not None
    assert delta["date_block_bootstrap"]["n_blocks"] == 2 * 6

    # Predictions must cover every arm and every seed, not just the first.
    preds = pd.read_csv(tmp_path / "forecast_predictions.csv")
    assert set(preds["seed"]) == {42, 43}
    assert set(preds["arm"]) == {name for name, _, _ in LADDER}


def test_information_gain_decomposition_is_exact_and_guards_config_drift(tmp_path):
    """Matched neutral controls must preserve both fused and price-only predictions."""
    prices = pd.concat([_prices("FPT", n=120), _prices("VNM", n=120)], ignore_index=True)
    news = pd.DataFrame(
        {
            "ticker": "FPT",
            "published_at": pd.to_datetime(prices["time"].unique()).strftime(
                "%Y-%m-%d 09:00:00+07:00"
            ),
            "prob_negative": 0.2,
            "prob_neutral": 0.3,
            "prob_positive": 0.5,
        }
    )
    cfg = ForecastConfig(
        n_windows=2, test_size=6, val_size=6, epochs=3, batch_size=32, seeds=(42,)
    )
    neutral_news = news.assign(
        prob_negative=0.0,
        prob_neutral=1.0,
        prob_positive=0.0,
    )
    real_dir, control_dir = tmp_path / "real", tmp_path / "control"
    run_experiment(
        assemble(prices, news),
        cfg=cfg,
        output_dir=real_dir,
        provenance={
            "news_sentiment": {"mode": "real", "source_hash": "news-v1", "rows": len(news)},
            "prices_hash": "prices-v1",
        },
    )
    run_experiment(
        assemble(prices, neutral_news),
        cfg=cfg,
        output_dir=control_dir,
        provenance={
            "news_sentiment": {
                "mode": "neutral_prior",
                "source_hash": "news-v1",
                "rows": len(news),
            },
            "prices_hash": "prices-v1",
        },
    )

    report = compare_information_gain(real_dir, control_dir)
    effect = report["effects"]["macro_f1"]
    assert (
        effect["architecture_and_news_presence_volume_effect"]
        + effect["information_gain"]
        == pytest.approx(effect["naive_delta"])
    )
    assert len(effect["information_gain_per_window"]) == 2

    # Config equality remains mandatory even when all prediction rows match.
    control_results_path = control_dir / "forecast_results.json"
    original_control_results = json.loads(
        control_results_path.read_text(encoding="utf-8")
    )
    config_drift = json.loads(control_results_path.read_text(encoding="utf-8"))
    config_drift["config"]["epochs"] += 1
    control_results_path.write_text(json.dumps(config_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="different configs"):
        compare_information_gain(real_dir, control_dir)
    control_results_path.write_text(json.dumps(original_control_results), encoding="utf-8")

    # Window records are keyed by their identifier, not their JSON list position.
    control_results = json.loads(control_results_path.read_text(encoding="utf-8"))
    control_results["windows"].reverse()
    control_results_path.write_text(json.dumps(control_results), encoding="utf-8")
    assert compare_information_gain(real_dir, control_dir) == report

    duplicated_results = json.loads(control_results_path.read_text(encoding="utf-8"))
    duplicated_results["windows"].append(duplicated_results["windows"][0])
    control_results_path.write_text(json.dumps(duplicated_results), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates a window identifier"):
        compare_information_gain(real_dir, control_dir)
    control_results_path.write_text(json.dumps(control_results), encoding="utf-8")

    # A run with observed probabilities cannot be its own neutralized control.
    with pytest.raises(ValueError, match="preserve the scored-news rows"):
        compare_information_gain(real_dir, real_dir)

    # Matching parameters alone are insufficient if the underlying prices changed.
    price_drift = tmp_path / "price-drift"
    run_experiment(
        assemble(prices, neutral_news),
        cfg=cfg,
        output_dir=price_drift,
        provenance={
            "news_sentiment": {
                "mode": "neutral_prior",
                "source_hash": "news-v1",
                "rows": len(news),
            },
            "prices_hash": "prices-v2",
        },
    )
    with pytest.raises(ValueError, match="different price provenance"):
        compare_information_gain(real_dir, price_drift)

    # A stale run with different observations cannot supply a paired information gain.
    control_predictions = pd.read_csv(control_dir / "forecast_predictions.csv")
    row = control_predictions["arm"].eq("lstm_price_sentiment").idxmax()
    original_target = control_predictions.loc[row, "y_true"]
    control_predictions.loc[row, "y_true"] = (
        "UP" if original_target != "UP" else "DOWN"
    )
    control_predictions.to_csv(control_dir / "forecast_predictions.csv", index=False)
    with pytest.raises(ValueError, match="different prediction keys"):
        compare_information_gain(real_dir, control_dir)

    # The control's price-only arm must remain identical, not merely have matching keys.
    control_predictions.loc[row, "y_true"] = original_target
    price_row = control_predictions["arm"].eq("lstm_price").idxmax()
    original_prediction = control_predictions.loc[price_row, "y_pred"]
    control_predictions.loc[price_row, "y_pred"] = (
        "UP" if original_prediction != "UP" else "DOWN"
    )
    control_predictions.to_csv(control_dir / "forecast_predictions.csv", index=False)
    with pytest.raises(ValueError, match="different price-arm predictions"):
        compare_information_gain(real_dir, control_dir)

    # Stale artifacts must report a documented pairing error, not leak a KeyError.
    control_predictions.drop(columns="has_news").to_csv(
        control_dir / "forecast_predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="required pairing columns"):
        compare_information_gain(real_dir, control_dir)

    control_predictions.drop(columns="y_pred").to_csv(
        control_dir / "forecast_predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="required pairing columns"):
        compare_information_gain(real_dir, control_dir)


def test_tft_preserves_feature_identity():
    """Each variable gets its own embedding, so the encoder sees more than one signal.

    A single shared ``nn.Linear(1, hidden)`` across all features makes the
    variable-selection sum telescope to ``w * sum_j(alpha_j * x_j) + b``, because the
    softmax weights sum to one. The encoder input then spans two dimensions -- one
    signal plus bias -- no matter how large ``hidden`` is or how many features are
    supplied, which silently destroys the price-vs-price+sentiment ablation. A
    forward-shape assertion cannot see this; the rank can.
    """
    import torch

    set_seed(0)
    for n_features in (6, 11):
        model = TemporalFusionClassifier(n_features, hidden=32)
        window = torch.randn(64, 5, n_features)

        embedded = window.unsqueeze(-1) * model.feature_weight + model.feature_bias
        weights = torch.softmax(model.variable_selection(window), dim=-1)
        encoder_input = (embedded * weights.unsqueeze(-1)).sum(dim=2)

        rank = int(torch.linalg.matrix_rank(encoder_input.reshape(-1, 32)))
        assert rank == n_features, f"expected rank {n_features} for the encoder input, got {rank}"

        # Zeroing one variable must change the prediction; under a shared embedding a
        # feature only shifts a weighted average and can be masked by the others.
        masked = window.clone()
        masked[:, :, 0] = 0.0
        model.eval()
        with torch.no_grad():
            assert not torch.allclose(model(window), model(masked))

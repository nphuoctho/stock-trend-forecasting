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
    cross_sectional_excess,
    fit_thresholds,
    label_panel,
    retarget_horizon,
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
    window = np.repeat([1, 2], 100)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "target_date": dates,
                    "window": window,
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
    assert out["macro_f1"]["n_window_strata"] == 2
    assert out["macro_f1"]["mean"] > 0.3
    assert out["macro_f1"]["low"] > 0.0
    assert out["macro_f1"]["one_sided_p_le_zero"] == 0.0
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


def test_date_block_bootstrap_centres_on_the_window_averaged_delta():
    """The interval must bracket the estimand the report quotes, not a pooled one.

    macro-F1 is non-linear in the confusion matrix, so a delta computed on one
    pooled matrix differs from the mean of the per-window deltas. The headline
    ``information_gain`` is the window-averaged quantity; if the bootstrap centres
    somewhere else the published point estimate can sit outside its own interval.
    """
    rng = np.random.default_rng(0)
    blocks = []
    # Two windows with deliberately different class mixes so pooling and averaging
    # disagree: window 1 is UP-heavy, window 2 is DOWN-heavy.
    for window_id, weights in ((1, [0.15, 0.25, 0.60]), (2, [0.60, 0.25, 0.15])):
        dates = pd.to_datetime(pd.bdate_range("2024-01-01", periods=30)).repeat(6)
        truth = rng.choice(3, size=len(dates), p=weights)
        treated = np.where(rng.random(len(dates)) < 0.55, truth, rng.choice(3, len(dates)))
        control = np.where(rng.random(len(dates)) < 0.35, truth, rng.choice(3, len(dates)))
        for arm, pred in (("treated", treated), ("control", control)):
            blocks.append(
                pd.DataFrame(
                    {
                        "target_date": dates,
                        "window": window_id,
                        "seed": 42,
                        "arm": arm,
                        "y_true": [TREND_LABELS[i] for i in truth],
                        "y_pred": [TREND_LABELS[i] for i in pred],
                    }
                )
            )
    frame = pd.concat(blocks, ignore_index=True)

    ids = {label: i for i, label in enumerate(TREND_LABELS)}

    def window_macro_f1(block: pd.DataFrame, arm: str) -> float:
        side = block[block["arm"] == arm]
        return evaluate_predictions(
            side["y_true"].map(ids).to_numpy(), side["y_pred"].map(ids).to_numpy()
        )["macro_f1"]

    expected = float(
        np.mean(
            [
                window_macro_f1(block, "treated") - window_macro_f1(block, "control")
                for _, block in frame.groupby("window", sort=True)
            ]
        )
    )
    out = _date_block_bootstrap(frame, "treated", "control", samples=800, seed=3)
    assert out["macro_f1"]["mean"] == pytest.approx(expected, abs=1e-9)
    assert out["macro_f1"]["low"] <= out["macro_f1"]["mean"] <= out["macro_f1"]["high"]


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


def test_tft_arm_survives_the_refit_load_round_trip(tmp_path):
    """A reloaded TFT arm reproduces the probabilities it produced before saving.

    refit_arm derives the input width from the concatenated feature matrix while
    load_arm recomputes it from the manifest's feature lists. If those two ever
    disagree the checkpoint loads into a mis-shaped net, which surfaces as silently
    different live predictions rather than as an error.
    """
    from stf.forecasting.experiment import ForecastConfig
    from stf.forecasting.serve import load_arm, predict_latest, refit_arm

    prices = pd.concat([_prices("FPT", n=90), _prices("VNM", n=90)], ignore_index=True)
    news = pd.DataFrame(
        {
            "ticker": ["FPT", "VNM"] * 20,
            "published_at": [
                f"2021-0{1 + i // 20}-{1 + i % 20:02d}T09:00:00+07:00" for i in range(40)
            ],
            "prob_negative": np.linspace(0.05, 0.5, 40),
            "prob_neutral": np.linspace(0.5, 0.3, 40),
            "prob_positive": np.linspace(0.45, 0.2, 40),
        }
    )
    panel = assemble(prices, news)
    cfg = ForecastConfig(epochs=1, patience=1, seeds=(42,), hidden=8)

    refit_arm(panel, "tft_price_sentiment", cfg=cfg, output_dir=tmp_path / "arm")
    arm = load_arm(tmp_path / "arm")
    first = predict_latest(panel, arm)
    second = predict_latest(panel, load_arm(tmp_path / "arm"))

    probabilities = ["prob_down", "prob_flat", "prob_up"]
    assert arm.family == "tft"
    pd.testing.assert_frame_equal(first[probabilities], second[probabilities])
    assert np.allclose(first[probabilities].sum(axis=1), 1.0)

    # The served probabilities must come from the persisted weights, not from a
    # freshly initialised net: corrupting the checkpoint has to change them.
    import torch

    state_files = sorted((tmp_path / "arm").glob("*.pt"))
    assert state_files, "refit_arm persisted no weights"
    for state_file in state_files:
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        torch.save({k: torch.zeros_like(v) for k, v in state.items()}, state_file)
    corrupted = predict_latest(panel, load_arm(tmp_path / "arm"))
    assert not np.allclose(
        first[probabilities].to_numpy(), corrupted[probabilities].to_numpy()
    )


def test_cross_sectional_excess_removes_the_shared_market_move():
    """The excess target must price a ticker against its own session, not the level."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"] * 2,
            "target_date": list(np.repeat(dates, 3)),
            # Session 1: a +2% market day with one laggard. Session 2: a -1% market day.
            "target_return": [0.03, 0.02, 0.01, -0.02, -0.01, 0.00],
        }
    )
    out = cross_sectional_excess(panel)

    # The raw frame is untouched; the excess target sums to zero inside each session.
    assert panel["target_return"].tolist() == [0.03, 0.02, 0.01, -0.02, -0.01, 0.00]
    per_date = out.groupby("target_date")["target_return"].sum()
    assert np.allclose(per_date.to_numpy(), 0.0)
    assert out.loc[0, "target_return"] == pytest.approx(0.01)
    assert out.loc[2, "target_return"] == pytest.approx(-0.01)

    # A ticker that rose less than the market on an up day is a DOWN case now,
    # which is exactly the relabelling the excess target is for.
    thresholds = fit_thresholds(out["target_return"].dropna().to_numpy())
    assert apply_labels(np.array([out.loc[2, "target_return"]]), thresholds)[0] == "DOWN"


def test_cross_sectional_excess_drops_single_ticker_sessions():
    """One stock is not a cross-section; its excess return must be missing, not zero."""
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA"],
            "target_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "target_return": [0.02, 0.00, 0.05],
        }
    )
    out = cross_sectional_excess(panel)
    assert out.loc[0, "target_return"] == pytest.approx(0.01)
    assert pd.isna(out.loc[2, "target_return"])


def test_excess_target_mode_changes_the_labels_the_experiment_scores():
    """`target_mode='excess'` must reach the labels, not just the config record."""
    # Three tickers whose returns share a common move plus a ticker-specific wobble;
    # without divergence the cross-section would be degenerate and prove nothing.
    frames = []
    for offset, ticker in enumerate(("AAA", "BBB", "CCC")):
        prices = _prices(ticker, n=150)
        wobble = np.cos(np.arange(len(prices)) + offset * 2.0) * (1.0 + offset)
        frames.append(prices.assign(close=prices["close"].to_numpy() + wobble))
    panel = assemble(pd.concat(frames, ignore_index=True))
    base = ForecastConfig(
        seq_len=3, n_windows=2, test_size=10, val_size=10, epochs=1, seeds=(42,)
    )
    excess = ForecastConfig(
        seq_len=3,
        n_windows=2,
        test_size=10,
        val_size=10,
        epochs=1,
        seeds=(42,),
        target_mode="excess",
    )
    raw_record = run_experiment(panel, cfg=base)
    excess_record = run_experiment(panel, cfg=excess)

    assert raw_record["config"]["target_mode"] == "raw"
    assert excess_record["config"]["target_mode"] == "excess"
    # Terciles of an excess return straddle zero by construction, so the thresholds
    # must move; identical thresholds would mean the mode never reached the labels.
    assert raw_record["windows"][0]["thresholds"] != excess_record["windows"][0]["thresholds"]




def test_retarget_horizon_looks_the_requested_number_of_sessions_ahead():
    """The label must span h sessions, and the last h rows must lose their target."""
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"] * 5,
            "observation_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
            ),
            "close": [100.0, 110.0, 121.0, 100.0, 50.0],
        }
    )
    out = retarget_horizon(panel, 2)
    # Row 0 spans 100 -> 121 over two sessions.
    assert out.loc[0, "target_return"] == pytest.approx(0.21)
    assert out.loc[0, "target_date"] == pd.Timestamp("2024-01-04")
    assert pd.isna(out.loc[3, "target_return"])
    assert pd.isna(out.loc[4, "target_return"])
    # h=1 must reproduce the plain next-session target.
    assert retarget_horizon(panel, 1).loc[0, "target_return"] == pytest.approx(0.10)


def test_retarget_horizon_never_mixes_tickers():
    """A ticker's target must not reach across into another ticker's prices."""
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "observation_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
            ),
            "close": [100.0, 110.0, 50.0, 40.0],
        }
    )
    out = retarget_horizon(panel, 1).set_index(["ticker", "observation_date"])
    assert out.loc[("AAA", pd.Timestamp("2024-01-02")), "target_return"] == pytest.approx(0.10)
    assert out.loc[("BBB", pd.Timestamp("2024-01-02")), "target_return"] == pytest.approx(-0.20)
    assert pd.isna(out.loc[("AAA", pd.Timestamp("2024-01-03")), "target_return"])


def test_forecast_config_rejects_a_non_positive_horizon():
    with pytest.raises(ValueError, match="horizon"):
        ForecastConfig(horizon=0)


def test_forecast_config_rejects_an_unknown_target_mode():
    with pytest.raises(ValueError, match="target_mode"):
        ForecastConfig(target_mode="relative")


# --- economic evaluation --------------------------------------------------------------


def _backtest_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two sessions where the cross-section is known, so the payoff is hand-checkable."""
    prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "time": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04"] * 2
            ),
            "close": [100.0, 110.0, 99.0, 100.0, 90.0, 99.0],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "volume": 1.0,
        }
    )
    predictions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "target_date": pd.to_datetime(
                ["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]
            ),
            "seed": 42,
            "arm": "lstm_price",
            "y_true": ["UP", "DOWN", "DOWN", "UP"],
            "y_pred": ["UP", "DOWN", "DOWN", "UP"],
        }
    )
    return prices, predictions


def test_backtest_book_is_cash_neutral_and_pays_the_cross_sectional_spread():
    """A perfect forecast must earn the long-short spread, not the market move."""
    prices, predictions = _backtest_inputs()
    returns = realized_returns(prices)
    book = daily_book(predictions, returns, cost_bps=0.0)

    # Session 1: AAA +10%, BBB -10% -> long AAA / short BBB earns the full 20% spread.
    assert book.loc[0, "gross_return"] == pytest.approx(0.20)
    # Session 2: AAA -10%, BBB +10%; the forecast flips, so it earns the spread again
    # even though the equal-weighted basket moved 0% on both days.
    assert book.loc[1, "gross_return"] == pytest.approx(0.20)
    assert book["gross_exposure"].to_numpy() == pytest.approx(2.0)


def test_backtest_charges_turnover_against_drifted_holdings():
    """Drifted holdings and targets must be quoted on the same capital base.

    With one name per leg the weights are +-1. Day 1 opens from flat: turnover 2.
    AAA then returns +10% and BBB -10%, so the book gains 20% and NAV becomes 1.2.
    As fractions of the NEW NAV the holdings are +1.1/1.2 and -0.9/1.2. Day 2 flips
    to -1 AAA / +1 BBB, so turnover is
    |-1 - 11/12| + |1 + 3/4| = 23/12 + 7/4 = 3.66667.
    Leaving the holdings in yesterday's units would report 4.0 instead.
    """
    prices, predictions = _backtest_inputs()
    returns = realized_returns(prices)
    free = daily_book(predictions, returns, cost_bps=0.0)

    assert free.loc[0, "turnover"] == pytest.approx(2.0)
    assert free.loc[1, "turnover"] == pytest.approx(23.0 / 12.0 + 7.0 / 4.0)
    assert (free["cost"] == 0.0).all()

    charged = daily_book(predictions, returns, cost_bps=100.0)
    assert charged.loc[0, "cost"] == pytest.approx(2.0 * 100.0 / 10_000.0)
    assert charged["net_return"].to_numpy() == pytest.approx(
        (charged["gross_return"] - charged["cost"]).to_numpy()
    )


def test_backtest_holding_a_constant_target_still_costs_money():
    """The regression this guards: a static target is not a free position."""
    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "time": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "close": [100.0, 110.0, 121.0, 100.0, 90.0, 81.0],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "volume": 1.0,
        }
    )
    predictions = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "target_date": pd.to_datetime(
                ["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]
            ),
            "seed": 42,
            "arm": "lstm_price",
            "y_true": ["UP", "DOWN", "UP", "DOWN"],
            "y_pred": ["UP", "DOWN", "UP", "DOWN"],
        }
    )
    free = daily_book(predictions, realized_returns(prices), cost_bps=0.0)
    # Same target both sessions, yet the +10%/-10% drift must be rebalanced back.
    # The book gains 20%, so on the new NAV the holdings are +1.1/1.2 and -0.9/1.2
    # against a +1 / -1 target: |1 - 11/12| + |-1 + 3/4| = 1/12 + 1/4 = 1/3.
    # Differencing target weights would report 0.
    assert free.loc[1, "turnover"] == pytest.approx(1.0 / 3.0)
    charged = daily_book(predictions, realized_returns(prices), cost_bps=100.0)
    assert charged.loc[1, "cost"] > 0.0


def test_backtest_reports_the_residual_exposure_of_a_one_sided_day():
    """A one-sided view is directional; the book must record it, not hide it."""
    prices, predictions = _backtest_inputs()
    one_sided = predictions.copy()
    one_sided.loc[one_sided["target_date"] == pd.Timestamp("2024-01-03"), "y_pred"] = "UP"
    book = daily_book(one_sided, realized_returns(prices), cost_bps=0.0)

    assert book.loc[0, "gross_exposure"] == pytest.approx(0.5)
    # Two longs at 0.25 each: the book is net long 0.5, not market-neutral.
    assert book.loc[0, "net_exposure"] == pytest.approx(0.5)
    assert book.loc[0, "one_sided"] == pytest.approx(1.0)
    assert book.loc[1, "one_sided"] == pytest.approx(0.0)
    assert book.loc[1, "net_exposure"] == pytest.approx(0.0)


def test_backtest_drawdown_counts_the_first_session_against_initial_capital():
    """Equity starts at 1; a first-session loss is a drawdown, not a flat start."""
    assert _drawdown(np.array([-0.10])) == pytest.approx(-0.10)
    assert _drawdown(np.array([-0.10, 0.05])) == pytest.approx(-0.10)
    # Peak after a gain, then a fall: -20% from 1.10 down to 0.88.
    assert _drawdown(np.array([0.10, -0.20])) == pytest.approx(-0.20)
    assert _drawdown(np.array([0.10, 0.10])) == pytest.approx(0.0)


def test_backtest_benchmarks_separate_rebalancing_from_buy_and_hold():
    """Daily-rebalanced equal weight is not buy-and-hold; they must not be conflated."""
    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "time": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "close": [100.0, 120.0, 144.0, 100.0, 80.0, 64.0],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "volume": 1.0,
        }
    )
    returns = realized_returns(prices)
    dates = set(pd.to_datetime(["2024-01-03", "2024-01-04"]))
    bench = benchmark_series(returns, dates)

    # Rebalanced resets to 50/50 each session: mean of +20% and -20% is 0 twice.
    assert bench["equal_weight_rebalanced"] == pytest.approx([0.0, 0.0])
    # Buy-and-hold lets the winner grow: after day 1 the weights are 0.6/0.4, so the
    # second session returns 0.6*0.20 + 0.4*(-0.20) = +0.04.
    assert bench["buy_and_hold"] == pytest.approx([0.0, 0.04])


def test_backtest_rejects_predictions_that_never_meet_a_price():
    prices, predictions = _backtest_inputs()
    orphaned = predictions.assign(target_date=pd.Timestamp("2030-01-02"))
    with pytest.raises(ValueError, match="realized return"):
        daily_book(orphaned, realized_returns(prices), cost_bps=0.0)


# --- signal information coefficient ---------------------------------------------------


def test_ic_targets_separate_the_same_session_from_the_next_one():
    """The whole efficiency argument dies if `next` is not actually shifted forward."""
    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "observation_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 2
            ),
            "close": [100.0, 110.0, 99.0, 99.0, 100.0, 90.0, 99.0, 99.0],
        }
    )
    targets = build_return_targets(prices)
    aaa = targets[targets["ticker"] == "AAA"].reset_index(drop=True)

    # 2024-01-03 rose 10%; the row for 2024-01-03 owns that as its same-session
    # return, while the row for 2024-01-02 owns it as its next-session return.
    assert aaa.loc[1, "same_session_return"] == pytest.approx(0.10)
    assert aaa.loc[0, "next_session_return"] == pytest.approx(0.10)
    assert pd.isna(aaa.loc[0, "same_session_return"])
    assert pd.isna(aaa.loc[3, "next_session_return"])

    # AAA +10% while BBB -10% on the same session: the market leg is 0, so the raw
    # and excess returns coincide only because the cross-section is symmetric here.
    assert aaa.loc[1, "same_session_excess_return"] == pytest.approx(0.10)


def test_ic_excess_target_removes_a_common_move():
    """A session where both names move together must leave zero idiosyncratic return."""
    prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "observation_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
            ),
            "close": [100.0, 105.0, 200.0, 210.0],
        }
    )
    targets = build_return_targets(prices)
    moved = targets[targets["observation_date"] == pd.Timestamp("2024-01-03")]
    assert moved["same_session_return"].to_numpy() == pytest.approx(0.05)
    assert moved["same_session_excess_return"].to_numpy() == pytest.approx(0.0)


def test_information_coefficient_recovers_a_planted_rank_relationship():
    """A signal built to rank the next session must score a high positive IC."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(5)
    rows = []
    for date in dates:
        for ticker in ("AAA", "BBB", "CCC", "DDD"):
            score = rng.normal()
            rows.append(
                {
                    "ticker": ticker,
                    "observation_date": date,
                    "has_news": 1,
                    "sent_pos_minus_neg": score,
                }
            )
    panel = pd.DataFrame(rows)
    # Plant the relationship directly in the target instead of inventing prices.
    targets = panel[["ticker", "observation_date"]].copy()
    targets["next_session_excess_return"] = panel["sent_pos_minus_neg"] * 0.01
    targets["next_session_return"] = targets["next_session_excess_return"]
    targets["same_session_return"] = rng.normal(size=len(panel)) * 0.01
    targets["same_session_excess_return"] = targets["same_session_return"]

    report = information_coefficients(panel, targets, samples=200)
    planted = next(
        row for row in report["rows"]
        if row["signal"] == "sent_pos_minus_neg"
        and row["target"] == "next_session_excess_return"
    )
    unrelated = next(
        row for row in report["rows"]
        if row["signal"] == "sent_pos_minus_neg"
        and row["target"] == "same_session_return"
    )
    assert planted["ic"] == pytest.approx(1.0)
    assert planted["low"] > 0.9
    # The planted signal must not bleed into an independent target. The interval on
    # that target is not asserted: one draw of unrelated noise may exclude zero.
    assert abs(unrelated["ic"]) < 0.15
    assert unrelated["high"] - unrelated["low"] > 0.1

"""Idealized close-to-close diagnostic for a stored forecast run.

macro-F1 answers "how often is the class right"; it does not say whether the edge
is large or small in return units. This module converts the stored test predictions
into a dated long/short book so the classification metric can be read in basis
points.

**This is a diagnostic, not a tradable backtest.** Read the execution caveat below
before quoting any number from it.

Execution timing (the binding caveat)
-------------------------------------
The features for observation date ``t`` include ``close_t`` and every article
published up to the 15:00 cutoff of ``t`` (:mod:`stf.forecasting.calendar`). The
payoff booked here is ``close_{t+1} / close_t - 1``. Entering at ``close_t`` means
transacting at the very price the signal consumed, which no participant can do: the
closing price is only known once the close has happened. The numbers therefore
describe an upper bound under perfect, costless, simultaneous execution at the
close, not an achievable return. A tradable variant would have to enter at the next
session's open or later and would forfeit the overnight move, which is where a large
part of a daily-frequency edge usually sits.

Other accounting choices
------------------------
* The payoff is recomputed from the price file rather than inverted from the labels,
  so a mislabelled row shows up as a loss instead of silently cancelling.
* Positions are equal-weighted inside each leg. When both legs fire the book is
  cash-neutral. When only one side fires it is **directional**: the live leg is
  halved to limit the exposure, but the book still carries market risk that day, and
  ``net_exposure`` records exactly how much. Do not describe the series as
  market-neutral; ``mean_abs_net_exposure`` and ``one_sided_fraction`` quantify the
  violation.
* Costs are charged on realized turnover against the *drifted* pre-trade holdings,
  not against yesterday's target weights: an unchanged target still has to be
  rebalanced once prices move, and pretending otherwise understates cost.
* Intervals resample whole sessions. They treat sessions as exchangeable and so do
  not model serial dependence; with 300 sessions they are indicative only. Always
  read them next to the permutation null for the same book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stf.forecasting.calendar import to_local
from stf.forecasting.labels import TREND_LABELS

TRADING_DAYS = 252


def realized_returns(prices: pd.DataFrame, *, tz: str = "Asia/Ho_Chi_Minh") -> pd.DataFrame:
    """Return ``(ticker, target_date, realized_return)`` for every session transition."""
    required = {"ticker", "time", "close"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"prices missing columns {missing}.")
    frames = []
    for ticker, group in prices.groupby("ticker", sort=True):
        g = group.sort_values("time")
        dates = to_local(g["time"], tz).dt.tz_localize(None).dt.normalize()
        close = g["close"].astype(float).to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "ticker": str(ticker),
                    "target_date": dates.to_numpy(),
                    "realized_return": np.concatenate(
                        [[np.nan], close[1:] / close[:-1] - 1.0]
                    ),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["realized_return"]).reset_index(drop=True)


def _positions(labels: pd.Series) -> np.ndarray:
    """Equal weights inside each leg: +1/n_long on UP, -1/n_short on DOWN, 0 on FLAT.

    Both legs present means the weights sum to zero. Only one side present means the
    book is directional for that session; the live leg is halved so a one-sided view
    does not become a double-sized market bet, but the residual exposure is real and
    is reported as ``net_exposure`` rather than assumed away.
    """
    values = labels.to_numpy()
    weights = np.zeros(len(values), dtype=float)
    longs = values == "UP"
    shorts = values == "DOWN"
    n_long, n_short = int(longs.sum()), int(shorts.sum())
    if n_long:
        weights[longs] = 1.0 / n_long
    if n_short:
        weights[shorts] = -1.0 / n_short
    if not (n_long and n_short):
        weights *= 0.5
    return weights


def daily_book(
    predictions: pd.DataFrame, returns: pd.DataFrame, *, cost_bps: float
) -> pd.DataFrame:
    """Build the per-session gross/net return series for one arm, averaged over seeds."""
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative.")
    frame = predictions.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.normalize()
    merged = frame.merge(returns, on=["ticker", "target_date"], how="inner")
    if merged.empty:
        raise ValueError("No prediction row could be matched to a realized return.")

    per_seed = []
    for seed, seed_block in merged.groupby("seed", sort=True):
        rows = []
        # Pre-trade holdings, expressed in the SAME units as the targets: fractions of
        # today's NAV. Target weights are fractions of current capital and ``_stats``
        # compounds, so yesterday's drifted position must be divided by the NAV growth
        # it produced. Leaving it in yesterday's units mixes two capital bases and
        # understates turnover.
        held: dict[str, float] = {}
        for date, day in seed_block.groupby("target_date", sort=True):
            day = day.sort_values("ticker")
            tickers = day["ticker"].tolist()
            weights = _positions(day["y_pred"])
            realized = day["realized_return"].to_numpy()
            target = dict(zip(tickers, weights, strict=True))
            names = set(target) | set(held)
            turnover = sum(abs(target.get(t, 0.0) - held.get(t, 0.0)) for t in names)
            cost = turnover * cost_bps / 10_000.0
            gross = float(np.dot(weights, realized))
            net = gross - cost
            nav_growth = 1.0 + net
            if nav_growth <= 0.0:
                raise ValueError(
                    f"Book was wiped out on {date}: NAV growth {nav_growth:.4f} <= 0."
                )
            held = {
                t: w * (1.0 + r) / nav_growth
                for t, w, r in zip(tickers, weights, realized, strict=True)
            }
            rows.append(
                {
                    "seed": seed,
                    "target_date": date,
                    "gross_return": gross,
                    "turnover": turnover,
                    "cost": cost,
                    "gross_exposure": float(np.abs(weights).sum()),
                    "net_exposure": float(weights.sum()),
                    "n_positions": int((weights != 0).sum()),
                    "one_sided": float(
                        not ((weights > 0).any() and (weights < 0).any())
                    ),
                }
            )
        per_seed.append(pd.DataFrame(rows))

    book = pd.concat(per_seed, ignore_index=True)
    book["net_return"] = book["gross_return"] - book["cost"]
    columns = [
        "gross_return",
        "cost",
        "net_return",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "n_positions",
        "one_sided",
    ]
    return (
        book.groupby("target_date", as_index=False)[columns]
        .mean()
        .sort_values("target_date")
        .reset_index(drop=True)
    )


def _drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline, measured from the initial capital.

    The equity path starts at 1 before any return is booked; omitting that point
    would report a first-day loss as a zero drawdown.
    """
    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _stats(returns: np.ndarray) -> dict[str, float]:
    if returns.size == 0:
        raise ValueError("Cannot summarize an empty return series.")
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    return {
        "sessions": int(returns.size),
        "mean_daily_return": mean,
        "daily_volatility": std,
        "annualized_return": float((1.0 + mean) ** TRADING_DAYS - 1.0),
        "annualized_sharpe": sharpe,
        "cumulative_return": float(np.prod(1.0 + returns) - 1.0),
        "hit_rate": float((returns > 0).mean()),
        "max_drawdown": _drawdown(returns),
    }


def _bootstrap(returns: np.ndarray, *, samples: int, seed: int) -> dict[str, dict[str, float]]:
    """Percentile interval over resampled sessions.

    Sessions are drawn independently, so the interval assumes they are exchangeable
    and does not model serial dependence in the book's returns.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, returns.size, size=(samples, returns.size))
    draws = returns[idx]
    means = draws.mean(axis=1)
    stds = draws.std(axis=1, ddof=1)
    sharpes = np.divide(
        means * np.sqrt(TRADING_DAYS), stds, out=np.zeros_like(means), where=stds > 0
    )
    return {
        "mean_daily_return": {
            "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)),
            "fraction_of_draws_le_zero": float((means <= 0).mean()),
        },
        "annualized_sharpe": {
            "low": float(np.quantile(sharpes, 0.025)),
            "high": float(np.quantile(sharpes, 0.975)),
        },
    }


def benchmark_series(returns: pd.DataFrame, dates: set) -> dict[str, np.ndarray]:
    """Two long-only references over the same sessions.

    ``equal_weight_rebalanced`` resets to equal weights every session, which is the
    cross-sectional mean return. ``buy_and_hold`` buys equal weights once and lets
    them drift, so winners grow into the portfolio. They are different products and
    the daily-rebalanced one is not a buy-and-hold benchmark.
    """
    block = returns[returns["target_date"].isin(dates)]
    wide = block.pivot_table(
        index="target_date", columns="ticker", values="realized_return"
    ).sort_index()
    rebalanced = wide.mean(axis=1).to_numpy()

    held = np.full(wide.shape[1], 1.0 / wide.shape[1])
    drifting = np.empty(len(wide))
    for i, row in enumerate(wide.to_numpy()):
        live = np.nan_to_num(row)
        drifting[i] = float(np.dot(held, live))
        held = held * (1.0 + live)
        held = held / held.sum()
    return {"equal_weight_rebalanced": rebalanced, "buy_and_hold": drifting}


def market_regression(book_returns: np.ndarray, market: np.ndarray) -> dict[str, float]:
    """OLS of the book on the basket, so a net-long tilt is not mistaken for skill.

    The book is not market-neutral on one-sided sessions, and the test period was a
    rising market. ``alpha`` is the intercept, in daily return units.
    """
    design = np.column_stack([np.ones(len(book_returns)), market])
    coef, *_ = np.linalg.lstsq(design, book_returns, rcond=None)
    residual = book_returns - design @ coef
    dof = len(book_returns) - design.shape[1]
    variance = float(residual @ residual / dof) if dof > 0 else float("nan")
    se = float(np.sqrt(variance * np.linalg.inv(design.T @ design)[0, 0]))
    residual_sd = float(residual.std(ddof=1))
    return {
        "alpha_daily": float(coef[0]),
        "alpha_t_statistic": float(coef[0] / se) if se > 0 else 0.0,
        "beta": float(coef[1]),
        "residual_annualized_sharpe": (
            float(coef[0] / residual_sd * np.sqrt(TRADING_DAYS)) if residual_sd > 0 else 0.0
        ),
    }


def permutation_null(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Sharpe of the same book when the labels are shuffled inside each session.

    This holds the dates, the cross-section and the whole accounting fixed and varies
    only the signal, so it answers two questions at once: is the accounting unbiased
    (the null should centre on zero), and how large a Sharpe does 300 noisy sessions
    produce by chance. Read any observed Sharpe against this spread, never against
    zero.
    """
    if draws < 1:
        raise ValueError("draws must be >= 1")
    rng = np.random.default_rng(seed)
    base = predictions[predictions["seed"] == sorted(predictions["seed"].unique())[0]]
    values = np.empty(draws)
    for i in range(draws):
        shuffled = base.copy()
        shuffled["y_pred"] = shuffled.groupby("target_date")["y_pred"].transform(
            lambda s: s.to_numpy()[rng.permutation(len(s))]
        )
        series = daily_book(shuffled, returns, cost_bps=0.0)["net_return"].to_numpy()
        sd = series.std(ddof=1)
        values[i] = series.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    return {
        "draws": int(draws),
        "mean_sharpe": float(values.mean()),
        "sd_sharpe": float(values.std(ddof=1)),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def backtest_arm(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    arm: str,
    cost_bps: float = 20.0,
    samples: int = 2000,
    seed: int = 7,
    permutations: int = 200,
) -> dict:
    """Score one arm's stored predictions as an idealized close-to-close book.

    ``cost_bps`` is charged per unit of turnover; 20 bps is the order of magnitude of
    HOSE brokerage plus the 0.1% sell tax, so it is a floor. It does not cover the
    execution-timing problem described in the module docstring, which no fee level
    can fix.
    """
    if arm not in set(predictions["arm"]):
        raise ValueError(f"predictions contain no arm {arm!r}.")
    unknown = set(predictions["y_pred"]) - set(TREND_LABELS)
    if unknown:
        raise ValueError(f"predictions carry unknown classes {sorted(unknown)}.")

    returns = realized_returns(prices)
    book = daily_book(predictions[predictions["arm"] == arm], returns, cost_bps=cost_bps)
    net = book["net_return"].to_numpy()
    gross = book["gross_return"].to_numpy()
    benchmarks = benchmark_series(returns, set(book["target_date"]))
    market = benchmarks["equal_weight_rebalanced"]

    report = {
        "arm": arm,
        "cost_bps": cost_bps,
        "execution": "idealized close-to-close; entry price is the signal's own close",
        "market_neutral": False,
        "net": _stats(net),
        "gross": _stats(gross),
        "equal_weight_rebalanced": _stats(market),
        "buy_and_hold": _stats(benchmarks["buy_and_hold"]),
        "mean_turnover": float(book["turnover"].mean()),
        "mean_gross_exposure": float(book["gross_exposure"].mean()),
        "mean_abs_net_exposure": float(book["net_exposure"].abs().mean()),
        "one_sided_fraction": float(book["one_sided"].mean()),
        "mean_positions": float(book["n_positions"].mean()),
        "breakeven_cost_bps": (
            float(10_000.0 * gross.mean() / book["turnover"].mean())
            if book["turnover"].mean() > 0
            else None
        ),
        "gross_vs_market": market_regression(gross, market),
        "net_bootstrap": _bootstrap(net, samples=samples, seed=seed),
        "gross_bootstrap": _bootstrap(gross, samples=samples, seed=seed),
    }
    if permutations:
        report["permutation_null_gross_sharpe"] = permutation_null(
            predictions[predictions["arm"] == arm], returns, draws=permutations, seed=seed
        )
    return report

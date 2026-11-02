import numpy as np
import pandas as pd

from deskcast.forecast import (
    apply_interval,
    daily_counts,
    forecast_with_intervals,
    interval_quantiles,
    log_residuals,
    mae,
    mape,
    rolling_origin_backtest,
    seasonal_naive,
)


def test_error_metrics():
    assert mae([10, 20, 30], [12, 18, 30]) == np.mean([2, 2, 0])
    assert np.isclose(mape([100, 50], [110, 40]), 15.0)
    assert np.isclose(mape([0, 100], [5, 90]), 10.0)  # zero actuals are skipped


def test_seasonal_naive_repeats_last_week():
    idx = pd.date_range("2025-01-01", periods=21, freq="D")
    s = pd.Series(np.arange(21, dtype=float), index=idx)
    fc = seasonal_naive(s, horizon=10)
    assert list(fc.to_numpy()) == [14, 15, 16, 17, 18, 19, 20, 14, 15, 16]
    assert fc.index[0] == idx[-1] + pd.Timedelta(days=1)


def test_ridge_beats_seasonal_naive_in_backtest(small_tickets):
    s = daily_counts(small_tickets)
    points, folds = rolling_origin_backtest(s, n_folds=3, horizon=28)
    assert len(points) == 84 and list(folds["fold"]) == [1, 2, 3]
    # test windows tile the end of the series without overlap
    assert points.index.is_unique and points.index[-1] == s.index[-1]
    assert mape(points.actual, points.forecast) < mape(points.actual, points.baseline)


def test_intervals_bracket_the_forecast(small_tickets):
    s = daily_counts(small_tickets)
    points, _ = rolling_origin_backtest(s, n_folds=3, horizon=28)
    q_lo, q_hi = interval_quantiles(log_residuals(points), 0.8)
    assert q_lo < 0 < q_hi
    fc = forecast_with_intervals(s, q_lo, q_hi, horizon=28)
    assert len(fc) == 28
    assert (fc["lo"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["hi"]).all()
    lo, hi = apply_interval(np.array([100.0]), -0.1, 0.1)
    assert np.isclose(np.log1p(hi[0]) - np.log1p(lo[0]), 0.2)

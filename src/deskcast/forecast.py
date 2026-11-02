"""Daily ticket-volume forecasting: seasonal-naive baseline vs calendar-feature ridge.

The ridge model is a *direct* multi-horizon model: every lag feature is at least
28 days old, so one fitted model forecasts all 28 days ahead without recursion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calendar_utils import holidays_between, next_business_day
from .config import BACKTEST_FOLDS, FORECAST_HORIZON, INTERVAL_LEVEL

MIN_LAG = 28
WEEKDAY_LAGS = (28, 35, 42, 49, 56, 63, 70, 77)  # same weekday, 4 to 11 weeks back
LEVEL_WINDOW = 56
DEFAULT_ALPHA = 10.0


def daily_counts(tickets: pd.DataFrame, start=None, end=None) -> pd.Series:
    """Tickets created per calendar day, zero-filled."""
    day = tickets["created_at"].dt.normalize()
    counts = day.value_counts().sort_index()
    start = pd.Timestamp(start) if start is not None else counts.index.min()
    end = pd.Timestamp(end) if end is not None else counts.index.max()
    idx = pd.date_range(start, end, freq="D")
    return counts.reindex(idx, fill_value=0).astype(float).rename("tickets")


def mae(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs(a - f)))


def mape(actual, forecast) -> float:
    """Mean absolute percentage error in percent (days with zero actuals are skipped)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    mask = a > 0
    return float(np.mean(np.abs(a[mask] - f[mask]) / a[mask]) * 100.0)


def seasonal_naive(history: pd.Series, horizon: int = FORECAST_HORIZON, season: int = 7) -> pd.Series:
    """Repeat the last observed season: y_hat(T+h) = y(T + h - season * ceil(h / season))."""
    last = history.iloc[-season:].to_numpy()
    values = np.array([last[(h - 1) % season] for h in range(1, horizon + 1)])
    idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(values, index=idx, name="seasonal_naive")


def calendar_features(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    """Day-of-week, month, holiday, bridge-day, year-end and trend features."""
    hol = holidays_between(dates.min() - pd.Timedelta(days=10), dates.max() + pd.Timedelta(days=10))
    after = {next_business_day(h, hol) for h in hol}
    d = [x.date() for x in dates]
    feats = pd.DataFrame(index=dates)
    for k in range(6):
        feats[f"dow_{k}"] = (dates.dayofweek == k).astype(float)
    for m in range(2, 13):
        feats[f"month_{m}"] = (dates.month == m).astype(float)
    feats["is_holiday"] = np.array([x in hol for x in d], dtype=float)
    feats["after_holiday"] = np.array([x in after for x in d], dtype=float)
    feats["year_end"] = ((dates.month == 12) & (dates.day >= 24) & (dates.dayofweek < 5)).astype(float)
    feats["year_end"] = feats["year_end"] * (1 - feats["is_holiday"])
    feats["trend"] = (dates - origin).days / 365.25
    return feats


def special_day_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True for holidays, the business day after a holiday, and year-end weekdays."""
    hol = holidays_between(dates.min() - pd.Timedelta(days=10), dates.max() + pd.Timedelta(days=10))
    after = {next_business_day(h, hol) for h in hol}
    return np.array(
        [d.date() in hol or d.date() in after or (d.month == 12 and d.day >= 24) for d in dates], dtype=bool
    )


def lag_features(series: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Holiday-robust lags that are at least 28 days old, so every horizon up to 28 is known.

    Special days are masked before taking the same-weekday median and the rolling
    level, so a holiday 5 weeks ago does not drag down today's forecast.
    """
    full_idx = pd.date_range(series.index.min(), dates.max(), freq="D")
    ext = series.reindex(full_idx)
    clean = ext.where(~special_day_mask(full_idx))
    lags = pd.concat({f"lag_{k}": clean.shift(k) for k in WEEKDAY_LAGS}, axis=1)
    out = pd.DataFrame(index=full_idx)
    out["log_weekday_median"] = np.log1p(lags.median(axis=1, skipna=True))
    level = clean.shift(MIN_LAG).rolling(LEVEL_WINDOW, min_periods=LEVEL_WINDOW // 2).mean()
    out["log_level"] = np.log1p(level)
    return out.reindex(dates)


class CalendarRidgeForecaster:
    """Ridge regression on log1p(volume) with calendar and long-lag features."""

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        self.model = None
        self.history: pd.Series | None = None
        self.origin: pd.Timestamp | None = None
        self.columns: list[str] = []

    def _design(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.concat([calendar_features(dates, self.origin), lag_features(self.history, dates)], axis=1)

    def fit(self, history: pd.Series) -> "CalendarRidgeForecaster":
        self.history = history.astype(float)
        self.origin = history.index[0]
        X = self._design(history.index)
        y = np.log1p(history)
        ok = X.notna().all(axis=1)
        self.columns = list(X.columns)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model.fit(X[ok].to_numpy(), y[ok].to_numpy())
        return self

    def predict(self, horizon: int = FORECAST_HORIZON) -> pd.Series:
        if horizon > MIN_LAG:
            raise ValueError(f"horizon must be <= {MIN_LAG} for the direct lag design")
        idx = pd.date_range(self.history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        X = self._design(idx)[self.columns]
        pred = np.expm1(self.model.predict(X.to_numpy()))
        return pd.Series(np.clip(pred, 0, None), index=idx, name="forecast")


def rolling_origin_backtest(
    series: pd.Series, n_folds: int = BACKTEST_FOLDS, horizon: int = FORECAST_HORIZON, alpha: float = DEFAULT_ALPHA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-fit at each origin and forecast the next `horizon` days.

    Origins are spaced `horizon` days apart and end at the last observation, so the
    test windows tile the final n_folds * horizon days without overlap.
    Returns (points, folds): one row per forecast day and one row per fold.
    """
    points, folds = [], []
    last = len(series)
    for k in range(n_folds, 0, -1):
        cut = last - k * horizon
        hist, test = series.iloc[:cut], series.iloc[cut : cut + horizon]
        model_fc = CalendarRidgeForecaster(alpha).fit(hist).predict(horizon)
        naive_fc = seasonal_naive(hist, horizon)
        fold_id = n_folds - k + 1
        df = pd.DataFrame(
            {
                "fold": fold_id,
                "origin": hist.index[-1],
                "horizon": np.arange(1, horizon + 1),
                "actual": test.to_numpy(),
                "forecast": model_fc.to_numpy(),
                "baseline": naive_fc.to_numpy(),
            },
            index=test.index,
        )
        points.append(df)
        folds.append(
            {
                "fold": fold_id,
                "origin": hist.index[-1].date().isoformat(),
                "test_start": test.index[0].date().isoformat(),
                "test_end": test.index[-1].date().isoformat(),
                "model_mae": mae(df.actual, df.forecast),
                "model_mape": mape(df.actual, df.forecast),
                "baseline_mae": mae(df.actual, df.baseline),
                "baseline_mape": mape(df.actual, df.baseline),
            }
        )
    pts = pd.concat(points)
    pts.index.name = "date"
    return pts, pd.DataFrame(folds)


def log_residuals(points: pd.DataFrame) -> np.ndarray:
    """Multiplicative residuals log((actual + 1) / (forecast + 1))."""
    return np.log1p(points["actual"].to_numpy()) - np.log1p(points["forecast"].to_numpy())


def interval_quantiles(residuals: np.ndarray, level: float = INTERVAL_LEVEL) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(residuals, alpha)), float(np.quantile(residuals, 1.0 - alpha))


def apply_interval(forecast: np.ndarray, q_lo: float, q_hi: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(forecast, float)
    lo = np.expm1(np.log1p(f) + q_lo)
    hi = np.expm1(np.log1p(f) + q_hi)
    return np.clip(lo, 0, None), hi


def interval_holdout_coverage(points: pd.DataFrame, level: float = INTERVAL_LEVEL) -> float:
    """Calibrate on the first half of the folds, measure coverage on the second half."""
    folds = sorted(points["fold"].unique())
    split = folds[: len(folds) // 2]
    calib = points[points["fold"].isin(split)]
    test = points[~points["fold"].isin(split)]
    q_lo, q_hi = interval_quantiles(log_residuals(calib), level)
    lo, hi = apply_interval(test["forecast"].to_numpy(), q_lo, q_hi)
    a = test["actual"].to_numpy()
    return float(np.mean((a >= lo) & (a <= hi)))


def forecast_with_intervals(
    series: pd.Series, q_lo: float, q_hi: float, horizon: int = FORECAST_HORIZON, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    fc = CalendarRidgeForecaster(alpha).fit(series).predict(horizon)
    lo, hi = apply_interval(fc.to_numpy(), q_lo, q_hi)
    base = seasonal_naive(series, horizon)
    out = pd.DataFrame({"forecast": fc, "lo": lo, "hi": hi, "baseline": base.to_numpy()}, index=fc.index)
    out.index.name = "date"
    hol = holidays_between(out.index.min(), out.index.max())
    out["holiday"] = [hol.get(d.date(), "") for d in out.index]
    return out


def hourly_profile(tickets: pd.DataFrame, end, weeks: int = 12) -> np.ndarray:
    """Share of each day's tickets that arrive in each hour, by weekday (7 x 24), from recent history.

    Holidays are excluded so the profile reflects normal operating days.
    """
    end = pd.Timestamp(end) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(weeks=weeks)
    recent = tickets[(tickets["created_at"] >= start) & (tickets["created_at"] < end)]
    hol = pd.to_datetime(list(holidays_between(start, end)))
    recent = recent[~recent["created_at"].dt.normalize().isin(hol)]
    counts = np.zeros((7, 24))
    np.add.at(counts, (recent["created_at"].dt.dayofweek.to_numpy(), recent["created_at"].dt.hour.to_numpy()), 1)
    return counts / counts.sum(axis=1, keepdims=True)

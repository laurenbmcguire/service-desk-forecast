"""Matplotlib charts written to docs/img (brand style from style.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import style
from .config import DOW_NAMES, FEATURE_LABELS


def forecast_chart(series: pd.Series, backtest: pd.DataFrame, future: pd.DataFrame, q: tuple[float, float],
                   metrics: dict, path: Path) -> Path:
    """Hero chart: actual daily volume, backtest forecasts, and the 28-day forecast with its interval."""
    from .forecast import apply_interval

    fig, ax = style.new_figure()
    hist = series[series.index >= backtest.index.min() - pd.Timedelta(days=28)]
    bt_lo, bt_hi = apply_interval(backtest["forecast"].to_numpy(), *q)
    ax.fill_between(backtest.index, bt_lo, bt_hi, color=style.SAND, alpha=0.55, linewidth=0, step=None,
                    label="80% interval")
    ax.fill_between(future.index, future["lo"], future["hi"], color=style.SAND, alpha=0.95, linewidth=0)
    ax.plot(hist.index, hist.to_numpy(), color=style.INK, linewidth=1.3, label="Actual")
    ax.plot(backtest.index, backtest["forecast"], color=style.CRIMSON, linewidth=1.4, linestyle="--",
            label="Backtest forecast (rolling origin)")
    ax.plot(future.index, future["forecast"], color=style.DEEP_CRIMSON, linewidth=2.0, label="Next 28 days")
    ax.axvline(series.index[-1] + pd.Timedelta(hours=12), color=style.MUTED, linewidth=0.9, linestyle=":")
    ax.text(series.index[-1] + pd.Timedelta(days=1), ax.get_ylim()[1] * 0.97, "forecast origin",
            color=style.MUTED, fontsize=9, va="top")
    ax.set_title("Daily ticket volume: actual vs forecast with 80% interval")
    ax.set_ylabel("Tickets per day")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    note = (f"Backtest ({metrics['folds']} x 28-day folds): ridge MAPE {metrics['model_mape']:.1f}% vs "
            f"seasonal-naive {metrics['baseline_mape']:.1f}%  |  MAE {metrics['model_mae']:.1f} vs "
            f"{metrics['baseline_mae']:.1f} tickets/day")
    ax.text(0.0, -0.13, note, transform=ax.transAxes, color=style.MUTED, fontsize=9.5)
    ax.legend(loc="upper left", ncol=4, fontsize=9.5)
    ax.set_ylim(0, max(hist.max(), future["hi"].max()) * 1.15)
    fig.tight_layout()
    return style.save(fig, path)


def _heatmap(ax, matrix: np.ndarray, annotate: bool, fmt: str = "{:.0f}", vmax=None):
    cmap = style.brand_cmap()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_yticks(range(7), DOW_NAMES)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.grid(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    if annotate:
        top = np.nanmax(matrix)
        for (r, c), v in np.ndenumerate(matrix):
            color = style.BG if v > 0.6 * top else style.INK
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8.5, color=color)
    return im


def arrival_heatmap(matrix: np.ndarray, path: Path) -> Path:
    fig, ax = style.new_figure()
    im = _heatmap(ax, matrix, annotate=True, fmt="{:.0f}")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Mean tickets per hour")
    cb.outline.set_visible(False)
    ax.set_title("When tickets arrive: mean hourly arrivals by weekday (non-holiday days)")
    fig.tight_layout()
    return style.save(fig, path)



def importance_chart(importances: pd.DataFrame, auc: float, path: Path, top: int = 10) -> Path:
    fig, ax = style.new_figure()
    data = importances.head(top).iloc[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in data["feature"]]
    colors = [style.CRIMSON if i >= len(data) - 3 else style.GOLD for i in range(len(data))]
    ax.barh(labels, data["importance"], xerr=data["std"], color=colors,
            error_kw={"ecolor": style.MUTED, "elinewidth": 0.8, "capsize": 2})
    ax.set_title(f"What drives SLA breach risk (gradient boosting, test ROC AUC {auc:.3f})")
    ax.set_xlabel("Permutation importance: drop in ROC AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    pad = 0.01 * float(data["importance"].max())
    for y, (v, sd) in enumerate(zip(data["importance"], data["std"])):
        shown = 0.0 if abs(v) < 5e-4 else v
        ax.text(max(v, 0) + sd + pad, y, f"{shown:.3f}", va="center", fontsize=9, color=style.INK)
    fig.tight_layout()
    return style.save(fig, path)


def staffing_heatmap(agents: np.ndarray, aht: float, path: Path) -> Path:
    fig, ax = style.new_figure()
    im = _heatmap(ax, agents.astype(float), annotate=True, fmt="{:.0f}")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Agents on queue")
    cb.outline.set_visible(False)
    ax.set_title(f"Agents needed per hour for 80% answered in 20 s (Erlang C, AHT {aht:.1f} min)")
    fig.tight_layout()
    return style.save(fig, path)

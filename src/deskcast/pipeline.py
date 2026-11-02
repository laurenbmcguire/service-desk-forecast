"""End-to-end pipeline: data -> forecast backtest -> SLA model -> staffing -> charts and reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts
from .calendar_utils import holidays_between
from .config import (
    ANSWER_TIME_SECONDS,
    BACKTEST_FOLDS,
    DOW_NAMES,
    FORECAST_HORIZON,
    INTERVAL_LEVEL,
    MAX_OCCUPANCY,
    SERVICE_LEVEL_TARGET,
    SHRINKAGE,
    SLA_TARGET_HOURS,
)
from .forecast import (
    daily_counts,
    forecast_with_intervals,
    hourly_profile,
    interval_holdout_coverage,
    interval_quantiles,
    log_residuals,
    mae,
    mape,
    rolling_origin_backtest,
    special_day_mask,
)
from .generator import GeneratorConfig, generate_tickets, load_tickets, save_tickets, validate_tickets
from .report import render_report
from .sla import build_features, train_sla_model
from .staffing import scheduled_fte, staffing_matrix

TICKETS_FILE = "data/tickets.csv"
SAMPLE_FILE = "data/sample_tickets.csv"
DAILY_FILE = "data/daily_volume.csv"
SAMPLE_ROWS = 1000
SLA_TEST_MONTHS = 4
PROFILE_WEEKS = 12


def _r(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def ensure_data(root: Path, regenerate: bool = False, cfg: GeneratorConfig | None = None) -> Path:
    path = root / TICKETS_FILE
    if regenerate or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_tickets(generate_tickets(cfg), path)
    return path


def arrival_matrix(tickets: pd.DataFrame, start, end) -> np.ndarray:
    """Mean tickets per (weekday, hour) over non-holiday days."""
    hol = pd.to_datetime(list(holidays_between(start, end)))
    t = tickets[~tickets["created_at"].dt.normalize().isin(hol)]
    counts = np.zeros((7, 24))
    np.add.at(counts, (t["created_at"].dt.dayofweek.to_numpy(), t["created_at"].dt.hour.to_numpy()), 1)
    days = pd.date_range(start, end, freq="D")
    days = days[~days.isin(hol)]
    n_days = np.bincount(days.dayofweek, minlength=7).astype(float)
    return counts / n_days[:, None]


def typical_week_arrivals(future: pd.DataFrame, profile: np.ndarray) -> np.ndarray:
    """Forecast hourly arrivals for a typical week: mean non-holiday daily forecast by weekday x hourly share."""
    normal = future[future["holiday"] == ""]
    by_dow = normal.groupby(normal.index.dayofweek)["forecast"].mean().reindex(range(7))
    return by_dow.to_numpy()[:, None] * profile


def run(root: str | Path = ".", regenerate: bool = False, cfg: GeneratorConfig | None = None, verbose: bool = True) -> dict:
    root = Path(root)
    say = print if verbose else (lambda *a, **k: None)
    data_path = ensure_data(root, regenerate, cfg)
    tickets = load_tickets(data_path)
    say(f"[deskcast] loaded {len(tickets):,} tickets from {data_path.relative_to(root)}")

    start = tickets["created_at"].min().normalize()
    end = tickets["created_at"].max().normalize()
    extract_end = end + pd.Timedelta(days=1)
    quality = validate_tickets(tickets)

    # ---- Forecasting -------------------------------------------------------
    series = daily_counts(tickets, start, end)
    points, folds = rolling_origin_backtest(series, BACKTEST_FOLDS, FORECAST_HORIZON)
    q_lo, q_hi = interval_quantiles(log_residuals(points), INTERVAL_LEVEL)
    coverage = interval_holdout_coverage(points, INTERVAL_LEVEL)
    future = forecast_with_intervals(series, q_lo, q_hi, FORECAST_HORIZON)
    bt = {
        "folds": BACKTEST_FOLDS,
        "horizon_days": FORECAST_HORIZON,
        "model_mae": mae(points.actual, points.forecast),
        "model_mape": mape(points.actual, points.forecast),
        "baseline_mae": mae(points.actual, points.baseline),
        "baseline_mape": mape(points.actual, points.baseline),
    }
    bt["mae_improvement_pct"] = 100 * (bt["baseline_mae"] - bt["model_mae"]) / bt["baseline_mae"]
    bt["mape_improvement_pct"] = 100 * (bt["baseline_mape"] - bt["model_mape"]) / bt["baseline_mape"]
    special = special_day_mask(points.index)
    split = {}
    for name, sel in (("special_days", special), ("normal_days", ~special)):
        pts = points[sel]
        split[name] = {
            "days": int(sel.sum()),
            "model_mape": _r(mape(pts.actual, pts.forecast), 2),
            "baseline_mape": _r(mape(pts.actual, pts.baseline), 2),
            "model_mae": _r(mae(pts.actual, pts.forecast), 2),
            "baseline_mae": _r(mae(pts.actual, pts.baseline), 2),
        }
    say(f"[deskcast] backtest MAPE ridge {bt['model_mape']:.1f}% vs seasonal-naive {bt['baseline_mape']:.1f}%")

    # ---- SLA breach model ----------------------------------------------------
    feats = build_features(tickets, extract_end)
    test_start = (end + pd.Timedelta(days=1)) - pd.DateOffset(months=SLA_TEST_MONTHS)
    sla = train_sla_model(feats, test_start)
    say(f"[deskcast] SLA model ROC AUC {sla['roc_auc']:.3f}, precision@top10% {sla['precision_at_top10']:.1%}")

    # ---- Staffing -------------------------------------------------------------
    recent = tickets[tickets["created_at"] >= extract_end - pd.Timedelta(weeks=PROFILE_WEEKS)]
    aht = float(recent["handle_minutes"].mean())
    profile = hourly_profile(tickets, end, PROFILE_WEEKS)
    week_arrivals = typical_week_arrivals(future, profile)
    agents = staffing_matrix(week_arrivals, aht)
    agent_hours = int(agents.sum())
    peak = np.unravel_index(int(np.argmax(agents)), agents.shape)
    say(f"[deskcast] staffing peak {agents.max()} agents ({DOW_NAMES[peak[0]]} {peak[1]:02d}:00), {agent_hours} agent-hours/week")

    # ---- Charts -----------------------------------------------------------------
    img = root / "docs" / "img"
    arrivals = arrival_matrix(tickets, start, end)
    charts.forecast_chart(series, points, future, (q_lo, q_hi), bt, img / "forecast_hero.png")
    charts.arrival_heatmap(arrivals, img / "arrival_heatmap.png")
    charts.importance_chart(sla["importances"], sla["roc_auc"], img / "sla_feature_importance.png")
    charts.staffing_heatmap(agents, aht, img / "staffing_heatmap.png")

    # ---- Summary ----------------------------------------------------------------
    daily_by_dow = series.groupby(series.index.dayofweek).mean()
    labelled = feats[feats["breached"].notna()]
    incidents = []
    for inc in (cfg or GeneratorConfig()).incidents:
        day = pd.Timestamp(inc.start).normalize()
        if start <= day <= end:
            dow_mean = float(daily_by_dow.loc[day.dayofweek])
            incidents.append({
                "name": inc.name,
                "date": day.date().isoformat(),
                "category": inc.category,
                "tickets_that_day": int(series.loc[day]),
                "vs_weekday_mean": _r(series.loc[day] / dow_mean, 2),
            })
    imp = sla["importances"]
    summary = {
        "project": "service-desk-forecast",
        "data": {
            "synthetic": True,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "days": int(len(series)),
            "tickets": int(len(tickets)),
            "mean_daily_weekday": _r(series[series.index.dayofweek < 5].mean(), 1),
            "mean_daily_weekend": _r(series[series.index.dayofweek >= 5].mean(), 1),
            "category_mix": {k: _r(v, 3) for k, v in tickets["category"].value_counts(normalize=True).sort_index().items()},
            "priority_mix": {k: _r(v, 3) for k, v in tickets["priority"].value_counts(normalize=True).sort_index().items()},
            "sla_breach_rate": _r(labelled["breached"].mean()),
            "breach_rate_by_created_weekday": {
                DOW_NAMES[k]: _r(v, 3) for k, v in labelled.groupby(labelled["created_at"].dt.dayofweek)["breached"].mean().items()
            },
            "breach_rate_by_group": {
                k: _r(v, 3) for k, v in labelled.groupby("assignee_group")["breached"].mean().sort_values().items()
            },
            "reopen_rate": _r(tickets["reopened"].mean()),
            "median_resolution_hours": _r(feats["resolution_hours"].median(), 2),
            "quality_checks": quality,
            "incidents": incidents,
        },
        "forecast": {
            "model": "Ridge on log1p(volume): weekday, month, holiday, bridge-day, year-end, trend, 4-11 week same-weekday lags",
            "baseline": "Seasonal naive (same weekday last week)",
            "backtest": {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in bt.items()},
            "backtest_by_day_type": split,
            "interval": {
                "level": INTERVAL_LEVEL,
                "method": "Quantiles of backtest log residuals",
                "log_residual_q_lo": _r(q_lo),
                "log_residual_q_hi": _r(q_hi),
                "holdout_coverage": _r(coverage, 3),
            },
            "next_28_days": {
                "start": future.index[0].date().isoformat(),
                "end": future.index[-1].date().isoformat(),
                "total": _r(future["forecast"].sum(), 0),
                "total_lo": _r(future["lo"].sum(), 0),
                "total_hi": _r(future["hi"].sum(), 0),
                "peak_date": future["forecast"].idxmax().date().isoformat(),
                "peak_forecast": _r(future["forecast"].max(), 1),
                "holidays": {d.date().isoformat(): h for d, h in future["holiday"].items() if h},
            },
        },
        "sla": {
            "targets_hours": SLA_TARGET_HOURS,
            "model": "HistGradientBoostingClassifier on creation-time features",
            "train_start": sla["train_start"],
            "test_start": sla["test_start"],
            "test_end": sla["test_end"],
            "n_train": sla["n_train"],
            "n_test": sla["n_test"],
            "roc_auc": _r(sla["roc_auc"]),
            "roc_auc_without_queue_features": _r(sla["roc_auc_without_queue"]),
            "roc_auc_priority_only": _r(sla["roc_auc_priority_rule"]),
            "test_breach_rate": _r(sla["base_rate"]),
            "precision_at_top10pct": _r(sla["precision_at_top10"]),
            "lift_at_top10pct": _r(sla["lift_at_top10"], 2),
            "top_features": [
                {"feature": r.feature, "importance": _r(r.importance)} for r in imp.head(5).itertuples()
            ],
        },
        "staffing": {
            "method": "Erlang C with occupancy cap",
            "service_level_target": SERVICE_LEVEL_TARGET,
            "answer_time_seconds": ANSWER_TIME_SECONDS,
            "max_occupancy": MAX_OCCUPANCY,
            "aht_minutes": _r(aht, 2),
            "peak_agents": int(agents.max()),
            "peak_slot": f"{DOW_NAMES[peak[0]]} {peak[1]:02d}:00",
            "peak_arrivals_per_hour": _r(week_arrivals[peak], 2),
            "weekly_agent_hours": agent_hours,
            "shrinkage": SHRINKAGE,
            "scheduled_fte": _r(scheduled_fte(agent_hours), 1),
            "weekday_business_hours_mean_agents": _r(agents[:5, 8:18].mean(), 2),
        },
    }

    # ---- Files --------------------------------------------------------------------
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fc_out = future.copy()
    fc_out.index = fc_out.index.date
    fc_out.index.name = "date"
    fc_out.round(1).to_csv(reports / "forecast_28d.csv")
    folds.round(3).to_csv(reports / "backtest_folds.csv", index=False)
    staff = pd.DataFrame(agents, index=list(DOW_NAMES), columns=[f"{h:02d}" for h in range(24)])
    staff.index.name = "day"
    staff.to_csv(reports / "staffing_typical_week.csv")
    report_md = render_report(summary, folds, future, sla["importances"], sla["by_priority"], agents, week_arrivals)
    (reports / "service_desk_report.md").write_text(report_md, encoding="utf-8")

    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(tickets), size=min(SAMPLE_ROWS, len(tickets)), replace=False))
    save_tickets(tickets.iloc[pick], root / SAMPLE_FILE)
    daily = series.rename("tickets").astype(int).to_frame()
    daily.index = daily.index.date
    daily.index.name = "date"
    daily.to_csv(root / DAILY_FILE)
    say("[deskcast] wrote reports/summary.json, reports/service_desk_report.md and 4 charts in docs/img/")
    return summary

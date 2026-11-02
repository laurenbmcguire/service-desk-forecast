"""Markdown report built from the pipeline summary and result tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DOW_NAMES


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_report(summary: dict, folds: pd.DataFrame, future: pd.DataFrame, importances: pd.DataFrame,
                  by_priority: pd.DataFrame, agents: np.ndarray, week_arrivals: np.ndarray) -> str:
    d, f, s, st = summary["data"], summary["forecast"], summary["sla"], summary["staffing"]
    bt, iv, nxt = f["backtest"], f["interval"], f["next_28_days"]
    lines = [
        "# Service Desk Forecast report",
        "",
        f"Synthetic IT service desk data, {d['start']} to {d['end']} ({d['days']} days, {d['tickets']:,} tickets). "
        "All numbers below are computed by `python -m deskcast run` on data from the seeded generator in this repo.",
        "",
        "## Headline",
        "",
        f"- **Forecast:** the calendar-feature ridge model cuts MAPE from {bt['baseline_mape']:.1f}% to "
        f"{bt['model_mape']:.1f}% ({bt['mape_improvement_pct']:.0f}% better) and MAE from {bt['baseline_mae']:.1f} to "
        f"{bt['model_mae']:.1f} tickets/day ({bt['mae_improvement_pct']:.0f}% better) versus seasonal naive, "
        f"over {bt['folds']} rolling 28-day folds.",
        f"- **Next 28 days ({nxt['start']} to {nxt['end']}):** {nxt['total']:,.0f} tickets forecast "
        f"(80% band {nxt['total_lo']:,.0f} to {nxt['total_hi']:,.0f}, summing daily bounds).",
        f"- **SLA risk:** test ROC AUC {s['roc_auc']:.3f}; the riskiest 10% of new tickets breach "
        f"{s['precision_at_top10pct']:.1%} of the time against a {s['test_breach_rate']:.1%} base rate "
        f"({s['lift_at_top10pct']:.1f}x lift).",
        f"- **Staffing:** {st['weekly_agent_hours']} on-queue agent-hours in a typical week, peaking at "
        f"{st['peak_agents']} agents ({st['peak_slot']}); about {st['scheduled_fte']} scheduled FTE at "
        f"{st['shrinkage']:.0%} shrinkage.",
        "",
        "## Data",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Tickets", f"{d['tickets']:,}"],
                ["Mean tickets per weekday / weekend day", f"{d['mean_daily_weekday']} / {d['mean_daily_weekend']}"],
                ["SLA breach rate (labelled tickets)", f"{d['sla_breach_rate']:.1%}"],
                ["Reopen rate", f"{d['reopen_rate']:.1%}"],
                ["Median resolution time", f"{d['median_resolution_hours']} h"],
                ["Open at extract (censored)", d["quality_checks"]["open_at_extract"]],
                ["Resolved before created", d["quality_checks"]["resolved_before_created"]],
                ["Duplicate ticket IDs", d["quality_checks"]["duplicate_ids"]],
            ],
        ),
        "",
        "Injected incident surges and the volume they produced:",
        "",
        _table(
            ["Date", "Incident", "Category", "Tickets that day", "x weekday mean"],
            [[i["date"], i["name"], i["category"], i["tickets_that_day"], f"{i['vs_weekday_mean']:.2f}"]
             for i in d["incidents"]],
        ),
        "",
        "## Forecast backtest",
        "",
        f"Rolling origin, {bt['folds']} folds, 28-day horizon, model re-fit at every origin.",
        "",
        _table(
            ["Fold", "Test window", "Ridge MAE", "Ridge MAPE", "Naive MAE", "Naive MAPE"],
            [[int(r.fold), f"{r.test_start} to {r.test_end}", f"{r.model_mae:.1f}", f"{r.model_mape:.1f}%",
              f"{r.baseline_mae:.1f}", f"{r.baseline_mape:.1f}%"] for r in folds.itertuples()]
            + [["**All**", "", f"**{bt['model_mae']:.1f}**", f"**{bt['model_mape']:.1f}%**",
                f"**{bt['baseline_mae']:.1f}**", f"**{bt['baseline_mape']:.1f}%**"]],
        ),
        "",
        "Split by day type (special days = holidays, the business day after a holiday, and Dec 24-31 weekdays):",
        "",
        _table(
            ["Day type", "Days", "Ridge MAE", "Ridge MAPE", "Naive MAE", "Naive MAPE"],
            [[k.replace("_", " "), v["days"], f"{v['model_mae']:.1f}", f"{v['model_mape']:.1f}%",
              f"{v['baseline_mae']:.1f}", f"{v['baseline_mape']:.1f}%"] for k, v in f["backtest_by_day_type"].items()],
        ),
        "",
        f"Prediction intervals use the {iv['level']:.0%} quantiles of backtest log residuals "
        f"({iv['log_residual_q_lo']:+.3f}, {iv['log_residual_q_hi']:+.3f}). Calibrating on the first half of the folds "
        f"and testing on the second half gives {iv['holdout_coverage']:.1%} coverage.",
        "",
        "## 28-day forecast by week",
        "",
    ]
    weekly = future.copy()
    weekly["week"] = (np.arange(len(weekly)) // 7) + 1
    rows = []
    for wk, g in weekly.groupby("week"):
        hol = ", ".join(h for h in g["holiday"] if h) or "-"
        rows.append([wk, f"{g.index[0].date()} to {g.index[-1].date()}", f"{g['forecast'].sum():,.0f}",
                     f"{g['lo'].sum():,.0f} to {g['hi'].sum():,.0f}", f"{g['baseline'].sum():,.0f}", hol])
    lines += [
        _table(["Week", "Dates", "Forecast", "80% band", "Seasonal naive", "Holidays"], rows),
        "",
        "## SLA breach model",
        "",
        f"Targets (hours, creation to resolution): " + ", ".join(f"{k} {v:g}" for k, v in s["targets_hours"].items()) + ". "
        f"Train {s['train_start']} to before {s['test_start']} ({s['n_train']:,} tickets), test {s['test_start']} to "
        f"{s['test_end']} ({s['n_test']:,} tickets). Features are limited to what is known when the ticket is logged.",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["ROC AUC (gradient boosting)", f"{s['roc_auc']:.3f}"],
                ["ROC AUC without queue features", f"{s['roc_auc_without_queue_features']:.3f}"],
                ["ROC AUC, priority-only rule", f"{s['roc_auc_priority_only']:.3f}"],
                ["Test breach rate", f"{s['test_breach_rate']:.1%}"],
                ["Precision at top 10% risk", f"{s['precision_at_top10pct']:.1%}"],
                ["Lift at top 10%", f"{s['lift_at_top10pct']:.2f}x"],
            ],
        ),
        "",
        "Permutation importance (drop in test ROC AUC when the feature is shuffled):",
        "",
        _table(["Feature", "Importance", "Std"],
               [[r.feature, f"{r.importance:.4f}", f"{r.std:.4f}"] for r in importances.itertuples()]),
        "",
        "Breach rate by weekday of creation (all labelled tickets). P3/P4 work only accrues in business hours while "
        "SLA clocks run in calendar hours, so tickets logged late in the week carry the weekend:",
        "",
        _table(list(d["breach_rate_by_created_weekday"]),
               [[f"{v:.1%}" for v in d["breach_rate_by_created_weekday"].values()]]),
        "",
        "Breach rate by assignee group:",
        "",
        _table(["Group", "Breach rate"], [[k, f"{v:.1%}"] for k, v in d["breach_rate_by_group"].items()]),
        "",
        "Test-period breach rate by priority:",
        "",
        _table(["Priority", "Tickets", "Breach rate", "Mean predicted risk"],
               [[p, int(r.tickets), f"{r.breach_rate:.1%}", f"{r.mean_risk:.1%}"] for p, r in by_priority.iterrows()]),
        "",
        "## Staffing (Erlang C)",
        "",
        f"Typical week built from the non-holiday days of the 28-day forecast and the last 12 weeks' hourly profile. "
        f"AHT {st['aht_minutes']} min, target {st['service_level_target']:.0%} answered within "
        f"{st['answer_time_seconds']} s, occupancy cap {st['max_occupancy']:.0%}, minimum 1 agent per hour.",
        "",
    ]
    rows = []
    for i, day in enumerate(DOW_NAMES):
        rows.append([day, f"{week_arrivals[i].sum():.0f}", int(agents[i].max()), int(agents[i].sum()),
                     f"{int(np.argmax(agents[i])):02d}:00"])
    lines += [
        _table(["Day", "Forecast tickets", "Peak agents", "Agent-hours", "First peak hour"], rows),
        "",
        f"Total {st['weekly_agent_hours']} on-queue agent-hours per week; with {st['shrinkage']:.0%} shrinkage that is "
        f"about {st['scheduled_fte']} scheduled FTE at 40 h/week.",
        "",
        "## Charts",
        "",
        "![Forecast](../docs/img/forecast_hero.png)",
        "![Arrivals](../docs/img/arrival_heatmap.png)",
        "![SLA importance](../docs/img/sla_feature_importance.png)",
        "![Staffing](../docs/img/staffing_heatmap.png)",
        "",
    ]
    return "\n".join(lines)

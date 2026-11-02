"""Seeded synthetic IT service desk ticket generator.

Arrivals are simulated hour by hour as a Poisson process whose rate combines
weekday and hour-of-day seasonality, month seasonality, a gentle growth trend,
company holidays (and the busy day after them), the quiet year-end week and a
handful of injected incident surges. Resolution times depend on category,
priority, who works the ticket, business-hours coverage and the queue load at
the time the ticket arrives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar_utils import add_business_hours, business_hour_flags, holidays_between, next_business_day
from .config import CATEGORIES, CATEGORY_GROUP, CHANNELS, GROUPS, L1_GROUP, PRIORITIES

TICKET_COLUMNS = [
    "ticket_id",
    "created_at",
    "resolved_at",
    "category",
    "priority",
    "channel",
    "assignee_group",
    "handle_minutes",
    "reopened",
]


@dataclass(frozen=True)
class Incident:
    """An injected major incident that adds a decaying burst of tickets."""

    name: str
    start: str
    category: str
    peak_per_hour: float
    decay_hours: float
    duration_hours: int


INCIDENTS: tuple[Incident, ...] = (
    Incident("Email outage: inbound mail flow stopped", "2025-03-11 09:00", "email", 30.0, 9.0, 36),
    Incident("VPN gateway failure", "2025-06-17 08:00", "network", 24.0, 6.0, 24),
    Incident("SSO / MFA provider outage", "2025-10-07 10:00", "access", 38.0, 5.0, 18),
    Incident("Phishing campaign wave", "2026-02-03 08:00", "security", 12.0, 18.0, 60),
    Incident("Office suite update regression", "2026-05-12 09:00", "software", 16.0, 14.0, 48),
)

# Daily volume multipliers
DOW_FACTOR = np.array([1.18, 1.08, 1.02, 0.98, 0.86, 0.21, 0.17])
MONTH_FACTOR = np.array([1.08, 1.02, 1.00, 0.98, 0.97, 0.95, 0.90, 0.94, 1.06, 1.03, 0.98, 0.90])
HOLIDAY_FACTOR = 0.22
AFTER_HOLIDAY_FACTOR = 1.16
YEAR_END_FACTOR = 0.68

# Hour-of-day shape (relative weights, normalised below)
WEEKDAY_HOURS = np.array(
    [0.15, 0.10, 0.08, 0.08, 0.10, 0.20, 0.50, 1.40, 4.20, 7.60, 8.40, 7.80,
     6.20, 6.90, 7.40, 6.80, 5.60, 3.90, 2.20, 1.40, 1.00, 0.70, 0.45, 0.25]
)
WEEKEND_HOURS = np.array(
    [0.40, 0.30, 0.25, 0.25, 0.30, 0.40, 0.70, 1.20, 2.00, 2.80, 3.30, 3.50,
     3.40, 3.30, 3.20, 3.00, 2.70, 2.30, 1.90, 1.60, 1.30, 1.00, 0.80, 0.55]
)

CATEGORY_MIX = np.array([0.28, 0.14, 0.14, 0.12, 0.22, 0.10])

# P1..P4 probabilities per category (rows follow CATEGORIES)
PRIORITY_MIX = np.array(
    [
        [0.01, 0.07, 0.52, 0.40],  # access
        [0.01, 0.06, 0.48, 0.45],  # hardware
        [0.02, 0.10, 0.55, 0.33],  # email
        [0.05, 0.17, 0.50, 0.28],  # network
        [0.01, 0.08, 0.50, 0.41],  # software
        [0.08, 0.22, 0.48, 0.22],  # security
    ]
)
INCIDENT_PRIORITY_MIX = np.array([0.03, 0.30, 0.55, 0.12])

# portal, email, phone, chat, monitoring per category
CHANNEL_MIX = np.array(
    [
        [0.40, 0.12, 0.28, 0.20, 0.00],
        [0.50, 0.15, 0.20, 0.15, 0.00],
        [0.38, 0.15, 0.25, 0.22, 0.00],
        [0.25, 0.10, 0.20, 0.15, 0.30],
        [0.48, 0.17, 0.15, 0.20, 0.00],
        [0.25, 0.30, 0.10, 0.05, 0.30],
    ]
)

L1_SHARE = np.array([0.55, 0.10, 0.30, 0.08, 0.35, 0.05])  # resolved at first line
HANDLE_MEDIAN_MIN = np.array([9.0, 24.0, 13.0, 18.0, 16.0, 21.0])
CHANNEL_HANDLE_MULT = np.array([0.90, 0.95, 1.00, 1.10, 0.70])

# Work-time medians in hours (P1/P2: 24x7 clock, P3/P4: business hours)
PRIORITY_WORK_MEDIAN_H = np.array([1.2, 3.2, 8.0, 15.0])
CATEGORY_WORK_MULT = np.array([0.60, 1.50, 0.90, 1.10, 1.15, 1.25])
L1_WORK_MULT = 0.35
WORK_SIGMA = 0.75
LOAD_WINDOW_H = 72
LOAD_EXPONENT = 0.8
CAPACITY_SHOCK_PHI = 0.85
CAPACITY_SHOCK_SD = 0.14


@dataclass
class GeneratorConfig:
    start: str = "2025-01-01"
    end: str = "2026-06-30"  # inclusive
    seed: int = 42
    base_weekday_daily: float = 128.0
    annual_growth: float = 0.07
    daily_noise_sigma: float = 0.06
    incidents: tuple[Incident, ...] = field(default=INCIDENTS)


def _day_multipliers(days: pd.DatetimeIndex, holidays: dict, cfg: GeneratorConfig, rng) -> np.ndarray:
    dates = [d.date() for d in days]
    hol = set(holidays)
    after = {next_business_day(h, hol) for h in hol}
    mult = DOW_FACTOR[days.dayofweek] * MONTH_FACTOR[days.month - 1]
    years = (days - days[0]).days / 365.25
    mult = mult * (1.0 + cfg.annual_growth * np.asarray(years))
    is_hol = np.array([d in hol for d in dates])
    is_after = np.array([d in after for d in dates])
    is_year_end = np.asarray((days.month == 12) & (days.day >= 24) & (days.dayofweek < 5)) & ~is_hol
    mult = np.where(is_hol, mult * HOLIDAY_FACTOR / np.maximum(DOW_FACTOR[days.dayofweek], 0.5), mult)
    mult = np.where(is_after, mult * AFTER_HOLIDAY_FACTOR, mult)
    mult = np.where(is_year_end, mult * YEAR_END_FACTOR, mult)
    noise = np.exp(rng.normal(0.0, cfg.daily_noise_sigma, len(days)))
    return np.asarray(mult) * noise, is_hol


def _sample_rows(rng, cum_probs: np.ndarray) -> np.ndarray:
    """Vectorised categorical sampling: one draw per row of cumulative probabilities."""
    u = rng.random(cum_probs.shape[0])
    return (u[:, None] > cum_probs).sum(axis=1).clip(0, cum_probs.shape[1] - 1)


def generate_tickets(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate the synthetic ticket table (deterministic for a given config)."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)
    end_excl = pd.Timestamp(cfg.end) + pd.Timedelta(days=1)
    days = pd.date_range(start, cfg.end, freq="D")
    hours = pd.date_range(start, end_excl, freq="h", inclusive="left")
    n_hours = len(hours)

    # Business-hour grid extends well past the extract so late tickets can finish.
    grid_hours = pd.date_range(start, end_excl + pd.Timedelta(days=150), freq="h", inclusive="left")
    holidays = holidays_between(start, grid_hours[-1])
    bh_flags = business_hour_flags(grid_hours, holidays)

    day_mult, day_is_hol = _day_multipliers(days, holidays_between(start, cfg.end), cfg, rng)
    daily_expected = cfg.base_weekday_daily * day_mult

    wk = WEEKDAY_HOURS / WEEKDAY_HOURS.sum()
    we = WEEKEND_HOURS / WEEKEND_HOURS.sum()
    day_idx = np.repeat(np.arange(len(days)), 24)
    hod = np.tile(np.arange(24), len(days))
    quiet_day = (days.dayofweek.values >= 5) | day_is_hol
    shape = np.where(quiet_day[day_idx], we[hod], wk[hod])
    lam = daily_expected[day_idx] * shape
    base_counts = rng.poisson(lam)

    # Incident surges
    activity = WEEKDAY_HOURS / WEEKDAY_HOURS.max()
    inc_hour_idx, inc_cat = [], []
    for inc in cfg.incidents:
        h0 = int((pd.Timestamp(inc.start) - start) / pd.Timedelta(hours=1))
        if h0 < 0 or h0 >= n_hours:
            continue
        t = np.arange(min(inc.duration_hours, n_hours - h0))
        hh = h0 + t
        lam_inc = inc.peak_per_hour * np.exp(-t / inc.decay_hours) * (0.35 + 0.65 * activity[hh % 24])
        c = rng.poisson(lam_inc)
        inc_hour_idx.append(np.repeat(hh, c))
        inc_cat.append(np.full(c.sum(), CATEGORIES.index(inc.category)))

    base_idx = np.repeat(np.arange(n_hours), base_counts)
    n_base = len(base_idx)
    base_cat = _sample_rows(rng, np.tile(np.cumsum(CATEGORY_MIX), (n_base, 1)))
    hour_idx = np.concatenate([base_idx] + inc_hour_idx)
    cat = np.concatenate([base_cat] + inc_cat).astype(int)
    is_incident = np.concatenate([np.zeros(n_base, bool), np.ones(len(hour_idx) - n_base, bool)])
    n = len(hour_idx)

    created_h = hour_idx + rng.random(n)

    # Priority
    pri_cum = np.cumsum(PRIORITY_MIX, axis=1)[cat]
    pri_cum[is_incident] = np.cumsum(INCIDENT_PRIORITY_MIX)
    pri = _sample_rows(rng, pri_cum)

    # Channel (phone/chat shift to portal outside business hours)
    chan = _sample_rows(rng, np.cumsum(CHANNEL_MIX, axis=1)[cat])
    off_hours = ~bh_flags[hour_idx]
    live = np.isin(chan, [CHANNELS.index("phone"), CHANNELS.index("chat")])
    chan = np.where(off_hours & live & (rng.random(n) < 0.6), CHANNELS.index("portal"), chan)

    # Assignee group
    l1_prob = L1_SHARE[cat] * (pri != 0) * (chan != CHANNELS.index("monitoring"))
    is_l1 = rng.random(n) < l1_prob
    group_names = np.array(GROUPS)
    cat_group_idx = np.array([GROUPS.index(CATEGORY_GROUP[c]) for c in CATEGORIES])
    grp = np.where(is_l1, GROUPS.index(L1_GROUP), cat_group_idx[cat])

    # Handle time (agent touch time, minutes)
    handle = HANDLE_MEDIAN_MIN[cat] * CHANNEL_HANDLE_MULT[chan] * np.exp(rng.normal(0, 0.45, n))
    handle = np.round(np.clip(handle, 2.0, 180.0), 1)

    # Queue load: trailing 72h arrivals per group relative to that group's average
    counts = np.zeros((len(GROUPS), n_hours))
    np.add.at(counts, (grp, hour_idx), 1.0)
    csum = np.cumsum(counts, axis=1)
    trailing = csum - np.concatenate([np.zeros((len(GROUPS), LOAD_WINDOW_H)), csum[:, :-LOAD_WINDOW_H]], axis=1)
    ratio = trailing / np.maximum(trailing[:, LOAD_WINDOW_H:].mean(axis=1, keepdims=True), 1e-9)
    load_mult = np.clip(ratio[grp, hour_idx], 0.5, 6.0) ** LOAD_EXPONENT

    # Capacity shocks: absences, training weeks and project work slow a group down for days at a time
    shocks = np.zeros((len(GROUPS), len(days)))
    eps = rng.normal(0.0, CAPACITY_SHOCK_SD, shocks.shape)
    for t in range(1, len(days)):
        shocks[:, t] = CAPACITY_SHOCK_PHI * shocks[:, t - 1] + eps[:, t]
    capacity_mult = np.exp(shocks[grp, hour_idx // 24])

    # Work time and resolution timestamp
    median = PRIORITY_WORK_MEDIAN_H[pri] * CATEGORY_WORK_MULT[cat] * np.where(is_l1, L1_WORK_MULT, 1.0)
    work = median * load_mult * capacity_mult * np.exp(rng.normal(0, WORK_SIGMA, n))
    work = np.maximum(work, 0.05)
    resolved_h = created_h + work
    bh_only = pri >= 2
    resolved_h[bh_only] = add_business_hours(created_h[bh_only], work[bh_only], bh_flags)

    # Reopens add a second stretch of calendar time
    p_reopen = 0.035 + 0.03 * (cat == CATEGORIES.index("software")) + 0.03 * is_l1 + 0.04 * (load_mult > 1.5)
    reopened = rng.random(n) < p_reopen
    resolved_h = resolved_h + reopened * 18.0 * np.exp(rng.normal(0, 0.7, n))

    # Censor: tickets still open at the extract time have no resolved_at
    extract_h = n_hours
    resolved_h = np.where(resolved_h > extract_h, np.nan, resolved_h)

    order = np.argsort(created_h, kind="stable")
    created_ts = start + pd.to_timedelta(np.round(created_h[order] * 3600), unit="s")
    res_sec = np.round(resolved_h[order] * 3600)
    resolved_ts = start + pd.to_timedelta(res_sec, unit="s")

    df = pd.DataFrame(
        {
            "ticket_id": [f"INC{i:07d}" for i in range(1, n + 1)],
            "created_at": created_ts,
            "resolved_at": resolved_ts,
            "category": np.array(CATEGORIES)[cat[order]],
            "priority": np.array(PRIORITIES)[pri[order]],
            "channel": np.array(CHANNELS)[chan[order]],
            "assignee_group": group_names[grp[order]],
            "handle_minutes": handle[order],
            "reopened": reopened[order].astype(int),
        }
    )
    return df[TICKET_COLUMNS]


def save_tickets(df: pd.DataFrame, path) -> None:
    out = df.copy()
    fmt = "%Y-%m-%d %H:%M:%S"
    out["created_at"] = out["created_at"].dt.strftime(fmt)
    out["resolved_at"] = out["resolved_at"].dt.strftime(fmt).fillna("")
    out.to_csv(path, index=False)


def load_tickets(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"resolved_at": "string"})
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M:%S")
    df["resolved_at"] = pd.to_datetime(df["resolved_at"].replace("", pd.NA), format="%Y-%m-%d %H:%M:%S")
    for col in ("category", "priority", "channel", "assignee_group", "ticket_id"):
        df[col] = df[col].astype(str)
    df["reopened"] = df["reopened"].astype(int)
    return df[TICKET_COLUMNS]


def validate_tickets(df: pd.DataFrame) -> dict:
    """Basic timestamp and schema data-quality checks."""
    res = df["resolved_at"]
    return {
        "rows": int(len(df)),
        "duplicate_ids": int(df["ticket_id"].duplicated().sum()),
        "missing_created": int(df["created_at"].isna().sum()),
        "open_at_extract": int(res.isna().sum()),
        "resolved_before_created": int((res < df["created_at"]).sum()),
        "unknown_priority": int((~df["priority"].isin(PRIORITIES)).sum()),
        "unknown_category": int((~df["category"].isin(CATEGORIES)).sum()),
    }

"""Company holiday calendar and business-hour arithmetic."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import BUSINESS_END_HOUR, BUSINESS_START_HOUR


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday (Mon=0) of a month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """Saturday holidays are observed on Friday, Sunday holidays on Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def company_holidays(year: int) -> dict[date, str]:
    """US-style company holidays (observed dates) for one calendar year."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    raw = {
        _observed(date(year, 1, 1)): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Presidents' Day",
        _last_weekday(year, 5, 0): "Memorial Day",
        _observed(date(year, 6, 19)): "Juneteenth",
        _observed(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        thanksgiving: "Thanksgiving",
        thanksgiving + timedelta(days=1): "Day after Thanksgiving",
        _observed(date(year, 12, 25)): "Christmas Day",
    }
    return raw


def holidays_between(start, end) -> dict[date, str]:
    """All observed holidays with start <= date <= end."""
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    out: dict[date, str] = {}
    for year in range(start.year - 1, end.year + 2):
        for d, name in company_holidays(year).items():
            if start <= d <= end:
                out[d] = name
    return dict(sorted(out.items()))


def is_business_day(d: date, holidays) -> bool:
    return d.weekday() < 5 and d not in holidays


def next_business_day(d: date, holidays) -> date:
    d = d + timedelta(days=1)
    while not is_business_day(d, holidays):
        d += timedelta(days=1)
    return d


def business_hour_flags(hours: pd.DatetimeIndex, holidays) -> np.ndarray:
    """Boolean array: True for hours that fall inside Mon-Fri business hours on non-holidays."""
    hol = pd.DatetimeIndex(pd.to_datetime(sorted(holidays))) if holidays else pd.DatetimeIndex([])
    in_hours = (hours.hour >= BUSINESS_START_HOUR) & (hours.hour < BUSINESS_END_HOUR)
    weekday = hours.dayofweek < 5
    not_holiday = ~hours.normalize().isin(hol)
    return np.asarray(in_hours & weekday & not_holiday, dtype=bool)


def business_cumulative(flags: np.ndarray) -> np.ndarray:
    """cum[i] = number of business hours strictly before hour i (length n + 1)."""
    return np.concatenate([[0.0], np.cumsum(flags.astype(float))])


def add_business_hours(start_h: np.ndarray, work_h: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Add `work_h` business hours to fractional hour offsets `start_h` on an hourly grid.

    `flags[i]` says whether grid hour i (offset i..i+1) is a business hour. Work only
    accrues during business hours, so a ticket logged Friday 17:00 with 2 hours of work
    finishes Monday 09:00. `work_h` must be strictly positive.
    """
    start_h = np.asarray(start_h, dtype=float)
    work_h = np.asarray(work_h, dtype=float)
    if np.any(work_h <= 0):
        raise ValueError("work_h must be > 0")
    cum = business_cumulative(flags)
    i = np.floor(start_h).astype(int)
    if i.min() < 0 or i.max() >= len(flags):
        raise ValueError("start_h outside the business-hour grid")
    frac = start_h - i
    bh_start = cum[i] + flags[i] * frac
    target = bh_start + work_h
    if target.max() > cum[-1]:
        raise ValueError("business-hour grid too short for requested work")
    k = np.searchsorted(cum, target, side="left")
    return (k - 1) + (target - cum[k - 1])

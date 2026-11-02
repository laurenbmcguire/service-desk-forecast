"""Erlang C staffing: turn hourly arrivals and average handle time into agents needed."""
from __future__ import annotations

import math

import numpy as np

from .config import ANSWER_TIME_SECONDS, MAX_OCCUPANCY, MIN_AGENTS, SERVICE_LEVEL_TARGET, SHRINKAGE


def offered_load(arrivals_per_hour: float, aht_minutes: float) -> float:
    """Traffic intensity A in Erlangs: arrival rate x average handle time."""
    return arrivals_per_hour * aht_minutes / 60.0


def erlang_b(n: int, a: float) -> float:
    """Erlang B blocking probability via the stable recursion B(k) = A B(k-1) / (k + A B(k-1))."""
    b = 1.0
    for k in range(1, n + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(n: int, a: float) -> float:
    """Probability an arrival has to wait, P(W > 0), for n agents and load a Erlangs."""
    if a <= 0:
        return 0.0
    if n <= a:
        return 1.0
    b = erlang_b(n, a)
    return n * b / (n - a * (1.0 - b))


def service_level(n: int, arrivals_per_hour: float, aht_minutes: float, answer_seconds: float = ANSWER_TIME_SECONDS) -> float:
    """Share of arrivals answered within `answer_seconds`: 1 - C(n, A) exp(-(n - A) t / AHT)."""
    a = offered_load(arrivals_per_hour, aht_minutes)
    if a <= 0:
        return 1.0
    if n <= a:
        return 0.0
    pw = erlang_c(n, a)
    return 1.0 - pw * math.exp(-(n - a) * answer_seconds / (aht_minutes * 60.0))


def agents_required(
    arrivals_per_hour: float,
    aht_minutes: float,
    target_sl: float = SERVICE_LEVEL_TARGET,
    answer_seconds: float = ANSWER_TIME_SECONDS,
    max_occupancy: float = MAX_OCCUPANCY,
    min_agents: int = MIN_AGENTS,
) -> int:
    """Smallest n meeting the service-level target and the occupancy cap A / n <= max_occupancy."""
    a = offered_load(arrivals_per_hour, aht_minutes)
    if a <= 0:
        return int(min_agents)
    n = max(int(min_agents), int(math.floor(a)) + 1)
    while service_level(n, arrivals_per_hour, aht_minutes, answer_seconds) < target_sl or a / n > max_occupancy:
        n += 1
    return n


def staffing_matrix(hourly_arrivals: np.ndarray, aht_minutes: float, **kwargs) -> np.ndarray:
    """Apply `agents_required` to every cell of a (days x hours) arrival matrix."""
    arr = np.asarray(hourly_arrivals, float)
    out = np.zeros(arr.shape, dtype=int)
    for idx, lam in np.ndenumerate(arr):
        out[idx] = agents_required(float(lam), aht_minutes, **kwargs)
    return out


def scheduled_fte(agent_hours_per_week: float, shrinkage: float = SHRINKAGE, hours_per_fte: float = 40.0) -> float:
    """Convert on-queue agent hours into scheduled FTE after shrinkage (breaks, training, absence)."""
    return agent_hours_per_week / (1.0 - shrinkage) / hours_per_fte

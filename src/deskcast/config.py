"""Shared constants: ticket taxonomy, SLA targets and model defaults."""
from __future__ import annotations

CATEGORIES = ("access", "hardware", "email", "network", "software", "security")
PRIORITIES = ("P1", "P2", "P3", "P4")
CHANNELS = ("portal", "email", "phone", "chat", "monitoring")

L1_GROUP = "Service Desk L1"
CATEGORY_GROUP = {
    "access": "Identity & Access",
    "hardware": "End-User Computing",
    "email": "Messaging",
    "network": "Network Ops",
    "software": "Application Support",
    "security": "Security Ops",
}
GROUPS = (L1_GROUP,) + tuple(CATEGORY_GROUP[c] for c in CATEGORIES)

# Resolution SLA targets in calendar hours, measured from creation to final resolution.
SLA_TARGET_HOURS = {"P1": 4.0, "P2": 8.0, "P3": 72.0, "P4": 120.0}

# Business hours used for P3/P4 work and for the "is_business_hours" feature.
BUSINESS_START_HOUR = 8
BUSINESS_END_HOUR = 18  # exclusive

# Forecasting defaults
FORECAST_HORIZON = 28
BACKTEST_FOLDS = 6
INTERVAL_LEVEL = 0.80

# Staffing defaults (classic contact-center 80/20 target)
SERVICE_LEVEL_TARGET = 0.80
ANSWER_TIME_SECONDS = 20
MAX_OCCUPANCY = 0.85
SHRINKAGE = 0.30
MIN_AGENTS = 1

DOW_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Human-readable names for SLA model features (charts and AI PM text)
FEATURE_LABELS = {
    "dow": "Day of week",
    "assignee_group": "Assignee group",
    "queue_open_group": "Open queue, same group",
    "queue_open_total": "Open queue, whole desk",
    "arrivals_prev_24h_group": "Group arrivals, last 24 h",
    "priority": "Priority",
    "category": "Category",
    "hour": "Hour of day",
    "channel": "Channel",
    "is_holiday": "Holiday",
    "is_business_hours": "Inside business hours",
}

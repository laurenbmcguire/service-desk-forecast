import numpy as np
import pandas as pd

from deskcast.config import CATEGORIES, CHANNELS, GROUPS, PRIORITIES
from deskcast.generator import TICKET_COLUMNS, GeneratorConfig, generate_tickets, load_tickets, save_tickets, validate_tickets


def test_generator_is_deterministic_and_valid(small_tickets, tmp_path):
    again = generate_tickets(GeneratorConfig(start="2025-01-01", end="2025-09-30", seed=7))
    pd.testing.assert_frame_equal(small_tickets, again)
    assert list(small_tickets.columns) == TICKET_COLUMNS
    assert set(small_tickets["category"]) == set(CATEGORIES)
    assert set(small_tickets["priority"]) <= set(PRIORITIES)
    assert set(small_tickets["channel"]) <= set(CHANNELS)
    assert set(small_tickets["assignee_group"]) <= set(GROUPS)
    q = validate_tickets(small_tickets)
    assert q["duplicate_ids"] == 0 and q["resolved_before_created"] == 0
    # CSV round trip keeps the schema
    save_tickets(small_tickets.head(200), tmp_path / "t.csv")
    back = load_tickets(tmp_path / "t.csv")
    assert list(back.columns) == TICKET_COLUMNS and len(back) == 200


def test_generator_has_weekly_and_hourly_seasonality(small_tickets):
    created = small_tickets["created_at"]
    daily = created.dt.normalize().value_counts().sort_index()
    by_dow = daily.groupby(daily.index.dayofweek).mean()
    assert by_dow.loc[[0, 1, 2, 3, 4]].min() > 3 * by_dow.loc[[5, 6]].max()
    by_hour = created.dt.hour.value_counts()
    assert by_hour.idxmax() in range(9, 16)
    # The 2025-03-11 email outage shows up as a spike in email tickets
    email = small_tickets[small_tickets["category"] == "email"]
    email_daily = email["created_at"].dt.normalize().value_counts()
    assert email_daily.idxmax() == pd.Timestamp("2025-03-11")

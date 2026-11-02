from datetime import date

import numpy as np
import pandas as pd

from deskcast.calendar_utils import add_business_hours, business_hour_flags, company_holidays, holidays_between


def test_holiday_rules_and_observed_dates():
    h25 = company_holidays(2025)
    assert date(2025, 11, 27) in h25  # Thanksgiving: 4th Thursday
    assert date(2025, 11, 28) in h25  # day after
    assert date(2025, 5, 26) in h25  # Memorial Day: last Monday of May
    assert date(2025, 1, 20) in h25  # MLK: 3rd Monday of January
    h26 = holidays_between("2026-07-01", "2026-07-31")
    assert date(2026, 7, 3) in h26  # July 4 2026 is a Saturday -> observed Friday


def test_add_business_hours_rolls_over_weekend():
    hours = pd.date_range("2025-03-07", periods=24 * 7, freq="h")  # Friday 00:00
    flags = business_hour_flags(hours, holidays_between("2025-03-01", "2025-03-31"))
    friday_5pm = 17.0
    end = add_business_hours(np.array([friday_5pm]), np.array([2.0]), flags)[0]
    # 1 business hour left on Friday (17-18), the second lands Monday 08:00-09:00
    assert hours[0] + pd.Timedelta(hours=end) == pd.Timestamp("2025-03-10 09:00")
    # Saturday start waits for Monday opening
    sat = 24 + 10.5
    end2 = add_business_hours(np.array([sat]), np.array([0.5]), flags)[0]
    assert hours[0] + pd.Timedelta(hours=end2) == pd.Timestamp("2025-03-10 08:30")

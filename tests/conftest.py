import pandas as pd
import pytest

from deskcast.generator import GeneratorConfig, generate_tickets


@pytest.fixture(scope="session")
def small_tickets() -> pd.DataFrame:
    """About 9 months of tickets: enough for backtests, fast enough for unit tests."""
    return generate_tickets(GeneratorConfig(start="2025-01-01", end="2025-09-30", seed=7))

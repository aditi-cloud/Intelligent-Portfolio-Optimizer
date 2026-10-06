from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from portfolio_optimizer import PortfolioRequest
from portfolio_optimizer.data import PriceDataset


@pytest.fixture
def data_request():
    return PortfolioRequest(("RELIANCE.NS", "TCS.NS"), 100_000, date(2025, 1, 2), date(2025, 1, 9))


@pytest.fixture
def price_dataset():
    prices = pd.read_csv(Path(__file__).parent / "fixtures" / "prices.csv", index_col="Date", parse_dates=True)
    return PriceDataset(prices, {"RELIANCE.NS": "INR", "TCS.NS": "INR"},
                        "fixture", datetime(2025, 1, 10, tzinfo=timezone.utc))


@pytest.fixture
def analysis_result(price_dataset, data_request):
    from portfolio_optimizer import AnalysisConfig
    from portfolio_optimizer.data.memory import InMemoryPriceProvider
    from portfolio_optimizer.evaluation.backtest import BacktestConfig
    from portfolio_optimizer.services import AnalysisOptions, PortfolioAnalysisService
    service = PortfolioAnalysisService(InMemoryPriceProvider(price_dataset), AnalysisConfig(min_observations=2),
                                       clock=lambda: datetime(2026, 10, 6, tzinfo=timezone.utc))
    return service.analyze(data_request, AnalysisOptions(frontier_points=4, simulation_scenarios=20,
                                                        backtest=BacktestConfig(2, 1, 10)))

from datetime import date
from io import BytesIO
import json
from unittest.mock import Mock
from zipfile import ZipFile

import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data.cache import CachedPriceProvider
from portfolio_optimizer.data.yahoo import YahooPriceProvider
from portfolio_optimizer.evaluation.backtest import BacktestConfig
from portfolio_optimizer.services import AnalysisOptions, PortfolioAnalysisService, report_bundle


@pytest.mark.integration
def test_request_to_downloadable_report(tmp_path, price_dataset, data_request):
    download = Mock(return_value=pd.concat({"Close": price_dataset.prices}, axis=1))
    yahoo = YahooPriceProvider(download=download, currency_lookup=lambda ticker: "INR", clock=lambda: price_dataset.retrieved_at)
    provider = CachedPriceProvider(yahoo, tmp_path, clock=lambda: price_dataset.retrieved_at)
    service = PortfolioAnalysisService(provider, AnalysisConfig(min_observations=2), clock=lambda: price_dataset.retrieved_at)
    result = service.analyze(data_request, AnalysisOptions(frontier_points=4, simulation_scenarios=20,
                                                          backtest=BacktestConfig(2, 1, 10)), today=date(2026, 10, 6))
    bundle = report_bundle(result)
    download.assert_called_once()
    assert result.simulation is not None and result.backtest is not None
    assert not result.issues
    with ZipFile(BytesIO(bundle)) as archive:
        data = json.loads(archive.read("analysis.json"))
        assert data["data"]["provider"] == "yahoo"
        assert data["portfolio"]["status"] == "optimal"
        assert archive.read("report.html").startswith(b"<!doctype html>")

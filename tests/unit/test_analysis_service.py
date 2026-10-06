from dataclasses import replace
from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data.memory import InMemoryPriceProvider
from portfolio_optimizer.errors import BacktestError, OptimizationError, ProviderError, RequestValidationError, SimulationError
from portfolio_optimizer.evaluation.backtest import BacktestConfig
from portfolio_optimizer.services import AnalysisOptions, PortfolioAnalysisService


@pytest.fixture
def service(price_dataset):
    provider = Mock(wraps=InMemoryPriceProvider(price_dataset))
    return PortfolioAnalysisService(provider, AnalysisConfig(min_observations=2),
                                     clock=lambda: datetime(2026, 10, 6, tzinfo=timezone.utc))


def test_fetches_once_and_returns_consistent_result(service, data_request):
    result = service.analyze(data_request, AnalysisOptions(frontier_points=3, simulation_scenarios=10))
    service.provider.fetch_prices.assert_called_once_with(result.request)
    assert result.estimates.tickers == result.request.tickers
    assert result.simulation.estimation_cutoff == result.estimates.cutoff
    assert result.simulation.seed == result.config.random_seed
    assert result.backtest is None
    assert result.issues == ()
    assert result.generated_at.utcoffset() is not None


def test_disabled_evaluation_never_called(service, data_request):
    with patch("portfolio_optimizer.services.analysis.simulate_portfolio") as simulation:
        with patch("portfolio_optimizer.services.analysis.compare_backtests") as replay:
            result = service.analyze(data_request, AnalysisOptions(frontier_points=3, include_simulation=False))
    simulation.assert_not_called()
    replay.assert_not_called()
    assert result.simulation is None and result.backtest is None


def test_partial_simulation_failure_is_explicit(service, data_request):
    with patch("portfolio_optimizer.services.analysis.simulate_portfolio", side_effect=SimulationError("incompatible moments")):
        result = service.analyze(data_request, AnalysisOptions(frontier_points=3, simulation_scenarios=10))
    assert result.simulation is None
    assert result.issues[0].component == "simulation"
    assert "incompatible" in result.issues[0].message
    assert result.portfolio.status == "optimal"


def test_strict_optional_failure_propagates(service, data_request):
    with patch("portfolio_optimizer.services.analysis.simulate_portfolio", side_effect=SimulationError("failed")):
        with pytest.raises(SimulationError):
            service.analyze(data_request, AnalysisOptions(frontier_points=3, allow_partial_evaluation=False))


def test_short_history_preserves_core_analysis(service, data_request):
    result = service.analyze(data_request, AnalysisOptions(frontier_points=3, include_simulation=False,
                                                          backtest=BacktestConfig(60)))
    assert result.backtest is None
    assert result.issues[0].component == "backtest"
    assert result.allocation.budget == 100_000


def test_programming_errors_are_not_hidden(service, data_request):
    with patch("portfolio_optimizer.services.analysis.simulate_portfolio", side_effect=RuntimeError("bug")):
        with pytest.raises(RuntimeError, match="bug"):
            service.analyze(data_request, AnalysisOptions(frontier_points=3))


def test_core_optimization_failure_propagates(service, data_request):
    with patch("portfolio_optimizer.services.analysis.optimize_portfolio", side_effect=OptimizationError("failed")):
        with pytest.raises(OptimizationError):
            service.analyze(data_request)


def test_cache_io_error_is_provider_error(service, data_request):
    service.provider.fetch_prices.side_effect = PermissionError("cannot write cache")
    with pytest.raises(ProviderError, match="cache I/O"):
        service.analyze(data_request)


@pytest.mark.parametrize("updates", [{"investment_amount": 0}, {"estimator": "xgboost"}])
def test_invalid_request_rejected_before_fetch(service, data_request, updates):
    with pytest.raises(RequestValidationError):
        service.analyze(replace(data_request, **updates))
    service.provider.fetch_prices.assert_not_called()


def test_future_date_uses_supplied_local_date(service, data_request):
    with pytest.raises(RequestValidationError, match="future"):
        service.analyze(data_request, today=date(2025, 1, 8))
    service.provider.fetch_prices.assert_not_called()


def test_resource_limit_checked_before_fetch(service, data_request):
    with pytest.raises(RequestValidationError, match="stored-value"):
        service.analyze(data_request, AnalysisOptions(simulation_scenarios=1_000_000))
    service.provider.fetch_prices.assert_not_called()


@pytest.mark.parametrize("updates", [
    {"frontier_points": 1}, {"frontier_points": True}, {"simulation_scenarios": 1},
    {"include_simulation": "yes"}, {"allow_partial_evaluation": 1}, {"backtest": {}},
])
def test_invalid_options(updates):
    with pytest.raises(ValueError):
        AnalysisOptions(**updates)


def test_invalid_options_type(service, data_request):
    with pytest.raises(RequestValidationError):
        service.analyze(data_request, {})
    service.provider.fetch_prices.assert_not_called()


def test_strict_backtest_failure_propagates(service, data_request):
    with pytest.raises(BacktestError):
        service.analyze(data_request, AnalysisOptions(frontier_points=3, include_simulation=False,
                                                     backtest=BacktestConfig(60), allow_partial_evaluation=False))


def test_currency_precision_rejected_before_fetch(service, data_request):
    with pytest.raises(OptimizationError, match="decimal places"):
        service.analyze(replace(data_request, investment_amount=.001))
    service.provider.fetch_prices.assert_not_called()

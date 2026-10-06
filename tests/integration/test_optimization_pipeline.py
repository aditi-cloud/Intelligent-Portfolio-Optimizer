from decimal import Decimal

import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.evaluation import equal_weight_metrics
from portfolio_optimizer.forecasting import estimate_market
from portfolio_optimizer.optimization import allocate_currency, build_frontier, optimize_portfolio


@pytest.mark.integration
def test_prices_to_optimized_allocation(price_dataset, data_request):
    config = AnalysisConfig(min_observations=2)
    prepared = prepare_data(price_dataset, data_request, config)
    estimates = estimate_market(prepared, data_request, config)
    result = optimize_portfolio(estimates, data_request, config)
    benchmark = equal_weight_metrics(estimates)
    allocation = allocate_currency(result.weights, data_request.investment_amount, currency=data_request.currency)
    frontier = build_frontier(estimates, max_asset_weight=data_request.max_asset_weight, points=5)
    assert tuple(result.weights.index) == data_request.tickers
    assert sum(allocation.amounts.values()) == Decimal("100000")
    assert len(frontier.points) == 5
    # The objective optimized here is utility, not Sharpe ratio.
    benchmark_utility = benchmark.expected_annual_return - config.risk_aversion(data_request.risk_level) * benchmark.annual_volatility**2
    assert result.objective_value >= benchmark_utility - 1e-6

import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.evaluation.backtest import BacktestConfig, compare_backtests
from portfolio_optimizer.evaluation.simulation import simulate_portfolio
from portfolio_optimizer.forecasting import estimate_market
from portfolio_optimizer.optimization import optimize_portfolio


@pytest.mark.integration
def test_replay_comparison_and_scenarios(price_dataset, data_request):
    config = AnalysisConfig(min_observations=2)
    prepared = prepare_data(price_dataset, data_request, config)
    comparison = compare_backtests(prepared, data_request, config, BacktestConfig(2, 1, 10))
    assert comparison.optimized.equity.index.equals(comparison.equal_weight.equity.index)
    assert comparison.optimized.settings == comparison.equal_weight.settings
    assert comparison.optimized.total_cost > 0
    assert comparison.equal_weight.total_cost > 0
    estimates = estimate_market(prepared, data_request, config)
    portfolio = optimize_portfolio(estimates, data_request, config)
    scenarios = simulate_portfolio(estimates, portfolio.weights, data_request.investment_amount,
                                   horizon_days=data_request.horizon_days, scenarios=100, seed=config.random_seed)
    assert scenarios.horizon_days == data_request.horizon_days
    assert scenarios.terminal_quantiles["p05"] <= scenarios.terminal_quantiles["p95"]

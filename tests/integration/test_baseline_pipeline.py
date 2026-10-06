import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.evaluation import equal_weight_metrics
from portfolio_optimizer.forecasting import estimate_market


@pytest.mark.integration
def test_prices_to_estimates_to_benchmark(price_dataset, data_request):
    config = AnalysisConfig(min_observations=2, trading_days_per_year=10, annual_risk_free_rate=.03)
    prepared = prepare_data(price_dataset, data_request, config)
    estimates = estimate_market(prepared, data_request, config)
    benchmark = equal_weight_metrics(estimates, annual_risk_free_rate=config.annual_risk_free_rate)
    assert estimates.tickers == data_request.tickers
    assert estimates.cutoff <= data_request.end_date
    assert benchmark.expected_annual_return == pytest.approx(.25)
    assert benchmark.annual_volatility > 0
    assert benchmark.expected_sharpe_ratio == pytest.approx((.25 - .03) / benchmark.annual_volatility)

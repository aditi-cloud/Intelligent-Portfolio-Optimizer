from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.errors import SimulationError
from portfolio_optimizer.evaluation.simulation import lognormal_parameters, simulate_portfolio
from portfolio_optimizer.forecasting import MarketEstimates


@pytest.fixture
def simulation_market():
    return MarketEstimates(pd.Series([.1, .2], index=["A", "B"]),
                          pd.DataFrame([[.04, .01], [.01, .09]], index=["A", "B"], columns=["A", "B"]),
                          date(2025, 1, 1), 60)


@pytest.fixture
def weights():
    return pd.Series([.4, .6], index=["A", "B"])


def test_seed_reproducibility_and_shapes(simulation_market, weights):
    a = simulate_portfolio(simulation_market, weights, 100, horizon_days=10, scenarios=100)
    b = simulate_portfolio(simulation_market, weights, 100, horizon_days=10, scenarios=100)
    pd.testing.assert_frame_equal(a.wealth_paths, b.wealth_paths)
    assert a.wealth_paths.shape == (11, 100)
    assert (a.wealth_paths > 0).all().all()
    assert (a.wealth_paths.iloc[0] == 100).all()
    assert (np.diff(a.path_quantiles, axis=1) >= 0).all()
    assert 0 <= a.model_loss_probability <= 1
    assert a.assumptions


def test_zero_volatility_deterministic_growth(simulation_market, weights):
    market = replace(simulation_market, covariance=simulation_market.covariance * 0)
    result = simulate_portfolio(market, weights, 100, horizon_days=10, scenarios=100)
    expected = 100 * (.4 * (1 + .1/252)**10 + .6 * (1 + .2/252)**10)
    np.testing.assert_allclose(result.wealth_paths.iloc[-1], expected)
    assert result.model_loss_probability == 0


def test_moment_matching_is_exact_in_parameters(simulation_market):
    mean, covariance, _ = lognormal_parameters(simulation_market)
    gross_mean = np.exp(mean + .5 * np.diag(covariance))
    gross_cov = np.outer(gross_mean, gross_mean) * np.expm1(covariance)
    np.testing.assert_allclose(gross_mean - 1, simulation_market.expected_returns / 252, atol=1e-15)
    np.testing.assert_allclose(gross_cov, simulation_market.covariance / 252, atol=1e-15)


def test_sample_moments_agree_with_estimates(simulation_market, weights):
    result = simulate_portfolio(simulation_market, weights, 100, horizon_days=1, scenarios=120_000)
    samples = result.terminal_asset_returns
    expected_mean = simulation_market.expected_returns / 252
    expected_cov = simulation_market.covariance / 252
    standard_error = np.sqrt(np.diag(expected_cov) / len(samples))
    assert (np.abs(samples.mean() - expected_mean) < 5 * standard_error).all()
    # Low daily variances make the Gaussian covariance SE a useful approximation.
    # Five SE bounds account for finite sampling instead of using a fixed epsilon.
    matrix = expected_cov.to_numpy()
    covariance_se = np.sqrt((np.outer(np.diag(matrix), np.diag(matrix)) + matrix**2) / len(samples))
    assert (np.abs(samples.cov().to_numpy() - matrix) < 5 * covariance_se).all()


def test_weight_order_and_budget_scaling(simulation_market, weights):
    a = simulate_portfolio(simulation_market, weights, 100, scenarios=10)
    b = simulate_portfolio(simulation_market, weights.iloc[::-1], 200, scenarios=10)
    np.testing.assert_allclose(b.wealth_paths, 2 * a.wealth_paths)


def test_incompatible_log_covariance_rejected(simulation_market, weights):
    covariance = pd.DataFrame([[2.52, 5.04], [5.04, 10.08]], index=["A", "B"], columns=["A", "B"])
    market = replace(simulation_market, covariance=covariance)
    with pytest.raises(SimulationError, match="positive semidefinite"):
        simulate_portfolio(market, weights, 100)


@pytest.mark.parametrize("updates", [
    {"horizon_days": 0}, {"horizon_days": True}, {"scenarios": 1},
    {"scenarios": 1.5}, {"seed": -1}, {"seed": True},
    {"scenarios": 10_000_000},
])
def test_invalid_simulation_settings(simulation_market, weights, updates):
    with pytest.raises(SimulationError):
        simulate_portfolio(simulation_market, weights, 100, **updates)


@pytest.mark.parametrize("amount", [0, -1, np.nan, np.inf, True])
def test_invalid_simulation_amount(simulation_market, weights, amount):
    with pytest.raises(SimulationError):
        simulate_portfolio(simulation_market, weights, amount)


def test_nonpositive_expected_gross_return(simulation_market, weights):
    market = replace(simulation_market, expected_returns=pd.Series([-252, .1], index=["A", "B"]))
    with pytest.raises(SimulationError, match="gross returns"):
        simulate_portfolio(market, weights, 100)


def test_different_seed_changes_scenarios(simulation_market, weights):
    a = simulate_portfolio(simulation_market, weights, 100, scenarios=10, seed=1)
    b = simulate_portfolio(simulation_market, weights, 100, scenarios=10, seed=2)
    assert not a.wealth_paths.equals(b.wealth_paths)
    assert a.initial_wealth == 100
    assert a.estimation_cutoff == simulation_market.cutoff
    assert a.trading_days_per_year == 252

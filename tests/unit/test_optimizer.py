from dataclasses import replace
from datetime import date
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig, PortfolioRequest
from portfolio_optimizer.errors import OptimizationError, RequestValidationError
from portfolio_optimizer.forecasting import MarketEstimates
from portfolio_optimizer.optimization import build_frontier, minimum_variance, optimize_portfolio


@pytest.fixture
def market():
    return MarketEstimates(pd.Series([.1, .2], index=["A", "B"]),
                          pd.DataFrame([[.04, 0], [0, .09]], index=["A", "B"], columns=["A", "B"]),
                          date(2025, 1, 1), 60)


@pytest.fixture
def optimization_request():
    return PortfolioRequest(("A", "B"), 100_000, date(2024, 1, 1), date(2025, 1, 1))


def test_analytic_minimum_variance(market):
    result = minimum_variance(market)
    np.testing.assert_allclose(result.weights, [9/13, 4/13], atol=1e-5)
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(.04 * .09 / .13, abs=1e-8)


def test_analytic_utility_optimum(market, optimization_request):
    # Solve derivative: x=(2*lambda*.09 - .1)/(2*lambda*.13).
    result = optimize_portfolio(market, optimization_request)
    expected_a = (2 * 3 * .09 - .1) / (2 * 3 * .13)
    np.testing.assert_allclose(result.weights, [expected_a, 1 - expected_a], atol=1e-5)


def test_equal_assets_equal_weights(market, optimization_request):
    symmetric = replace(market, expected_returns=pd.Series([.1, .1], index=["A", "B"]),
                        covariance=pd.DataFrame(np.eye(2) * .04, index=["A", "B"], columns=["A", "B"]))
    np.testing.assert_allclose(optimize_portfolio(symmetric, optimization_request).weights, [.5, .5], atol=1e-6)


def test_cap_and_sum(market, optimization_request):
    result = optimize_portfolio(market, replace(optimization_request, risk_level="high", max_asset_weight=.6))
    assert result.weights.sum() == pytest.approx(1, abs=1e-10)
    assert (result.weights >= 0).all()
    assert (result.weights <= .6 + 1e-10).all()


def test_fixed_cap_forces_equal_weight(market):
    np.testing.assert_allclose(minimum_variance(market, max_asset_weight=.5).weights, [.5, .5], atol=1e-10)


def test_infeasible_request(market, optimization_request):
    with pytest.raises(RequestValidationError):
        optimize_portfolio(market, replace(optimization_request, max_asset_weight=.49))


def test_infeasible_target(market):
    with pytest.raises(OptimizationError, match="infeasible"):
        minimum_variance(market, target_return=.3)


def test_solver_failure(market):
    with patch("cvxpy.Problem.solve", side_effect=cp.error.SolverError("failed")):
        with pytest.raises(OptimizationError, match="solver"):
            minimum_variance(market)


def test_inaccurate_status_rejected(market):
    def inaccurate(problem, **kwargs):
        problem._status = cp.OPTIMAL_INACCURATE
    with patch("cvxpy.Problem.solve", inaccurate):
        with pytest.raises(OptimizationError, match="optimal_inaccurate"):
            minimum_variance(market)


def test_missing_solver(market):
    with pytest.raises(OptimizationError, match="solver"):
        minimum_variance(market, solver="NONEXISTENT")


def test_risk_setting_on_known_fixture(market, optimization_request):
    low = optimize_portfolio(market, replace(optimization_request, risk_level="low"))
    high = optimize_portfolio(market, replace(optimization_request, risk_level="high"))
    assert high.metrics.annual_volatility > low.metrics.annual_volatility


def test_amount_does_not_change_weights(market, optimization_request):
    a = optimize_portfolio(market, optimization_request)
    b = optimize_portfolio(market, replace(optimization_request, investment_amount=500_000))
    np.testing.assert_allclose(a.weights, b.weights)


def test_convention_mismatch(market, optimization_request):
    with pytest.raises(OptimizationError, match="Annualization"):
        optimize_portfolio(market, optimization_request, AnalysisConfig(trading_days_per_year=250))


def test_asset_mismatch(market, optimization_request):
    with pytest.raises(OptimizationError, match="order"):
        optimize_portfolio(market, replace(optimization_request, tickers=("B", "A")))


def test_frontier_feasibility_and_endpoint(market):
    frontier = build_frontier(market, points=6, max_asset_weight=.8)
    assert len(frontier.points) == 6
    assert frontier.maximum_feasible_return == pytest.approx(.18)
    returns = [point.metrics.expected_annual_return for point in frontier.points]
    assert np.diff(returns).min() >= -1e-7
    assert returns[-1] == pytest.approx(.18, abs=1e-7)
    for point in frontier.points:
        assert point.weights.sum() == pytest.approx(1)
        assert point.weights.max() <= .8 + 1e-10
        if point.target_return is not None:
            assert point.metrics.expected_annual_return >= point.target_return - 1e-7


def test_degenerate_frontier(market):
    same_returns = replace(market, expected_returns=pd.Series([.1, .1], index=["A", "B"]))
    frontier = build_frontier(same_returns)
    assert frontier.degenerate
    assert len(frontier.points) == 1


def test_single_asset_frontier(market):
    single = replace(market, expected_returns=market.expected_returns.iloc[:1], covariance=market.covariance.iloc[:1, :1])
    assert build_frontier(single).degenerate


@pytest.mark.parametrize("cap", [0, .4, float("nan"), True, 1.1])
def test_invalid_cap(market, cap):
    with pytest.raises(OptimizationError):
        minimum_variance(market, max_asset_weight=cap)


@pytest.mark.parametrize("points", [0, 1, 1.5, True])
def test_invalid_frontier_point_count(market, points):
    with pytest.raises(OptimizationError):
        build_frontier(market, points=points)


@pytest.mark.parametrize("values", [[-.1, 1.1], [.2, .2], [float("nan"), .5]])
def test_invalid_solver_weights_rejected(market, values):
    def invalid_solution(problem, **kwargs):
        problem._status = cp.OPTIMAL
        problem._value = 0
        # Inject directly to simulate a faulty solver without CVXPY's setter validation.
        problem.variables()[0].save_value(np.array(values))
    with patch("cvxpy.Problem.solve", invalid_solution):
        with pytest.raises(OptimizationError, match="invalid portfolio"):
            minimum_variance(market)


def test_zero_covariance_prefers_highest_return(market, optimization_request):
    zero_risk = replace(market, covariance=market.covariance * 0)
    result = optimize_portfolio(zero_risk, optimization_request)
    assert result.weights["B"] == pytest.approx(1, abs=1e-6)
    assert result.metrics.expected_sharpe_ratio is None


def test_cutoff_mismatch(market, optimization_request):
    with pytest.raises(OptimizationError, match="cutoff"):
        optimize_portfolio(replace(market, cutoff=date(2026, 1, 1)), optimization_request)


def test_fixed_weight_frontier(market):
    frontier = build_frontier(market, max_asset_weight=.5)
    assert frontier.degenerate
    assert frontier.maximum_feasible_return == pytest.approx(.15)


def test_target_solution_matches_known_weights(market):
    result = minimum_variance(market, target_return=.18)
    np.testing.assert_allclose(result.weights, [.2, .8], atol=1e-5)

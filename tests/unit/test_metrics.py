from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.errors import EstimationError
from portfolio_optimizer.evaluation import equal_weights, equal_weight_metrics, expected_metrics
from portfolio_optimizer.forecasting import MarketEstimates


@pytest.fixture
def estimates():
    return MarketEstimates(pd.Series([.1, .2], index=["A", "B"]),
                          pd.DataFrame([[.04, .01], [.01, .09]], index=["A", "B"], columns=["A", "B"]),
                          date(2025, 1, 1), 60)


def test_known_portfolio_metrics(estimates):
    metrics = expected_metrics(pd.Series([.25, .75], index=["A", "B"]), estimates, annual_risk_free_rate=.03)
    variance = .25**2 * .04 + .75**2 * .09 + 2 * .25 * .75 * .01
    assert metrics.expected_annual_return == pytest.approx(.175)
    assert metrics.annual_volatility == pytest.approx(np.sqrt(variance))
    assert metrics.expected_sharpe_ratio == pytest.approx((.175 - .03) / np.sqrt(variance))


def test_labels_control_weight_alignment(estimates):
    a = expected_metrics(pd.Series([.25, .75], index=["A", "B"]), estimates)
    b = expected_metrics(pd.Series([.75, .25], index=["B", "A"]), estimates)
    assert a == b


def test_equal_weight_reference(estimates):
    result = equal_weight_metrics(estimates)
    assert result.expected_annual_return == pytest.approx(.15)
    assert result.annual_volatility == pytest.approx(np.sqrt(.0375))
    assert equal_weights(("A",)).iloc[0] == 1


def test_zero_volatility_undefined_sharpe(estimates):
    estimates = replace(estimates, covariance=estimates.covariance * 0)
    result = equal_weight_metrics(estimates)
    assert result.annual_volatility == 0
    assert result.expected_sharpe_ratio is None
    assert "undefined" in result.warnings[0]


@pytest.mark.parametrize("weights", [
    pd.Series([.2, .2], index=["A", "B"]), pd.Series([-.1, 1.1], index=["A", "B"]),
    pd.Series([np.nan, .5], index=["A", "B"]), pd.Series([.5, .5], index=["A", "C"]),
    pd.Series([.5, .5], index=["A", "A"]), pd.Series([".5", ".5"], index=["A", "B"]),
])
def test_invalid_weights(estimates, weights):
    with pytest.raises(EstimationError):
        expected_metrics(weights, estimates)


@pytest.mark.parametrize("rate", [np.nan, np.inf, -1, True, "0.03"])
def test_invalid_risk_free_rate(estimates, rate):
    with pytest.raises(EstimationError):
        equal_weight_metrics(estimates, annual_risk_free_rate=rate)


def test_mutated_covariance_revalidated(estimates):
    estimates.covariance.iloc[0, 0] = -1
    with pytest.raises(EstimationError):
        equal_weight_metrics(estimates)

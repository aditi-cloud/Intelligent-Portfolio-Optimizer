from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.errors import EstimationError
from portfolio_optimizer.forecasting import MarketEstimates, estimate_covariance, estimate_market, estimate_returns


@pytest.fixture
def small_config():
    return AnalysisConfig(trading_days_per_year=10, min_observations=2)


@pytest.fixture
def daily_returns():
    return pd.DataFrame({"A": [-.1, .1], "B": [.1, -.1]}, index=pd.date_range("2025-01-01", periods=2))


def test_arithmetic_annualization(daily_returns, small_config):
    daily_returns["A"] = [.01, .03]
    result = estimate_returns(daily_returns, small_config)
    np.testing.assert_allclose(result.values, [.2, 0])
    assert list(result.index) == ["A", "B"]


def test_known_zero_shrinkage_covariance(daily_returns, small_config):
    result = estimate_covariance(daily_returns, small_config)
    np.testing.assert_allclose(result.annual_covariance.values, [[.1, -.1], [-.1, .1]], atol=1e-14)
    assert result.shrinkage == 0


def test_single_asset_population_variance(daily_returns, small_config):
    result = estimate_covariance(daily_returns[["A"]], small_config)
    assert result.annual_covariance.iloc[0, 0] == pytest.approx(.1)


def test_all_constant_assets_zero_covariance(daily_returns, small_config):
    daily_returns[:] = .02
    result = estimate_covariance(daily_returns, small_config)
    np.testing.assert_allclose(result.annual_covariance.values, 0, atol=1e-20)


def test_collinear_covariance_is_psd(small_config):
    rng = np.random.default_rng(42)
    a = rng.normal(0, .02, 100)
    returns = pd.DataFrame({"B": 2 * a, "A": a, "C": np.zeros(100)}, index=pd.date_range("2025-01-01", periods=100))
    result = estimate_covariance(returns, small_config)
    matrix = result.annual_covariance.values
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    assert list(result.annual_covariance.columns) == ["B", "A", "C"]
    assert 0 <= result.shrinkage <= 1


@pytest.mark.parametrize("mutation", [
    lambda f: f.iloc[:1], lambda f: f.iloc[::-1], lambda f: f.astype(str),
    lambda f: f.assign(A=np.nan), lambda f: f.assign(A=np.inf), lambda f: f.assign(A=-1),
    lambda f: f.set_axis([f.index[0]] * len(f)), lambda f: f.set_axis(["A", "A"], axis=1),
    lambda f: f.set_axis(range(len(f))), lambda f: f.astype(bool),
])
@pytest.mark.parametrize("estimator", [estimate_returns, estimate_covariance])
def test_invalid_returns_rejected(daily_returns, small_config, mutation, estimator):
    with pytest.raises(EstimationError):
        estimator(mutation(daily_returns), small_config)


def test_baseline_cutoff_and_units(price_dataset, data_request, small_config):
    prepared = prepare_data(price_dataset, data_request, small_config)
    result = estimate_market(prepared, data_request, small_config)
    np.testing.assert_allclose(result.expected_returns.values, [.4, .1])
    assert result.cutoff == date(2025, 1, 9)
    assert result.observations == 5
    assert result.trading_days_per_year == 10


def test_baseline_rejects_unimplemented_estimator(price_dataset, data_request, small_config):
    prepared = prepare_data(price_dataset, data_request, small_config)
    with pytest.raises(EstimationError, match="historical"):
        estimate_market(prepared, replace(data_request, estimator="xgboost"), small_config)


def test_estimates_reject_misaligned_covariance():
    with pytest.raises(EstimationError, match="order"):
        MarketEstimates(pd.Series([.1, .2], index=["A", "B"]),
                        pd.DataFrame(np.eye(2), index=["B", "A"], columns=["B", "A"]), date(2025, 1, 1), 60)


@pytest.mark.parametrize("matrix", [ [[1, 2], [2, 1]], [[1, .2], [.1, 1]], [[-1, 0], [0, 1]], [[np.nan, 0], [0, 1]] ])
def test_estimates_reject_invalid_covariance(matrix):
    with pytest.raises(EstimationError):
        MarketEstimates(pd.Series([.1, .2], index=["A", "B"]),
                        pd.DataFrame(matrix, index=["A", "B"], columns=["A", "B"]), date(2025, 1, 1), 60)


def test_annualization_scales_both_moments(daily_returns, small_config):
    daily_returns["A"] += .02
    doubled = replace(small_config, trading_days_per_year=20)
    np.testing.assert_allclose(estimate_returns(daily_returns, doubled), 2 * estimate_returns(daily_returns, small_config))
    np.testing.assert_allclose(estimate_covariance(daily_returns, doubled).annual_covariance,
                               2 * estimate_covariance(daily_returns, small_config).annual_covariance)


def test_baseline_rejects_asset_order_mismatch(price_dataset, data_request, small_config):
    prepared = prepare_data(price_dataset, data_request, small_config)
    with pytest.raises(EstimationError, match="order"):
        estimate_market(replace(prepared, returns=prepared.returns.iloc[:, ::-1]), data_request, small_config)


def test_baseline_rejects_future_training_data(price_dataset, data_request, small_config):
    prepared = prepare_data(price_dataset, data_request, small_config)
    with pytest.raises(EstimationError, match="window"):
        estimate_market(prepared, replace(data_request, end_date=date(2025, 1, 8)), small_config)


def test_estimates_copy_input_frames():
    mu = pd.Series([.1], index=["A"])
    covariance = pd.DataFrame([[.04]], index=["A"], columns=["A"])
    estimates = MarketEstimates(mu, covariance, date(2025, 1, 1), 60)
    mu.iloc[0] = .9
    covariance.iloc[0, 0] = .9
    assert estimates.expected_returns.iloc[0] == .1
    assert estimates.covariance.iloc[0, 0] == .04


@pytest.mark.parametrize("updates", [
    {"expected_returns": None}, {"covariance": None}, {"observations": 1},
    {"trading_days_per_year": True}, {"shrinkage": np.nan}, {"shrinkage": 1.1},
    {"return_method": ""}, {"cutoff": "2025-01-01"},
])
def test_estimate_contract_metadata(updates):
    estimates = MarketEstimates(pd.Series([.1], index=["A"]),
                                pd.DataFrame([[.04]], index=["A"], columns=["A"]), date(2025, 1, 1), 60)
    with pytest.raises(EstimationError):
        replace(estimates, **updates)


def test_machine_epsilon_shrinkage_normalized():
    returns = pd.DataFrame({"A": [-.1, .1], "B": [0, -.1]}, index=pd.bdate_range("2025-01-01", periods=2))
    result = estimate_covariance(returns, AnalysisConfig(min_observations=2))
    assert 0 <= result.shrinkage <= 1

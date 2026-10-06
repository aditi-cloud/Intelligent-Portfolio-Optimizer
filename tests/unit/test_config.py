from dataclasses import replace

import pytest

from portfolio_optimizer import AnalysisConfig, RiskLevel


def test_default_risk_mapping_is_ordered():
    config = AnalysisConfig()
    assert config.risk_aversion(RiskLevel.LOW) > config.risk_aversion(RiskLevel.MEDIUM)
    assert config.risk_aversion(RiskLevel.MEDIUM) > config.risk_aversion(RiskLevel.HIGH)


@pytest.mark.parametrize("updates", [
    {"trading_days_per_year": 0}, {"trading_days_per_year": True},
    {"min_observations": 1}, {"random_seed": -1}, {"random_seed": 1.5},
    {"annual_risk_free_rate": float("nan")}, {"annual_risk_free_rate": -1},
    {"low_risk_aversion": 0}, {"high_risk_aversion": float("inf")},
    {"low_risk_aversion": 2}, {"medium_risk_aversion": 1},
])
def test_invalid_configuration(updates):
    with pytest.raises(ValueError):
        replace(AnalysisConfig(), **updates)


def test_negative_risk_free_rate_above_minus_one_is_valid():
    assert AnalysisConfig(annual_risk_free_rate=-0.01).annual_risk_free_rate == -0.01

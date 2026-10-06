import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.evaluation.realized import realized_metrics
from portfolio_optimizer.errors import BacktestError


def test_hand_calculated_metrics():
    returns = pd.Series([.1, -.2, .1], index=pd.bdate_range("2025-01-01", periods=3))
    result = realized_metrics(returns, AnalysisConfig(trading_days_per_year=3))
    assert result.total_return == pytest.approx(1.1 * .8 * 1.1 - 1)
    assert result.annualized_geometric_return == pytest.approx(result.total_return)
    assert result.annualized_arithmetic_return == pytest.approx(0)
    assert result.annual_volatility == pytest.approx(np.std(returns, ddof=1) * np.sqrt(3))
    assert result.realized_sharpe_ratio == pytest.approx(0)
    assert result.max_drawdown == pytest.approx(-.2)


def test_entry_loss_is_in_drawdown():
    returns = pd.Series([-.1, 0], index=pd.bdate_range("2025-01-01", periods=2))
    assert realized_metrics(returns).max_drawdown == pytest.approx(-.1)


def test_zero_volatility_and_risk_free_rate():
    returns = pd.Series([0, 0], index=pd.bdate_range("2025-01-01", periods=2))
    result = realized_metrics(returns, AnalysisConfig(annual_risk_free_rate=.05))
    assert result.realized_sharpe_ratio is None
    assert result.warnings


def test_compounded_daily_risk_free_rate():
    returns = pd.Series([.01, .02], index=pd.bdate_range("2025-01-01", periods=2))
    config = AnalysisConfig(trading_days_per_year=2, annual_risk_free_rate=.04)
    result = realized_metrics(returns, config)
    daily_rf = np.sqrt(1.04) - 1
    assert result.realized_sharpe_ratio == pytest.approx((.015 - daily_rf) / returns.std() * np.sqrt(2))


@pytest.mark.parametrize("values", [[.01], [-1, .1], [-2, .1]])
def test_invalid_realized_returns(values):
    with pytest.raises(BacktestError):
        realized_metrics(pd.Series(values, index=pd.bdate_range("2025-01-01", periods=len(values))))

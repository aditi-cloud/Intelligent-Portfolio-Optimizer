from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AnalysisConfig, _finite_number
from ..errors import BacktestError
from ..forecasting.validation import numeric_values


@dataclass(frozen=True)
class RealizedMetrics:
    total_return: float
    annualized_geometric_return: float
    annualized_arithmetic_return: float
    annual_volatility: float
    realized_sharpe_ratio: float | None
    max_drawdown: float
    observations: int
    warnings: tuple[str, ...] = ()


def realized_metrics(net_returns: pd.Series, config: AnalysisConfig | None = None) -> RealizedMetrics:
    """Summarize sequential net session returns, including entry fees.

    Geometric annualization uses observation count, not calendar days. The caller
    must supply contiguous session returns; missing periods cannot be inferred.
    Drawdown includes starting wealth before the first recorded return.
    """
    config = config or AnalysisConfig()
    if not isinstance(net_returns, pd.Series) or len(net_returns) < 2:
        raise BacktestError("Realized metrics need at least two session returns")
    if not isinstance(net_returns.index, pd.DatetimeIndex) or net_returns.index.hasnans or net_returns.index.has_duplicates or not net_returns.index.is_monotonic_increasing:
        raise BacktestError("Realized returns require sorted unique session dates")
    values = numeric_values(net_returns)
    if (values <= -1).any():
        raise BacktestError("Net returns must be greater than -1")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            logs = np.log1p(values)
            wealth = np.exp(np.cumsum(logs))
            total = np.expm1(logs.sum())
            geometric = np.expm1(logs.mean() * config.trading_days_per_year)
            arithmetic = values.mean() * config.trading_days_per_year
            daily_std = values.std(ddof=1)
            volatility = daily_std * np.sqrt(config.trading_days_per_year)
            daily_rf = np.expm1(np.log1p(config.annual_risk_free_rate) / config.trading_days_per_year)
            sharpe = None if daily_std == 0 else float((values - daily_rf).mean() / daily_std * np.sqrt(config.trading_days_per_year))
            curve = np.r_[1.0, wealth]
            drawdown = float((curve / np.maximum.accumulate(curve) - 1).min())
    except FloatingPointError as exc:
        raise BacktestError("Realized metric calculation overflowed") from exc
    outputs = [total, geometric, arithmetic, volatility, drawdown]
    if not all(_finite_number(float(value)) for value in outputs) or (sharpe is not None and not np.isfinite(sharpe)) or (wealth <= 0).any():
        raise BacktestError("Realized metrics are non-finite")
    warnings = ("Realized Sharpe is undefined for zero return volatility",) if sharpe is None else ()
    return RealizedMetrics(float(total), float(geometric), float(arithmetic), float(volatility), sharpe,
                           drawdown, len(values), warnings)

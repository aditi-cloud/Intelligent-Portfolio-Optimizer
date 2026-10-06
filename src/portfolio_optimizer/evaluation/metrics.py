from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from ..config import _finite_number
from ..errors import EstimationError
from ..forecasting.contracts import MarketEstimates, covariance_tolerance, validate_estimates
from ..forecasting.validation import numeric_values


@dataclass(frozen=True)
class PortfolioMetrics:
    expected_annual_return: float
    annual_volatility: float
    expected_sharpe_ratio: float | None
    annual_risk_free_rate: float
    warnings: tuple[str, ...] = ()


def equal_weights(tickers: tuple[str, ...]) -> pd.Series:
    if not tickers or len(set(tickers)) != len(tickers) or any(not isinstance(t, str) or not t for t in tickers):
        raise EstimationError("Equal weights require unique nonempty tickers")
    return pd.Series(1 / len(tickers), index=list(tickers), name="weight")


def expected_metrics(weights: pd.Series, estimates: MarketEstimates, *,
                     annual_risk_free_rate: float = 0.0) -> PortfolioMetrics:
    """Expected annual metrics for a fully invested, long-only portfolio.

    Explicitly labeled expected: these do not establish realized performance.
    Asset labels may arrive in a different order but must match exactly.
    """
    validate_estimates(estimates)
    if not isinstance(weights, pd.Series) or weights.index.has_duplicates or set(weights.index) != set(estimates.tickers):
        raise EstimationError("Weights must include exactly the estimated assets")
    values = numeric_values(weights.reindex(estimates.tickers))
    if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, rtol=0, atol=1e-8):
        raise EstimationError("Weights must be long-only and sum to one")
    if not _finite_number(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise EstimationError("Annual risk-free rate must be finite and greater than -1")
    matrix = estimates.covariance.to_numpy(dtype=float)
    expected = float(values @ estimates.expected_returns.to_numpy(dtype=float))
    variance = float(values @ matrix @ values)
    if not np.isfinite(expected) or not np.isfinite(variance):
        raise EstimationError("Portfolio metric calculation overflowed")
    if variance < -covariance_tolerance(matrix):
        raise EstimationError("Portfolio variance is negative")
    volatility = sqrt(max(variance, 0))
    if volatility == 0:
        return PortfolioMetrics(expected, volatility, None, float(annual_risk_free_rate),
                                ("Sharpe ratio is undefined for zero estimated volatility",))
    sharpe = (expected - annual_risk_free_rate) / volatility
    if not np.isfinite(sharpe):
        raise EstimationError("Sharpe ratio calculation overflowed")
    return PortfolioMetrics(expected, volatility, float(sharpe), float(annual_risk_free_rate))


def equal_weight_metrics(estimates: MarketEstimates, *, annual_risk_free_rate: float = 0.0) -> PortfolioMetrics:
    return expected_metrics(equal_weights(estimates.tickers), estimates,
                            annual_risk_free_rate=annual_risk_free_rate)

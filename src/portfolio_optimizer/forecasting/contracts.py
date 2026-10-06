from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..errors import EstimationError
from .validation import numeric_values


@dataclass(frozen=True)
class MarketEstimates:
    """Annual arithmetic mean returns and covariance in identical ticker order.

    These are annualized daily moments, not compounded wealth predictions.
    Frames are copied at construction; consumers revalidate mutable pandas data.
    """

    expected_returns: pd.Series
    covariance: pd.DataFrame
    cutoff: date
    observations: int
    trading_days_per_year: int = 252
    return_method: str = "historical_arithmetic_mean"
    covariance_method: str = "ledoit_wolf"
    shrinkage: float | None = None

    def __post_init__(self):
        if not isinstance(self.expected_returns, pd.Series) or not isinstance(self.covariance, pd.DataFrame):
            raise EstimationError("Estimates require a return Series and covariance DataFrame")
        object.__setattr__(self, "expected_returns", self.expected_returns.copy())
        object.__setattr__(self, "covariance", self.covariance.copy())
        validate_estimates(self)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.expected_returns.index)


def validate_estimates(estimates: MarketEstimates) -> None:
    mu, covariance = estimates.expected_returns, estimates.covariance
    if not isinstance(mu, pd.Series) or mu.empty or not isinstance(covariance, pd.DataFrame):
        raise EstimationError("Estimates require a return Series and covariance DataFrame")
    if mu.index.has_duplicates or any(not isinstance(t, str) or not t for t in mu.index):
        raise EstimationError("Estimates require unique ticker labels")
    if list(mu.index) != list(covariance.index) or list(mu.index) != list(covariance.columns):
        raise EstimationError("Return and covariance asset order must match exactly")
    numeric_values(mu)
    matrix = numeric_values(covariance)
    tolerance = covariance_tolerance(matrix)
    if not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
        raise EstimationError("Covariance must be symmetric")
    if (np.diag(matrix) < 0).any():
        raise EstimationError("Covariance variances cannot be negative")
    if np.linalg.eigvalsh((matrix + matrix.T) / 2).min() < -tolerance:
        raise EstimationError("Covariance must be positive semidefinite")
    if type(estimates.cutoff) is not date:
        raise EstimationError("Estimation cutoff must be a date")
    if type(estimates.observations) is not int or estimates.observations < 2:
        raise EstimationError("Estimates require at least two observations")
    if type(estimates.trading_days_per_year) is not int or estimates.trading_days_per_year < 1:
        raise EstimationError("Trading days per year must be a positive integer")
    if any(not isinstance(method, str) or not method.strip()
           for method in (estimates.return_method, estimates.covariance_method)):
        raise EstimationError("Estimation methods must be recorded")
    if estimates.shrinkage is not None:
        if isinstance(estimates.shrinkage, bool) or not isinstance(estimates.shrinkage, (int, float)) or not np.isfinite(estimates.shrinkage) or not 0 <= estimates.shrinkage <= 1:
            raise EstimationError("Shrinkage must be in [0, 1]")


def covariance_tolerance(matrix: np.ndarray) -> float:
    return 1e-10 * max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..config import AnalysisConfig
from ..errors import EstimationError
from .validation import validate_returns


@dataclass(frozen=True)
class CovarianceEstimate:
    annual_covariance: pd.DataFrame
    shrinkage: float


def estimate_covariance(returns: pd.DataFrame, config: AnalysisConfig | None = None) -> CovarianceEstimate:
    """Centered Ledoit-Wolf daily covariance, scaled by trading days/year.

    Uses the estimator's population normalization, not sample ddof=1. Shrinkage
    can assign positive estimated variance to a constant asset in a mixed basket.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
    """
    config = config or AnalysisConfig()
    frame = validate_returns(returns, config)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            model = LedoitWolf(assume_centered=False, store_precision=False).fit(frame.to_numpy())
            annual = model.covariance_ * config.trading_days_per_year
    except (ValueError, FloatingPointError) as exc:
        raise EstimationError("Covariance estimation failed") from exc
    if not np.isfinite(annual).all():
        raise EstimationError("Covariance contains non-finite values")
    shrinkage = float(model.shrinkage_)
    if not np.isfinite(shrinkage) or not -1e-12 <= shrinkage <= 1 + 1e-12:
        raise EstimationError("Covariance estimator returned an invalid shrinkage coefficient")
    # Tiny-window fits can return a negative coefficient at machine epsilon.
    # Normalize metadata only within numerical tolerance; reject genuine errors.
    shrinkage = min(1.0, max(0.0, shrinkage))
    return CovarianceEstimate(pd.DataFrame(annual, index=frame.columns, columns=frame.columns),
                              shrinkage)

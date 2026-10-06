import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..errors import EstimationError
from .validation import validate_returns


def estimate_returns(returns: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.Series:
    """Annualize arithmetic daily simple-return means (no compounding)."""
    config = config or AnalysisConfig()
    frame = validate_returns(returns, config)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = frame.mean() * config.trading_days_per_year
    if not np.isfinite(expected.to_numpy()).all():
        raise EstimationError("Expected returns overflowed")
    expected.name = "expected_annual_return"
    return expected

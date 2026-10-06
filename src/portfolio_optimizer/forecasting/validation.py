import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..errors import EstimationError


def numeric_values(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    dtypes = frame.dtypes if isinstance(frame, pd.DataFrame) else [frame.dtype]
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
           or pd.api.types.is_complex_dtype(dtype) for dtype in dtypes):
        raise EstimationError("Values must be real numeric data")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise EstimationError("Values must be finite, without missing observations")
    return values


def validate_returns(returns: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise EstimationError("Returns must be a nonempty DataFrame")
    if returns.columns.has_duplicates or any(not isinstance(ticker, str) or not ticker for ticker in returns.columns):
        raise EstimationError("Returns require unique ticker labels")
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.hasnans:
        raise EstimationError("Returns require a valid date index")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise EstimationError("Return dates must be sorted and unique")
    if not returns.index.equals(returns.index.normalize()):
        raise EstimationError("Returns require daily date labels")
    if len(returns) < config.min_observations:
        raise EstimationError(f"Need at least {config.min_observations} return observations")
    values = numeric_values(returns)
    if (values <= -1).any():
        raise EstimationError("Adjusted-equity simple returns must be greater than -1")
    return returns.copy().astype(float)

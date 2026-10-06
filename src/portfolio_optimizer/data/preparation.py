from datetime import datetime

import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..contracts import PortfolioRequest
from ..errors import DataQualityError
from ..validation import validate_request
from .contracts import DataQualityReport, PreparedData, PriceDataset


def validate_prices(dataset: PriceDataset, request: PortfolioRequest) -> pd.DataFrame:
    """Validate provenance and prices, copy, sort, and preserve requested order.

    Dates represent exchange-local daily labels, not UTC instants. Unexpected
    dates and assets are rejected rather than silently changing the dataset.
    Missing prices are allowed; invalid observed prices are not.
    """
    request = validate_request(request)
    frame = dataset.prices
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataQualityError("No price data available")
    if dataset.adjustment_policy != "adjusted_close":
        raise DataQualityError("Only adjusted closing prices are supported")
    if not dataset.provider or not isinstance(dataset.retrieved_at, datetime) or dataset.retrieved_at.utcoffset() is None:
        raise DataQualityError("Price provenance requires provider and timezone-aware retrieval time")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise DataQualityError("Prices require a valid date index")
    index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    if not index.equals(index.normalize()) or index.has_duplicates:
        raise DataQualityError("Prices require unique daily dates")
    if any(d < request.start_date or d > request.end_date for d in index.date):
        raise DataQualityError("Price dates fall outside the requested window")
    if frame.columns.has_duplicates or set(frame.columns) != set(request.tickers):
        raise DataQualityError("Price columns must match all requested tickers")
    if any(dataset.currencies.get(ticker) != request.currency for ticker in request.tickers):
        raise DataQualityError("Missing or mixed asset currencies; FX conversion is not supported")
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
           or pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
        raise DataQualityError("Prices must contain real numeric values")
    frame = frame.loc[:, list(request.tickers)].copy().astype(float)
    frame.index = index
    frame = frame.sort_index()
    values = frame.to_numpy()
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise DataQualityError("Observed prices must be positive and finite")
    if frame.isna().all().any():
        raise DataQualityError("A requested ticker has no price history")
    return frame


def prepare_data(dataset: PriceDataset, request: PortfolioRequest,
                 config: AnalysisConfig | None = None) -> PreparedData:
    config = config or AnalysisConfig()
    prices = validate_prices(dataset, request)
    # Compute BEFORE removing incomplete rows: otherwise a gap becomes a
    # multi-session return incorrectly presented as a daily observation.
    daily = prices.pct_change(fill_method=None)
    aligned = daily.dropna(how="any")
    if not np.isfinite(aligned.to_numpy()).all():
        raise DataQualityError("Daily returns contain non-finite values")
    if len(aligned) < config.min_observations:
        raise DataQualityError(
            f"Need at least {config.min_observations} aligned returns; found {len(aligned)}"
        )
    excluded = daily.iloc[1:].index[daily.iloc[1:].isna().any(axis=1)]
    quality = DataQualityReport(
        input_price_rows=len(prices), complete_price_rows=len(prices.dropna()),
        missing_prices_by_ticker={ticker: int(prices[ticker].isna().sum()) for ticker in request.tickers},
        excluded_return_dates=tuple(day.date().isoformat() for day in excluded),
        observations=len(aligned),
    )
    return PreparedData(prices=prices, returns=aligned, quality=quality, source=dataset)

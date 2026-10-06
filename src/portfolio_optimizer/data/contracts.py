from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class PriceDataset:
    """Adjusted daily closes. Treat frames as read-only; preparation copies them."""

    prices: pd.DataFrame
    currencies: Mapping[str, str]
    provider: str
    retrieved_at: datetime
    adjustment_policy: str = "adjusted_close"


@dataclass(frozen=True)
class DataQualityReport:
    input_price_rows: int
    complete_price_rows: int
    missing_prices_by_ticker: Mapping[str, int]
    excluded_return_dates: tuple[str, ...]
    observations: int


@dataclass(frozen=True)
class PreparedData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    quality: DataQualityReport
    source: PriceDataset

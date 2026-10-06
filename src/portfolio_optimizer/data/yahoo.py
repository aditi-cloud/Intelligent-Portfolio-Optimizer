"""Yahoo adapter: explicitly request adjusted prices and inclusive end dates.

API reference: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
"""

from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd

from ..contracts import PortfolioRequest
from ..errors import DataQualityError, ProviderError
from ..validation import validate_request
from .contracts import PriceDataset
from .preparation import validate_prices


def adjusted_close_frame(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Handle flat single-symbol and either orientation of MultiIndex output."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ProviderError("Yahoo returned no price data")
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [level for level in range(raw.columns.nlevels)
                  if "Close" in raw.columns.get_level_values(level)]
        if len(levels) != 1:
            raise ProviderError("Yahoo response has no unambiguous adjusted Close field")
        frame = raw.xs("Close", axis=1, level=levels[0])
    elif len(tickers) == 1 and "Close" in raw.columns:
        frame = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise ProviderError("Unexpected Yahoo price layout")
    if frame.columns.has_duplicates or any(ticker not in frame.columns for ticker in tickers):
        raise ProviderError("Yahoo response is missing a requested stock")
    return frame.loc[:, list(tickers)].copy()


class YahooPriceProvider:
    name = "yahoo"
    adjustment_policy = "adjusted_close"

    def __init__(self, *, download: Callable | None = None,
                 currency_lookup: Callable[[str], str] | None = None,
                 version: str | None = None, clock: Callable | None = None):
        import yfinance as yf
        self.version = version or yf.__version__
        self._download = download or yf.download
        self._currency_lookup = currency_lookup or (lambda ticker: yf.Ticker(ticker).get_info().get("currency"))
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def fetch_prices(self, request: PortfolioRequest) -> PriceDataset:
        request = validate_request(request)
        try:
            raw = self._download(
                tickers=list(request.tickers), start=request.start_date.isoformat(),
                end=(request.end_date + timedelta(days=1)).isoformat(),
                interval="1d", auto_adjust=True, back_adjust=False,
                group_by="column", multi_level_index=True, progress=False,
                threads=False, timeout=15, keepna=True,
            )
            prices = adjusted_close_frame(raw, request.tickers)
            currencies = {ticker: self._currency_lookup(ticker) for ticker in request.tickers}
            dataset = PriceDataset(prices, currencies, self.name, self._clock(), self.adjustment_policy)
            canonical = validate_prices(dataset, request)
            return PriceDataset(canonical, currencies, self.name, dataset.retrieved_at, self.adjustment_policy)
        except ProviderError:
            raise
        except DataQualityError as exc:
            raise ProviderError(f"Yahoo price data failed validation: {exc}") from exc
        except Exception as exc:
            raise ProviderError("Yahoo price or currency fetch failed") from exc

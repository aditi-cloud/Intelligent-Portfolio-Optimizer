"""Offline provider for recorded or synthetic adjusted-price datasets."""

from dataclasses import replace

from ..contracts import PortfolioRequest
from ..errors import ProviderError
from ..validation import validate_request
from .contracts import PriceDataset


class InMemoryPriceProvider:
    version = "1"

    def __init__(self, dataset: PriceDataset):
        self.dataset = replace(dataset, prices=dataset.prices.copy(), currencies=dict(dataset.currencies))
        self.name = dataset.provider
        self.adjustment_policy = dataset.adjustment_policy

    def fetch_prices(self, request: PortfolioRequest) -> PriceDataset:
        request = validate_request(request)
        if any(ticker not in self.dataset.prices for ticker in request.tickers):
            raise ProviderError("Offline dataset is missing a requested ticker")
        frame = self.dataset.prices.loc[str(request.start_date):str(request.end_date), list(request.tickers)].copy()
        if frame.empty:
            raise ProviderError("Offline dataset has no prices in the requested window")
        return replace(self.dataset, prices=frame,
                       currencies={ticker: self.dataset.currencies.get(ticker) for ticker in request.tickers})

from typing import Protocol

from ..contracts import PortfolioRequest
from .contracts import PriceDataset


class PriceProvider(Protocol):
    name: str
    version: str
    adjustment_policy: str

    def fetch_prices(self, request: PortfolioRequest) -> PriceDataset:
        """Fetch the requested inclusive date range or raise ProviderError."""
        ...

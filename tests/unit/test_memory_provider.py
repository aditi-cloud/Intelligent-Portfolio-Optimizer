from dataclasses import replace
from datetime import date

import pytest

from portfolio_optimizer.data.memory import InMemoryPriceProvider
from portfolio_optimizer.errors import ProviderError


def test_date_and_asset_subset_is_copied(price_dataset, data_request):
    provider = InMemoryPriceProvider(price_dataset)
    request = replace(data_request, tickers=("TCS.NS",), start_date=date(2025, 1, 3), end_date=date(2025, 1, 7))
    data = provider.fetch_prices(request)
    assert list(data.prices.columns) == ["TCS.NS"]
    assert len(data.prices) == 3
    data.prices.iloc[0, 0] = 999
    assert provider.fetch_prices(request).prices.iloc[0, 0] == 210
    assert price_dataset.prices.iloc[1, 1] == 210


def test_missing_ticker_fails(price_dataset, data_request):
    with pytest.raises(ProviderError, match="missing"):
        InMemoryPriceProvider(price_dataset).fetch_prices(replace(data_request, tickers=("UNKNOWN",)))


def test_empty_window_fails(price_dataset, data_request):
    with pytest.raises(ProviderError, match="window"):
        InMemoryPriceProvider(price_dataset).fetch_prices(replace(data_request, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)))

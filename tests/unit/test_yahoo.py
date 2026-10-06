from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from portfolio_optimizer.data.yahoo import YahooPriceProvider, adjusted_close_frame
from portfolio_optimizer.errors import ProviderError


@pytest.mark.parametrize("reverse", [False, True])
def test_multiindex_layouts(price_dataset, data_request, reverse):
    raw = pd.concat({"Close": price_dataset.prices, "Open": price_dataset.prices * .9}, axis=1)
    if reverse:
        raw = raw.swaplevel(axis=1)
    pd.testing.assert_frame_equal(adjusted_close_frame(raw, data_request.tickers), price_dataset.prices)


def test_flat_single_ticker(price_dataset):
    frame = price_dataset.prices[["TCS.NS"]].rename(columns={"TCS.NS": "Close"})
    assert list(adjusted_close_frame(frame, ("TCS.NS",)).columns) == ["TCS.NS"]


def test_adapter_parameters_and_metadata(price_dataset, data_request):
    download = Mock(return_value=pd.concat({"Close": price_dataset.prices}, axis=1))
    lookup = Mock(return_value="INR")
    provider = YahooPriceProvider(download=download, currency_lookup=lookup, clock=lambda: price_dataset.retrieved_at)
    result = provider.fetch_prices(data_request)
    kwargs = download.call_args.kwargs
    assert kwargs["end"] == (data_request.end_date + timedelta(days=1)).isoformat()
    assert kwargs["auto_adjust"] is True
    assert kwargs["keepna"] is True
    assert result.provider == "yahoo"
    assert lookup.call_count == 2
    pd.testing.assert_frame_equal(result.prices, price_dataset.prices)


@pytest.mark.parametrize("raw", [None, pd.DataFrame(), pd.DataFrame({"Open": [10]})])
def test_empty_or_unexpected_response(raw, data_request):
    with pytest.raises(ProviderError):
        adjusted_close_frame(raw, data_request.tickers)


def test_missing_ticker(price_dataset, data_request):
    raw = pd.concat({"Close": price_dataset.prices[["TCS.NS"]]}, axis=1)
    with pytest.raises(ProviderError, match="missing"):
        adjusted_close_frame(raw, data_request.tickers)


def test_timeout_wrapped(data_request):
    provider = YahooPriceProvider(download=Mock(side_effect=TimeoutError), currency_lookup=Mock())
    with pytest.raises(ProviderError, match="fetch failed"):
        provider.fetch_prices(data_request)


def test_currency_not_inferred_from_symbol(price_dataset, data_request):
    provider = YahooPriceProvider(download=Mock(return_value=pd.concat({"Close": price_dataset.prices}, axis=1)),
                                  currency_lookup=Mock(return_value=None))
    with pytest.raises(ProviderError, match="currency metadata"):
        provider.fetch_prices(data_request)


def test_default_currency_lookup_uses_history_metadata(monkeypatch):
    ticker = Mock()
    ticker.get_history_metadata.return_value = {"currency": "INR"}
    factory = Mock(return_value=ticker)
    monkeypatch.setattr("yfinance.Ticker", factory)
    provider = YahooPriceProvider()
    assert provider._currency_lookup("TCS.NS") == "INR"
    ticker.get_info.assert_not_called()

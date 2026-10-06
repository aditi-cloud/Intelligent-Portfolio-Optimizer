from dataclasses import replace
from datetime import timedelta
import json
from unittest.mock import Mock

import pandas as pd
import numpy as np
import pytest

from portfolio_optimizer.data.cache import CachedPriceProvider
from portfolio_optimizer.errors import ProviderError


@pytest.fixture
def fake_provider(price_dataset):
    provider = Mock()
    provider.name = "fixture"
    provider.version = "1"
    provider.adjustment_policy = "adjusted_close"
    provider.fetch_prices.return_value = price_dataset
    return provider


def test_cache_hit_skips_provider(tmp_path, fake_provider, price_dataset, data_request):
    cache = CachedPriceProvider(fake_provider, tmp_path, clock=lambda: price_dataset.retrieved_at)
    first = cache.fetch_prices(data_request)
    second = cache.fetch_prices(data_request)
    assert fake_provider.fetch_prices.call_count == 1
    pd.testing.assert_frame_equal(first.prices, second.prices, check_names=False)
    assert second.retrieved_at == first.retrieved_at


def test_expired_cache_refetches(tmp_path, fake_provider, price_dataset, data_request):
    now = [price_dataset.retrieved_at]
    cache = CachedPriceProvider(fake_provider, tmp_path, ttl=timedelta(hours=1), clock=lambda: now[0])
    cache.fetch_prices(data_request)
    now[0] += timedelta(hours=1)
    cache.fetch_prices(data_request)
    assert fake_provider.fetch_prices.call_count == 2


@pytest.mark.parametrize("content", ["not json", "{}", '[]', '{"schema": 99}', '{"schema": 1, "key": "wrong"}'])
def test_corrupt_cache_refetches(tmp_path, fake_provider, price_dataset, data_request, content):
    cache = CachedPriceProvider(fake_provider, tmp_path, clock=lambda: price_dataset.retrieved_at)
    (tmp_path / f"{cache.cache_key(data_request)}.json").write_text(content)
    cache.fetch_prices(data_request)
    assert fake_provider.fetch_prices.call_count == 1


def test_key_uses_data_inputs_only(tmp_path, fake_provider, data_request):
    cache = CachedPriceProvider(fake_provider, tmp_path)
    key = cache.cache_key(data_request)
    assert key == cache.cache_key(replace(data_request, investment_amount=200_000, risk_level="high"))
    assert key != cache.cache_key(replace(data_request, currency="USD"))
    assert key != cache.cache_key(replace(data_request, end_date=data_request.end_date + timedelta(days=1)))
    fake_provider.version = "2"
    assert key != CachedPriceProvider(fake_provider, tmp_path).cache_key(data_request)
    fake_provider.adjustment_policy = "raw_close"
    assert key != CachedPriceProvider(fake_provider, tmp_path).cache_key(data_request)


def test_provider_failure_propagates_without_cache(tmp_path, fake_provider, data_request):
    fake_provider.fetch_prices.side_effect = ProviderError("offline")
    with pytest.raises(ProviderError, match="offline"):
        CachedPriceProvider(fake_provider, tmp_path).fetch_prices(data_request)
    assert not list(tmp_path.glob("*.json"))


def test_missing_prices_roundtrip(tmp_path, fake_provider, price_dataset, data_request):
    prices = price_dataset.prices.copy()
    prices.iloc[2, 0] = np.nan
    fake_provider.fetch_prices.return_value = replace(price_dataset, prices=prices)
    cache = CachedPriceProvider(fake_provider, tmp_path, clock=lambda: price_dataset.retrieved_at)
    cache.fetch_prices(data_request)
    restored = cache.fetch_prices(data_request)
    pd.testing.assert_frame_equal(prices, restored.prices)
    assert fake_provider.fetch_prices.call_count == 1


@pytest.mark.parametrize("mutation", [
    lambda document: document.update(currencies={}),
    lambda document: document["values"][0].__setitem__(0, -100),
    lambda document: document.update(provider="other"),
    lambda document: document.update(retrieved_at="2025-01-10T00:00:00"),
    lambda document: document.update(cached_at="2099-01-10T00:00:00+00:00"),
])
def test_invalid_cached_data_refetched(tmp_path, fake_provider, price_dataset, data_request, mutation):
    cache = CachedPriceProvider(fake_provider, tmp_path, clock=lambda: price_dataset.retrieved_at)
    cache.fetch_prices(data_request)
    path = tmp_path / f"{cache.cache_key(data_request)}.json"
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document))
    cache.fetch_prices(data_request)
    assert fake_provider.fetch_prices.call_count == 2


def test_cache_hit_is_not_shared_mutable_frame(tmp_path, fake_provider, price_dataset, data_request):
    cache = CachedPriceProvider(fake_provider, tmp_path, clock=lambda: price_dataset.retrieved_at)
    first = cache.fetch_prices(data_request)
    first.prices.iloc[0, 0] = 999
    assert cache.fetch_prices(data_request).prices.iloc[0, 0] == 100

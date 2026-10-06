from unittest.mock import Mock

import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.data.cache import CachedPriceProvider
from portfolio_optimizer.data.yahoo import YahooPriceProvider


@pytest.mark.integration
def test_yahoo_cache_preparation_offline(tmp_path, price_dataset, data_request):
    download = Mock(return_value=pd.concat({"Close": price_dataset.prices}, axis=1))
    yahoo = YahooPriceProvider(download=download, currency_lookup=lambda ticker: "INR",
                               clock=lambda: price_dataset.retrieved_at)
    provider = CachedPriceProvider(yahoo, tmp_path, clock=lambda: price_dataset.retrieved_at)
    first = prepare_data(provider.fetch_prices(data_request), data_request, AnalysisConfig(min_observations=2))
    second = prepare_data(provider.fetch_prices(data_request), data_request, AnalysisConfig(min_observations=2))
    pd.testing.assert_frame_equal(first.returns, second.returns, check_names=False)
    assert first.quality.observations == 5
    assert first.source.provider == "yahoo"
    download.assert_called_once()

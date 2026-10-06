from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.errors import DataQualityError


def test_exact_daily_returns_and_order(price_dataset, data_request):
    prepared = prepare_data(price_dataset, data_request, AnalysisConfig(min_observations=2))
    np.testing.assert_allclose(prepared.returns.values,
                               [[.1, .05], [-.1, 0], [.1, -.1], [0, .1], [.1, 0]], atol=1e-14)
    assert tuple(prepared.returns.columns) == data_request.tickers
    assert prepared.quality.observations == 5
    assert prepared.quality.excluded_return_dates == ()


def test_missing_price_does_not_bridge_gap(price_dataset, data_request):
    prices = price_dataset.prices.copy()
    prices.iloc[2, 0] = np.nan
    result = prepare_data(replace(price_dataset, prices=prices), data_request, AnalysisConfig(min_observations=2))
    assert result.quality.excluded_return_dates == ("2025-01-06", "2025-01-07")
    assert result.quality.missing_prices_by_ticker["RELIANCE.NS"] == 1
    assert len(result.returns) == 3
    assert np.isnan(prices.iloc[2, 0])


def test_sorts_without_mutating_source(price_dataset, data_request):
    source = replace(price_dataset, prices=price_dataset.prices.iloc[::-1, ::-1])
    result = prepare_data(source, data_request, AnalysisConfig(min_observations=2))
    assert result.prices.index.is_monotonic_increasing
    assert not source.prices.index.is_monotonic_increasing
    assert tuple(result.prices.columns) == data_request.tickers


@pytest.mark.parametrize("value", [0, -1, float("inf"), -float("inf")])
def test_invalid_prices_rejected(price_dataset, data_request, value):
    prices = price_dataset.prices.copy()
    prices.iloc[1, 0] = value
    with pytest.raises(DataQualityError, match="positive and finite"):
        prepare_data(replace(price_dataset, prices=prices), data_request)


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.iloc[:0],
    lambda frame: frame.drop(columns="TCS.NS"),
    lambda frame: pd.concat([frame, frame.iloc[[0]]]),
    lambda frame: frame.set_axis(range(len(frame))),
    lambda frame: frame.assign(**{"TCS.NS": np.nan}),
    lambda frame: frame.astype(str),
    lambda frame: frame.astype(bool),
    lambda frame: frame.set_axis(frame.index + pd.Timedelta(hours=1)),
])
def test_bad_frames_rejected(price_dataset, data_request, mutation):
    with pytest.raises(DataQualityError):
        prepare_data(replace(price_dataset, prices=mutation(price_dataset.prices)), data_request)


def test_mixed_currency_rejected(price_dataset, data_request):
    with pytest.raises(DataQualityError, match="currencies"):
        prepare_data(replace(price_dataset, currencies={"RELIANCE.NS": "INR", "TCS.NS": "USD"}), data_request)


def test_insufficient_history(price_dataset, data_request):
    with pytest.raises(DataQualityError, match="found 5"):
        prepare_data(price_dataset, data_request)


def test_out_of_window_rejected(price_dataset, data_request):
    with pytest.raises(DataQualityError, match="window"):
        prepare_data(price_dataset, replace(data_request, end_date=data_request.end_date.replace(day=8)))


def test_unadjusted_rejected(price_dataset, data_request):
    with pytest.raises(DataQualityError, match="adjusted"):
        prepare_data(replace(price_dataset, adjustment_policy="raw_close"), data_request)


def test_timezone_preserves_local_date_labels(price_dataset, data_request):
    prices = price_dataset.prices.copy()
    prices.index = prices.index.tz_localize("Asia/Kolkata")
    result = prepare_data(replace(price_dataset, prices=prices), data_request, AnalysisConfig(min_observations=2))
    assert result.prices.index[0].date() == data_request.start_date
    assert result.prices.index.tz is None


def test_missing_day_is_not_fabricated(price_dataset, data_request):
    result = prepare_data(price_dataset, data_request, AnalysisConfig(min_observations=2))
    assert pd.Timestamp("2025-01-04") not in result.prices.index

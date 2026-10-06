from dataclasses import replace
from datetime import date, datetime

import pytest

from portfolio_optimizer import PortfolioRequest, RequestValidationError, RiskLevel, validate_request


@pytest.fixture
def sample_request():
    return PortfolioRequest(
        tickers=("RELIANCE.NS", "TCS.NS"), investment_amount=100_000,
        start_date=date(2024, 1, 1), end_date=date(2025, 1, 1),
    )


def test_normalizes_without_mutating_input(sample_request):
    raw = replace(sample_request, tickers=[" reliance.ns ", "tcs.ns"], currency=" inr ", risk_level="low")
    result = validate_request(raw, today=date(2026, 10, 6))
    assert result.tickers == ("RELIANCE.NS", "TCS.NS")
    assert result.currency == "INR"
    assert result.risk_level is RiskLevel.LOW
    assert raw.tickers == [" reliance.ns ", "tcs.ns"]


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf"), True, "100", 10**1000])
def test_rejects_invalid_amount(sample_request, amount):
    with pytest.raises(RequestValidationError, match="amount"):
        validate_request(replace(sample_request, investment_amount=amount))


@pytest.mark.parametrize("tickers", [(), "TCS.NS", ("",), ("TCS.NS", " tcs.ns "), (None,), ("A B",), ("^NSEI",)])
def test_rejects_invalid_tickers(sample_request, tickers):
    with pytest.raises(RequestValidationError):
        validate_request(replace(sample_request, tickers=tickers))


@pytest.mark.parametrize("updates", [
    {"start_date": date(2025, 1, 1)},
    {"start_date": date(2026, 1, 1)},
    {"start_date": "2024-01-01"},
    {"start_date": datetime(2024, 1, 1)},
    {"currency": "EUR"}, {"currency": None},
    {"horizon_days": 0}, {"horizon_days": 1.5}, {"horizon_days": True},
    {"risk_level": "extreme"}, {"estimator": "unknown"},
    {"max_asset_weight": 0}, {"max_asset_weight": 1.1},
    {"max_asset_weight": float("nan")}, {"max_asset_weight": True},
    {"max_asset_weight": 0.49},
])
def test_rejects_invalid_fields(sample_request, updates):
    with pytest.raises(RequestValidationError):
        validate_request(replace(sample_request, **updates))


def test_future_end_date_is_rejected(sample_request):
    with pytest.raises(RequestValidationError, match="future"):
        validate_request(sample_request, today=date(2024, 12, 31))


def test_accepts_feasible_boundary_and_current_date(sample_request):
    result = validate_request(replace(sample_request, max_asset_weight=0.5, horizon_days=1), today=sample_request.end_date)
    assert result.max_asset_weight == 0.5


def test_single_stock_requires_full_weight(sample_request):
    with pytest.raises(RequestValidationError, match="cap"):
        validate_request(replace(sample_request, tickers=("TCS.NS",), max_asset_weight=0.9))
    assert validate_request(replace(sample_request, tickers=("TCS.NS",))).tickers == ("TCS.NS",)


def test_non_request_input_rejected():
    with pytest.raises(RequestValidationError):
        validate_request({})

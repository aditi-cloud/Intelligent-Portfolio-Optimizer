"""Validate and canonicalize inputs without network access or UI imports."""

from dataclasses import replace
from datetime import date
import re

from .config import _finite_number
from .contracts import EstimatorChoice, PortfolioRequest, RiskLevel
from .errors import RequestValidationError

# Equity symbols, including Yahoo exchange suffixes and share-class separators.
_TICKER = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,31}\Z")
_SUPPORTED_CURRENCIES = frozenset({"INR", "USD"})


def validate_request(request: PortfolioRequest, *, today: date | None = None) -> PortfolioRequest:
    """Return a canonical immutable request or raise RequestValidationError.

    Pass today's date from the application boundary to make date validation
    deterministic and use the user's local date. Initial currencies: INR/USD.
    Duplicate symbols are rejected after whitespace/case normalization.
    """
    if not isinstance(request, PortfolioRequest):
        raise RequestValidationError("Expected a PortfolioRequest")
    if not isinstance(request.tickers, (tuple, list)) or not request.tickers:
        raise RequestValidationError("Select at least one stock")
    tickers = []
    for ticker in request.tickers:
        if not isinstance(ticker, str):
            raise RequestValidationError("Tickers must be strings")
        symbol = ticker.strip().upper()
        if not _TICKER.fullmatch(symbol):
            raise RequestValidationError(f"Invalid stock ticker: {ticker!r}")
        tickers.append(symbol)
    if len(set(tickers)) != len(tickers):
        raise RequestValidationError("Duplicate stock tickers are not allowed")
    if not _finite_number(request.investment_amount) or request.investment_amount <= 0:
        raise RequestValidationError("Investment amount must be positive and finite")
    if type(request.start_date) is not date or type(request.end_date) is not date:
        raise RequestValidationError("Start and end dates must be date objects")
    if request.start_date >= request.end_date:
        raise RequestValidationError("Start date must be before end date")
    if today is not None:
        if type(today) is not date:
            raise RequestValidationError("Today's date must be a date object")
        if request.end_date > today:
            raise RequestValidationError("Historical end date cannot be in the future")
    if not isinstance(request.currency, str):
        raise RequestValidationError("Currency must be INR or USD")
    currency = request.currency.strip().upper()
    if currency not in _SUPPORTED_CURRENCIES:
        raise RequestValidationError("Currency must be INR or USD")
    if type(request.horizon_days) is not int or request.horizon_days < 1:
        raise RequestValidationError("Horizon must be a positive integer of trading days")
    try:
        risk = RiskLevel(request.risk_level)
        estimator = EstimatorChoice(request.estimator)
    except (ValueError, TypeError) as exc:
        raise RequestValidationError("Unsupported risk level or estimator") from exc
    cap = request.max_asset_weight
    if not _finite_number(cap) or not 0 < cap <= 1:
        raise RequestValidationError("Maximum asset weight must be in (0, 1]")
    if len(tickers) * cap < 1 - 1e-12:
        raise RequestValidationError("Weight cap is too small for a fully invested portfolio")
    return replace(
        request, tickers=tuple(tickers), investment_amount=float(request.investment_amount),
        currency=currency, risk_level=risk, estimator=estimator, max_asset_weight=float(cap),
    )

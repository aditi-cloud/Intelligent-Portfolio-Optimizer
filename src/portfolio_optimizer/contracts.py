"""Immutable public contracts. Return and covariance inputs are annualized.

Matrix/series representations will be defined with the data component, rather
than exposing unvalidated placeholder objects in the public API.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EstimatorChoice(str, Enum):
    HISTORICAL = "historical"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


@dataclass(frozen=True, slots=True)
class PortfolioRequest:
    """Raw request; call validate_request before using it in the pipeline.

    Dates are inclusive at this boundary. The provider adapter must translate
    them to its own API semantics. Horizon is measured in trading days.
    Currency is the desired analysis currency; asset currencies still need
    validation against provider metadata.
    """

    tickers: tuple[str, ...]
    investment_amount: float
    start_date: date
    end_date: date
    currency: str = "INR"
    horizon_days: int = 21
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_asset_weight: float = 1.0
    estimator: EstimatorChoice = EstimatorChoice.HISTORICAL

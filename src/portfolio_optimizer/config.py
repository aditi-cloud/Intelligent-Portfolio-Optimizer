"""Explicit numerical conventions shared by all analysis components."""

from dataclasses import dataclass
from math import isfinite

from .contracts import RiskLevel


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    trading_days_per_year: int = 252
    annual_risk_free_rate: float = 0.0
    min_observations: int = 60
    random_seed: int = 42
    low_risk_aversion: float = 10.0
    medium_risk_aversion: float = 3.0
    high_risk_aversion: float = 1.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("trading_days_per_year", 1), ("min_observations", 2), ("random_seed", 0)
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not _finite_number(self.annual_risk_free_rate) or self.annual_risk_free_rate <= -1:
            raise ValueError("annual_risk_free_rate must be finite and greater than -1")
        values = (self.low_risk_aversion, self.medium_risk_aversion, self.high_risk_aversion)
        if not all(_finite_number(value) and value > 0 for value in values):
            raise ValueError("risk aversion values must be positive and finite")
        if not values[0] > values[1] > values[2]:
            raise ValueError("risk aversion must decrease from low to high risk")

    def risk_aversion(self, risk_level: RiskLevel) -> float:
        return {
            RiskLevel.LOW: self.low_risk_aversion,
            RiskLevel.MEDIUM: self.medium_risk_aversion,
            RiskLevel.HIGH: self.high_risk_aversion,
        }[RiskLevel(risk_level)]


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False

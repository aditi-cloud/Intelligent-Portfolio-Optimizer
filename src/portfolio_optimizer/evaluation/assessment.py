"""Interpret baseline estimates without implying validated future performance."""

from dataclasses import dataclass

from ..forecasting.contracts import MarketEstimates
from .metrics import PortfolioMetrics


@dataclass(frozen=True)
class PortfolioAssessment:
    status: str
    headline: str
    explanation: str


def assess_baseline(estimates: MarketEstimates, metrics: PortfolioMetrics) -> PortfolioAssessment:
    if metrics.expected_annual_return <= 0:
        stock_context = (
            "Every selected stock has a nonpositive annualized historical mean. "
            if (estimates.expected_returns <= 0).all() else ""
        )
        return PortfolioAssessment(
            "nonpositive_historical_mean", "This allocation has no positive historical return estimate",
            stock_context + "The optimizer still allocates the full budget because this model requires 100% investment in the selected stocks. "
            "This result does not establish a reason to invest. High risk reduces the penalty on variance; it cannot create positive returns.",
        )
    if metrics.expected_annual_return < metrics.annual_risk_free_rate:
        return PortfolioAssessment(
            "below_reference_rate", "Historical mean return is below the configured reference rate",
            f"The annualized historical mean is {metrics.expected_annual_return:.2%}, versus the configured risk-free rate of {metrics.annual_risk_free_rate:.2%}. "
            "The full-investment constraint still produces a stock allocation. The reference rate is an analysis input, not a verified cash investment offer.",
        )
    return PortfolioAssessment(
        "historical_baseline", "Historical baseline analysis",
        "These estimates summarize the selected historical window. This version has no ML forecast of future returns. "
        "A positive historical estimate does not establish future profitability; review the separate out-of-sample backtest.",
    )

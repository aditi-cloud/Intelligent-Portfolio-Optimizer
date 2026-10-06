from dataclasses import dataclass

import numpy as np

from ..errors import OptimizationError
from ..forecasting.contracts import MarketEstimates, validate_estimates
from .solver import OptimizationResult, minimum_variance, validate_cap


@dataclass(frozen=True)
class EfficientFrontier:
    points: tuple[OptimizationResult, ...]
    minimum_variance_return: float
    maximum_feasible_return: float
    degenerate: bool


def build_frontier(estimates: MarketEstimates, *, max_asset_weight: float = 1.0,
                   points: int = 20, annual_risk_free_rate: float = 0.0,
                   solver: str = "CLARABEL") -> EfficientFrontier:
    """Trace the efficient upper branch from global minimum variance to max return.

    Return constraints are lower bounds. Failures propagate rather than dropping
    points silently. A fixed portfolio/identical-return basket yields one point.
    """
    validate_estimates(estimates)
    validate_cap(max_asset_weight, len(estimates.tickers))
    if type(points) is not int or points < 2:
        raise OptimizationError("Frontier requires an integer point count >= 2")
    base = minimum_variance(estimates, max_asset_weight=max_asset_weight,
                            annual_risk_free_rate=annual_risk_free_rate, solver=solver)
    mu = estimates.expected_returns.to_numpy(dtype=float)
    remaining, maximum = 1.0, 0.0
    for index in np.argsort(-mu, kind="stable"):
        weight = min(max_asset_weight, remaining)
        maximum += weight * mu[index]
        remaining -= weight
        if remaining <= 1e-14:
            break
    start = base.metrics.expected_annual_return
    if maximum - start <= 1e-9:
        return EfficientFrontier((base,), start, float(maximum), True)
    results = [base]
    for target in np.linspace(start, maximum, points)[1:]:
        results.append(minimum_variance(estimates, max_asset_weight=max_asset_weight,
                                       target_return=float(target), annual_risk_free_rate=annual_risk_free_rate,
                                       solver=solver))
    return EfficientFrontier(tuple(results), start, float(maximum), False)

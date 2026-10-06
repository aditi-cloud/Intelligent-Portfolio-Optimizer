"""Constrained mean-variance optimization with explicit numerical checks.

CVXPY status reference: https://www.cvxpy.org/tutorial/intro/
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from ..config import AnalysisConfig, _finite_number
from ..contracts import PortfolioRequest
from ..errors import OptimizationError
from ..evaluation.metrics import PortfolioMetrics, expected_metrics
from ..forecasting.contracts import MarketEstimates, validate_estimates
from ..validation import validate_request


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    metrics: PortfolioMetrics
    objective_value: float
    objective: str
    status: str
    solver: str
    max_asset_weight: float
    risk_aversion: float | None = None
    target_return: float | None = None
    warnings: tuple[str, ...] = ()


def validate_cap(cap: float, assets: int) -> None:
    if not _finite_number(cap) or not 0 < cap <= 1 or assets * cap < 1 - 1e-12:
        raise OptimizationError("Weight cap is invalid or infeasible")


def _project_weights(raw: np.ndarray, cap: float) -> np.ndarray:
    """Remove only small solver residuals by projection onto a capped simplex."""
    if not np.isfinite(raw).all() or (raw < -1e-6).any() or (raw > cap + 1e-6).any() or abs(raw.sum() - 1) > 1e-6:
        raise OptimizationError("Solver returned invalid portfolio weights")
    lo, hi = float(raw.min() - cap), float(raw.max())
    for _ in range(80):
        midpoint = (lo + hi) / 2
        if np.clip(raw - midpoint, 0, cap).sum() > 1:
            lo = midpoint
        else:
            hi = midpoint
    weights = np.clip(raw - (lo + hi) / 2, 0, cap)
    if abs(weights.sum() - 1) > 1e-10:
        raise OptimizationError("Cannot remove solver weight residuals")
    return weights


def _solve(estimates: MarketEstimates, cap: float, *, risk_aversion: float | None,
           target_return: float | None, annual_risk_free_rate: float, solver: str) -> OptimizationResult:
    validate_estimates(estimates)
    validate_cap(cap, len(estimates.tickers))
    mu = estimates.expected_returns.to_numpy(dtype=float)
    matrix = estimates.covariance.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2
    # Only tolerated floating-point negative eigenvalues can reach this point.
    eigenvalues, vectors = np.linalg.eigh(matrix)
    warnings = []
    if eigenvalues.min() < 0:
        matrix = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
        warnings.append("Removed numerical negative covariance eigenvalues for solving")
    w = cp.Variable(len(mu))
    variance = cp.quad_form(w, cp.psd_wrap(matrix))
    constraints = [cp.sum(w) == 1, w >= 0, w <= cap]
    if target_return is not None:
        if not _finite_number(target_return):
            raise OptimizationError("Target return must be finite")
        constraints.append(mu @ w >= target_return)
    if risk_aversion is None:
        objective = cp.Minimize(variance)
        label = "minimum_variance"
    else:
        objective = cp.Maximize(mu @ w - risk_aversion * variance)
        label = "mean_variance_utility"
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver)
    except cp.error.SolverError as exc:
        raise OptimizationError(f"Portfolio solver {solver} failed") from exc
    if problem.status != cp.OPTIMAL:
        raise OptimizationError(f"Optimization did not converge accurately: {problem.status}")
    if w.value is None or problem.value is None or not np.isfinite(problem.value):
        raise OptimizationError("Solver returned no finite solution")
    weights = pd.Series(_project_weights(np.asarray(w.value).reshape(-1), cap),
                        index=list(estimates.tickers), name="weight")
    metrics = expected_metrics(weights, estimates, annual_risk_free_rate=annual_risk_free_rate)
    if target_return is not None and metrics.expected_annual_return < target_return - 1e-7:
        raise OptimizationError("Solver solution violates the return target")
    objective_value = metrics.annual_volatility ** 2
    if risk_aversion is not None:
        objective_value = metrics.expected_annual_return - risk_aversion * objective_value
    return OptimizationResult(weights, metrics, objective_value, label, problem.status, solver, cap,
                              risk_aversion, target_return, tuple(warnings))


def optimize_portfolio(estimates: MarketEstimates, request: PortfolioRequest,
                       config: AnalysisConfig | None = None, *, solver: str = "CLARABEL") -> OptimizationResult:
    config = config or AnalysisConfig()
    request = validate_request(request)
    if estimates.tickers != request.tickers:
        raise OptimizationError("Estimated assets do not match the requested order")
    if estimates.cutoff > request.end_date or estimates.cutoff < request.start_date:
        raise OptimizationError("Estimation cutoff falls outside requested dates")
    if estimates.trading_days_per_year != config.trading_days_per_year:
        raise OptimizationError("Annualization conventions do not match")
    return _solve(estimates, request.max_asset_weight, risk_aversion=config.risk_aversion(request.risk_level),
                  target_return=None, annual_risk_free_rate=config.annual_risk_free_rate, solver=solver)


def minimum_variance(estimates: MarketEstimates, *, max_asset_weight: float = 1.0,
                     target_return: float | None = None, annual_risk_free_rate: float = 0.0,
                     solver: str = "CLARABEL") -> OptimizationResult:
    return _solve(estimates, max_asset_weight, risk_aversion=None, target_return=target_return,
                  annual_risk_free_rate=annual_risk_free_rate, solver=solver)

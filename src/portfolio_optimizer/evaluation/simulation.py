from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import _finite_number
from ..errors import SimulationError
from ..forecasting.contracts import MarketEstimates, covariance_tolerance, validate_estimates
from .metrics import expected_metrics


@dataclass(frozen=True)
class SimulationResult:
    wealth_paths: pd.DataFrame
    path_quantiles: pd.DataFrame
    terminal_asset_returns: pd.DataFrame
    terminal_quantiles: Mapping[str, float]
    mean_terminal_wealth: float
    model_loss_probability: float
    horizon_days: int
    scenarios: int
    seed: int
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    initial_wealth: float
    weights: pd.Series
    estimation_cutoff: date
    trading_days_per_year: int


def lognormal_parameters(estimates: MarketEstimates) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Match daily simple-return means/covariance with correlated lognormal gross returns.

    Cov(log G_i, log G_j) = log(1 + Cov(G_i,G_j)/(E[G_i]E[G_j])).
    Reject incompatible moments rather than silently simulating other estimates.
    """
    validate_estimates(estimates)
    daily_mean = estimates.expected_returns.to_numpy(dtype=float) / estimates.trading_days_per_year
    gross_mean = 1 + daily_mean
    daily_cov = estimates.covariance.to_numpy(dtype=float) / estimates.trading_days_per_year
    if (gross_mean <= 0).any():
        raise SimulationError("Daily expected gross returns must be positive")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            ratio = daily_cov / np.outer(gross_mean, gross_mean)
            if (ratio <= -1).any():
                raise SimulationError("Return moments are incompatible with positive lognormal prices")
            log_cov = np.log1p(ratio)
            log_cov = (log_cov + log_cov.T) / 2
            eigenvalues, vectors = np.linalg.eigh(log_cov)
            if eigenvalues.min() < -covariance_tolerance(log_cov):
                raise SimulationError("Moment-matched log covariance is not positive semidefinite")
            warnings = ()
            if eigenvalues.min() < 0:
                log_cov = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
                warnings = ("Removed numerical negative log-covariance eigenvalues",)
            log_mean = np.log(gross_mean) - .5 * np.diag(log_cov)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise SimulationError("Lognormal parameter calculation failed") from exc
    return log_mean, log_cov, warnings


def simulate_portfolio(estimates: MarketEstimates, weights: pd.Series, investment_amount: float, *,
                       horizon_days: int = 21, scenarios: int = 2_000, seed: int = 42) -> SimulationResult:
    """Correlated IID daily lognormal scenarios for a buy-and-hold portfolio.

    Model frequencies are conditional on fixed estimated moments; they are not
    empirical market probabilities. No daily rebalancing or trading fees.
    """
    expected_metrics(weights, estimates)  # Validate labels and portfolio constraints.
    for name, value, minimum in (("horizon_days", horizon_days, 1), ("scenarios", scenarios, 2), ("seed", seed, 0)):
        if type(value) is not int or value < minimum:
            raise SimulationError(f"{name} must be an integer >= {minimum}")
    if scenarios * (horizon_days + 1 + len(estimates.tickers)) > 10_000_000:
        raise SimulationError("Simulation exceeds the 10 million stored-value limit")
    if not _finite_number(investment_amount) or investment_amount <= 0:
        raise SimulationError("Simulation investment must be positive and finite")
    log_mean, log_cov, warnings = lognormal_parameters(estimates)
    eigenvalues, vectors = np.linalg.eigh(log_cov)
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0))
    rng = np.random.default_rng(seed)
    asset_growth = np.ones((scenarios, len(estimates.tickers)))
    wealth = np.empty((horizon_days + 1, scenarios))
    wealth[0] = investment_amount
    aligned_weights = weights.reindex(estimates.tickers).to_numpy(dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for day in range(1, horizon_days + 1):
                shocks = rng.standard_normal(asset_growth.shape) @ root.T
                asset_growth *= np.exp(log_mean + shocks)
                wealth[day] = investment_amount * (asset_growth @ aligned_weights)
                if (asset_growth <= 0).any() or not np.isfinite(asset_growth).all() or (wealth[day] <= 0).any():
                    raise SimulationError("Simulation produced nonpositive or non-finite wealth")
    except FloatingPointError as exc:
        raise SimulationError("Simulation overflowed; reduce horizon or review estimates") from exc
    quantile_levels = [.05, .25, .5, .75, .95]
    labels = ["p05", "p25", "p50", "p75", "p95"]
    quantiles = np.quantile(wealth, quantile_levels, axis=1).T
    days = pd.RangeIndex(horizon_days + 1, name="trading_day")
    terminal = wealth[-1]
    mean_wealth = float(terminal.mean())
    if not np.isfinite(mean_wealth):
        raise SimulationError("Mean scenario wealth overflowed")
    return SimulationResult(
        pd.DataFrame(wealth, index=days), pd.DataFrame(quantiles, index=days, columns=labels),
        pd.DataFrame(asset_growth - 1, columns=list(estimates.tickers)),
        dict(zip(labels, map(float, quantiles[-1]))), mean_wealth,
        float(np.mean(terminal < investment_amount)), horizon_days, scenarios, seed,
        ("Correlated lognormal daily gross returns matched to estimated daily simple-return moments",
         "IID sessions and fixed moments across the entire horizon; no volatility regime changes",
         "Buy-and-hold fractional portfolio with drifting weights; no fees, tax or slippage",
         "Loss frequency is conditional on model assumptions, not an observed market probability"), warnings,
        float(investment_amount), weights.reindex(estimates.tickers).copy(), estimates.cutoff, estimates.trading_days_per_year,
    )

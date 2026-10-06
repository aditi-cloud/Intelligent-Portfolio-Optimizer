"""Walk-forward close-to-close replay with a full session of execution delay.

Signal at close t uses only returns through t; execution is at close t+1;
new weights first earn the return from close t+1 to close t+2. Existing holdings
earn the execution day's return before rebalancing. Fractional holdings and
adjusted-close total-return accounting are used, with no actual trade execution.
"""

from dataclasses import dataclass, replace
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from ..config import AnalysisConfig, _finite_number
from ..contracts import EstimatorChoice, PortfolioRequest
from ..data.contracts import PreparedData
from ..data.preparation import validate_prices
from ..errors import BacktestError
from ..forecasting import MarketEstimates, estimate_covariance, estimate_returns
from ..forecasting.validation import numeric_values, validate_returns
from ..optimization.solver import optimize_portfolio
from ..validation import validate_request
from .metrics import equal_weights
from .realized import RealizedMetrics, realized_metrics

WeightStrategy = Callable[[pd.DataFrame, PortfolioRequest, AnalysisConfig], pd.Series]


@dataclass(frozen=True)
class BacktestConfig:
    training_observations: int = 60
    rebalance_every: int = 21
    transaction_cost_bps: float = 10.0

    def __post_init__(self):
        if type(self.training_observations) is not int or self.training_observations < 2:
            raise ValueError("Training observations must be an integer >= 2")
        if type(self.rebalance_every) is not int or self.rebalance_every < 1:
            raise ValueError("Rebalance frequency must be a positive integer")
        if not _finite_number(self.transaction_cost_bps) or not 0 <= self.transaction_cost_bps < 10_000:
            raise ValueError("Transaction cost must be in [0, 10000) basis points")


@dataclass(frozen=True)
class RebalanceRecord:
    training_start: date
    signal_date: date
    execution_date: date
    observations: int
    target_weights: pd.Series
    pre_trade_weights: pd.Series
    traded_notional: float
    turnover: float
    cost: float


@dataclass(frozen=True)
class StrategyBacktest:
    name: str
    initial_wealth: float
    equity: pd.Series
    net_returns: pd.Series
    end_of_day_weights: pd.DataFrame
    rebalances: tuple[RebalanceRecord, ...]
    metrics: RealizedMetrics
    total_cost: float
    settings: BacktestConfig
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class BacktestComparison:
    optimized: StrategyBacktest
    equal_weight: StrategyBacktest


def historical_strategy(training: pd.DataFrame, request: PortfolioRequest, config: AnalysisConfig) -> pd.Series:
    covariance = estimate_covariance(training, config)
    estimates = MarketEstimates(estimate_returns(training, config), covariance.annual_covariance,
                                training.index[-1].date(), len(training), config.trading_days_per_year,
                                shrinkage=covariance.shrinkage)
    return optimize_portfolio(estimates, request, config).weights


def equal_weight_strategy(training: pd.DataFrame, request: PortfolioRequest, config: AnalysisConfig) -> pd.Series:
    return equal_weights(request.tickers)


def _rebalance(holdings: np.ndarray, wealth: float, target: np.ndarray, rate: float):
    """Solve self-financing fee = rate * sum(abs(new holdings - old holdings))."""
    if rate == 0:
        cost = 0.0
    else:
        low, high = 0.0, wealth
        for _ in range(80):
            cost = (low + high) / 2
            notional = np.abs(target * (wealth - cost) - holdings).sum()
            if cost > rate * notional:
                high = cost
            else:
                low = cost
        cost = (low + high) / 2
    new_holdings = target * (wealth - cost)
    notional = float(np.abs(new_holdings - holdings).sum())
    return new_holdings, cost, notional


def run_backtest(prepared: PreparedData, request: PortfolioRequest, config: AnalysisConfig | None = None,
                 settings: BacktestConfig | None = None, *, strategy: WeightStrategy = historical_strategy,
                 name: str = "historical_optimized") -> StrategyBacktest:
    config, settings = config or AnalysisConfig(), settings or BacktestConfig()
    request = validate_request(request)
    if request.estimator is not EstimatorChoice.HISTORICAL:
        raise BacktestError("Backtesting currently supports only historical estimation")
    if settings.training_observations < config.min_observations:
        raise BacktestError("Training window is shorter than the estimator minimum")
    prices = validate_prices(prepared.source, request)
    returns = validate_returns(prepared.returns, config)
    # Skipping a missing return loses a holding-period move. Reject such replay.
    expected = prices.pct_change(fill_method=None).iloc[1:]
    if prices.isna().any().any() or not returns.index.equals(expected.index) or tuple(returns.columns) != request.tickers or not np.allclose(returns.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12):
        raise BacktestError("Backtesting requires complete consecutive price-session returns")
    window = settings.training_observations
    if len(returns) < window + 2:
        raise BacktestError("Need a training window, execution session and at least one holding session")
    holdings = np.zeros(len(request.tickers))
    cash = request.investment_amount
    prior_wealth = cash
    equity, net_returns, daily_weights, records = [], [], [], []
    for position in range(window, len(returns)):
        with np.errstate(over="ignore", invalid="ignore"):
            holdings = holdings * (1 + returns.iloc[position].to_numpy())
        wealth = float(holdings.sum() + cash)
        if not np.isfinite(wealth) or wealth <= 0:
            raise BacktestError("Backtest wealth became nonpositive or non-finite")
        if (position - window) % settings.rebalance_every == 0 and position < len(returns) - 1:
            training = returns.iloc[position - window:position].copy()
            window_request = replace(request, start_date=training.index[0].date(), end_date=training.index[-1].date())
            target_series = strategy(training.copy(), window_request, config)
            if not isinstance(target_series, pd.Series) or target_series.index.has_duplicates or set(target_series.index) != set(request.tickers):
                raise BacktestError("Strategy must return one weight per requested ticker")
            target = numeric_values(target_series.reindex(request.tickers))
            if (target < 0).any() or (target > request.max_asset_weight + 1e-8).any() or abs(target.sum() - 1) > 1e-8:
                raise BacktestError("Strategy returned infeasible weights")
            target = target / target.sum()
            pre_weights = holdings / wealth
            holdings, cost, notional = _rebalance(holdings, wealth, target, settings.transaction_cost_bps / 10_000)
            cash = 0.0
            records.append(RebalanceRecord(training.index[0].date(), training.index[-1].date(),
                                           returns.index[position].date(), len(training),
                                           pd.Series(target, index=list(request.tickers), name="weight"),
                                           pd.Series(pre_weights, index=list(request.tickers), name="pre_trade_weight"),
                                           notional, notional / wealth, cost))
            wealth = float(holdings.sum())
        if wealth <= 0 or not np.isfinite(wealth):
            raise BacktestError("Costs exhausted portfolio wealth")
        equity.append(wealth)
        net_returns.append(wealth / prior_wealth - 1)
        daily_weights.append(holdings / wealth)
        prior_wealth = wealth
    dates = returns.index[window:]
    net_series = pd.Series(net_returns, index=dates, name="net_return")
    return StrategyBacktest(
        name, request.investment_amount, pd.Series(equity, index=dates, name="equity"), net_series,
        pd.DataFrame(daily_weights, index=dates, columns=list(request.tickers)), tuple(records),
        realized_metrics(net_series, config), float(sum(record.cost for record in records)), settings,
        ("Signals use a rolling historical window ending before execution",
         "Execute at next observed adjusted close; weights first earn returns in the following session",
         "Fractional holdings and adjusted-close total-return accounting; cash earns zero before entry",
         "Fees apply to actual buy plus sell notional; no slippage, tax or terminal liquidation",
         "Session calendar completeness is assumed; missing price rows are rejected"),
    )


def compare_backtests(prepared: PreparedData, request: PortfolioRequest, config: AnalysisConfig | None = None,
                      settings: BacktestConfig | None = None) -> BacktestComparison:
    return BacktestComparison(run_backtest(prepared, request, config, settings),
                              run_backtest(prepared, request, config, settings,
                                           strategy=equal_weight_strategy, name="equal_weight"))

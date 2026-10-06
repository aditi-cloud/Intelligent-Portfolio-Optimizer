"""Offline baseline demonstration using synthetic prices, not market evidence.

Run from the repository root: .venv/bin/python examples/baseline.py
"""

import json

from portfolio_optimizer import AnalysisConfig, PortfolioRequest, validate_request
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.data.demo import demo_dataset, DEMO_START, DEMO_END, DEMO_TICKERS
from portfolio_optimizer.evaluation import equal_weight_metrics, equal_weights
from portfolio_optimizer.forecasting import estimate_market
from portfolio_optimizer.optimization import allocate_currency, build_frontier, optimize_portfolio
from portfolio_optimizer.evaluation.backtest import BacktestConfig, compare_backtests
from portfolio_optimizer.evaluation.simulation import simulate_portfolio


def synthetic_inputs():
    config = AnalysisConfig(annual_risk_free_rate=.04)
    request = validate_request(PortfolioRequest(DEMO_TICKERS, 100_000, DEMO_START, DEMO_END))
    dataset = demo_dataset()
    return config, request, dataset


def main():
    config, request, dataset = synthetic_inputs()
    prepared = prepare_data(dataset, request, config)
    estimates = estimate_market(prepared, request, config)
    metrics = equal_weight_metrics(estimates, annual_risk_free_rate=config.annual_risk_free_rate)
    optimized = optimize_portfolio(estimates, request, config)
    allocation = allocate_currency(optimized.weights, request.investment_amount, currency=request.currency)
    frontier = build_frontier(estimates, points=8, annual_risk_free_rate=config.annual_risk_free_rate)
    replay = compare_backtests(prepared, request, config, BacktestConfig(60, 21, 10))
    simulation = simulate_portfolio(estimates, optimized.weights, request.investment_amount,
                                     horizon_days=request.horizon_days, scenarios=1_000, seed=config.random_seed)
    result = {
        "source": "synthetic_demo — not observed market data",
        "cutoff": estimates.cutoff.isoformat(), "observations": estimates.observations,
        "annualization_days": estimates.trading_days_per_year,
        "return_method": estimates.return_method, "covariance_method": estimates.covariance_method,
        "shrinkage": estimates.shrinkage,
        "expected_annual_asset_returns": estimates.expected_returns.to_dict(),
        "equal_weights": equal_weights(request.tickers).to_dict(),
        "equal_weight_expected_metrics": {
            "annual_return": metrics.expected_annual_return, "annual_volatility": metrics.annual_volatility,
            "sharpe": metrics.expected_sharpe_ratio, "risk_free_rate": metrics.annual_risk_free_rate,
        },
        "optimized_weights": optimized.weights.to_dict(),
        "optimized_expected_metrics": {
            "annual_return": optimized.metrics.expected_annual_return,
            "annual_volatility": optimized.metrics.annual_volatility,
            "sharpe": optimized.metrics.expected_sharpe_ratio,
        },
        "solver_status": optimized.status,
        "currency": allocation.currency,
        "allocation": {ticker: str(amount) for ticker, amount in allocation.amounts.items()},
        "frontier": [{"annual_return": point.metrics.expected_annual_return,
                      "annual_volatility": point.metrics.annual_volatility} for point in frontier.points],
        "backtest": {
            strategy.name: {"terminal_wealth": float(strategy.equity.iloc[-1]),
                            "realized_total_return": strategy.metrics.total_return,
                            "realized_sharpe": strategy.metrics.realized_sharpe_ratio,
                            "max_drawdown": strategy.metrics.max_drawdown,
                            "total_cost": strategy.total_cost,
                            "rebalances": len(strategy.rebalances)}
            for strategy in (replay.optimized, replay.equal_weight)
        },
        "simulation": {
            "horizon_days": simulation.horizon_days, "scenarios": simulation.scenarios,
            "seed": simulation.seed, "terminal_wealth_quantiles": simulation.terminal_quantiles,
            "model_loss_probability": simulation.model_loss_probability,
            "assumptions": simulation.assumptions,
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

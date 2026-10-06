from dataclasses import asdict

import pandas as pd
import streamlit as st

from ..services.analysis import AnalysisResult
from .charts import allocation_chart, backtest_chart, frontier_chart, scenario_chart


def _sharpe(value):
    return "Undefined" if value is None else f"{value:.2f}"


def render_results(result: AnalysisResult):
    st.subheader("Constrained stock allocation")
    st.caption(f"{result.request.currency} {result.allocation.budget:,.2f} · {len(result.request.tickers)} stocks · "
               f"{result.request.risk_level.value.title()} risk · estimates through {result.estimates.cutoff}")
    metrics = result.portfolio.metrics
    if result.assessment.status != "historical_baseline":
        st.warning(f"{result.assessment.headline}. {result.assessment.explanation}")
    columns = st.columns(3)
    columns[0].metric("Historical mean return (annualized)", f"{metrics.expected_annual_return:.2%}")
    columns[1].metric("Historical volatility (annualized)", f"{metrics.annual_volatility:.2%}")
    columns[2].metric("Historical Sharpe estimate", _sharpe(metrics.expected_sharpe_ratio))
    st.caption("Model: historical baseline. Future-return forecasting is not available in this version. High risk changes risk aversion; it does not promise higher returns.")
    for issue in result.issues:
        st.warning(f"{issue.component.title()} unavailable: {issue.message}")
    for warning in result.warnings:
        st.warning(warning)
    allocation, frontier, scenarios, backtest, details = st.tabs(["Allocation", "Risk & return", "Scenarios", "Backtest", "Data & assumptions"])
    with allocation:
        left, right = st.columns([1, 1.2])
        with left:
            st.plotly_chart(allocation_chart(result), width="stretch", key="allocation_chart")
        with right:
            table = pd.DataFrame({
                "Stock": result.request.tickers,
                "Target weight (%)": [100 * result.portfolio.weights[t] for t in result.request.tickers],
                f"Amount ({result.request.currency})": [float(result.allocation.amounts[t]) for t in result.request.tickers],
                "Rounded weight (%)": [100 * result.allocation.realized_weights[t] for t in result.request.tickers],
            })
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Target weight (%)": st.column_config.NumberColumn(format="%.2f"),
                "Rounded weight (%)": st.column_config.NumberColumn(format="%.2f"),
                f"Amount ({result.request.currency})": st.column_config.NumberColumn(format="%.2f"),
            })
            st.caption("Amounts sum to your budget. Rounding may slightly change weights; share quantities are not included.")
    with frontier:
        st.plotly_chart(frontier_chart(result), width="stretch", key="frontier_chart")
        st.dataframe(pd.DataFrame([
            {"Portfolio": name, "Historical mean return (annualized, %)": m.expected_annual_return * 100,
             "Historical volatility (annualized, %)": m.annual_volatility * 100, "Historical Sharpe": _sharpe(m.expected_sharpe_ratio)}
            for name, m in [("Optimized", result.portfolio.metrics), ("Equal weight", result.benchmark)]
        ]), hide_index=True, width="stretch")
        st.caption("The selected risk setting balances expected return against variance under your weight cap.")
        if result.frontier.degenerate:
            st.info("These constraints produce a single frontier point.")
    with scenarios:
        if result.simulation:
            sim = result.simulation
            st.caption(f"{sim.scenarios:,} scenarios over {sim.horizon_days} trading days · seed {sim.seed}")
            st.plotly_chart(scenario_chart(result), width="stretch", key="scenario_chart")
            columns = st.columns(3)
            columns[0].metric("5th percentile value", f"{result.request.currency} {sim.terminal_quantiles['p05']:,.2f}")
            columns[1].metric("Median value", f"{result.request.currency} {sim.terminal_quantiles['p50']:,.2f}")
            columns[2].metric("Model loss frequency", f"{sim.model_loss_probability:.1%}")
            st.caption("The shaded area spans the 5th–95th percentiles of model paths. Loss frequency depends on the stated model assumptions.")
        else:
            st.info("Scenarios were not requested." if not result.options.include_simulation else "Scenarios could not be calculated; see the message above.")
    with backtest:
        if result.backtest:
            runs = [("Optimized", result.backtest.optimized), ("Equal weight", result.backtest.equal_weight)]
            st.plotly_chart(backtest_chart(result), width="stretch", key="backtest_chart")
            st.dataframe(pd.DataFrame([
                {"Strategy": label, "Realized total return (%)": run.metrics.total_return * 100,
                 "Realized Sharpe": _sharpe(run.metrics.realized_sharpe_ratio), "Max drawdown (%)": run.metrics.max_drawdown * 100,
                 f"Fees ({result.request.currency})": run.total_cost} for label, run in runs
            ]), hide_index=True, width="stretch")
            settings = result.backtest.optimized.settings
            st.caption(f"Training: {settings.training_observations} sessions · rebalance every {settings.rebalance_every} sessions · fees: {settings.transaction_cost_bps:g} bps. Signals execute at the next observed close.")
        else:
            st.info("Backtesting was not requested." if result.options.backtest is None else "Backtesting could not be completed; see the message above.")
    with details:
        st.write("**Price data**")
        st.write(f"Source: {result.prepared.source.provider} · adjusted closes · {result.estimates.observations} aligned returns")
        st.json(asdict(result.prepared.quality))
        with st.expander("Recent adjusted prices"):
            st.dataframe(result.prepared.prices.tail(20), width="stretch")
        st.write("**Analysis assumptions**")
        st.write(result.assessment.explanation)
        st.dataframe(result.estimates.expected_returns.mul(100).rename("Historical mean return (annualized, %)"), width="stretch")
        st.write(f"Annualization: {result.config.trading_days_per_year} sessions. Annual risk-free rate: {result.config.annual_risk_free_rate:.2%}.")
        st.write("Historical arithmetic return estimates and shrinkage covariance. Long-only, fully invested target portfolio. Risk settings are configurable risk-aversion parameters.")
        if result.simulation:
            for assumption in result.simulation.assumptions:
                st.write(f"• {assumption}")
        if result.backtest:
            for assumption in result.backtest.optimized.assumptions:
                st.write(f"• {assumption}")

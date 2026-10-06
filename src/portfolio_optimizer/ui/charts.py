"""Interactive views of existing results; no model fitting or optimization."""

import plotly.graph_objects as go

from ..services.analysis import AnalysisResult

COLORS = ["#4f46e5", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"]


def _style(figure: go.Figure, *, x_title=None, y_title=None) -> go.Figure:
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=30),
                         height=400, colorway=COLORS, legend=dict(orientation="h", y=-.2),
                         xaxis_title=x_title, yaxis_title=y_title)
    return figure


def allocation_chart(result: AnalysisResult) -> go.Figure:
    labels = [f"{ticker}<br>{weight:.1%}" if weight >= .001 else ""
              for ticker, weight in zip(result.request.tickers, result.portfolio.weights)]
    figure = go.Figure(go.Pie(labels=list(result.request.tickers), values=result.portfolio.weights.tolist(),
                             hole=.65, marker=dict(colors=COLORS), sort=False,
                             text=labels, textinfo="text", hovertemplate="%{label}: %{percent}<extra></extra>"))
    return _style(figure)


def frontier_chart(result: AnalysisResult) -> go.Figure:
    figure = go.Figure(go.Scatter(x=[p.metrics.annual_volatility * 100 for p in result.frontier.points],
                                 y=[p.metrics.expected_annual_return * 100 for p in result.frontier.points],
                                 mode="lines+markers", name="Efficient Frontier"))
    for name, metrics, color in [("Optimized", result.portfolio.metrics, COLORS[1]),
                                  ("Equal weight", result.benchmark, COLORS[2])]:
        figure.add_trace(go.Scatter(x=[metrics.annual_volatility * 100], y=[metrics.expected_annual_return * 100],
                                    mode="markers", name=name, marker=dict(size=14, color=color, symbol="diamond")))
    return _style(figure, x_title="Historical volatility (annualized, %)", y_title="Historical mean return (annualized, %)")


def scenario_chart(result: AnalysisResult) -> go.Figure:
    if result.simulation is None:
        raise ValueError("Scenario result is unavailable")
    quantiles = result.simulation.path_quantiles
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=quantiles.index.tolist(), y=quantiles["p05"].tolist(), mode="lines", name="5th percentile",
                                line=dict(width=1, color=COLORS[0])))
    figure.add_trace(go.Scatter(x=quantiles.index.tolist(), y=quantiles["p95"].tolist(), mode="lines", name="95th percentile",
                                fill="tonexty", fillcolor="rgba(79,70,229,.12)", line=dict(width=1, color=COLORS[0])))
    figure.add_trace(go.Scatter(x=quantiles.index.tolist(), y=quantiles["p50"].tolist(), mode="lines", name="Median",
                                line=dict(width=3, color=COLORS[0])))
    return _style(figure, x_title="Trading day", y_title=f"Portfolio value ({result.request.currency})")


def backtest_chart(result: AnalysisResult) -> go.Figure:
    if result.backtest is None:
        raise ValueError("Backtest result is unavailable")
    figure = go.Figure()
    for label, run in [("Optimized", result.backtest.optimized), ("Equal weight", result.backtest.equal_weight)]:
        figure.add_trace(go.Scatter(x=run.equity.index.tolist(), y=run.equity.tolist(), name=label, mode="lines"))
    return _style(figure, x_title="Evaluation date", y_title=f"Wealth after costs ({result.request.currency})")

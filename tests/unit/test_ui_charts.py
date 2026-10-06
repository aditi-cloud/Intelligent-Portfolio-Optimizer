import pytest

from portfolio_optimizer.ui.charts import allocation_chart, backtest_chart, frontier_chart, scenario_chart


def test_charts_use_result_values(analysis_result):
    allocation = allocation_chart(analysis_result)
    assert list(allocation.data[0].labels) == list(analysis_result.request.tickers)
    assert list(allocation.data[0].values) == analysis_result.portfolio.weights.tolist()
    frontier = frontier_chart(analysis_result)
    assert list(frontier.data[0].x) == [p.metrics.annual_volatility * 100 for p in analysis_result.frontier.points]
    assert list(frontier.data[1].y) == [analysis_result.portfolio.metrics.expected_annual_return * 100]
    scenario = scenario_chart(analysis_result)
    assert list(scenario.data[2].y) == analysis_result.simulation.path_quantiles["p50"].tolist()
    replay = backtest_chart(analysis_result)
    assert list(replay.data[0].y) == analysis_result.backtest.optimized.equity.tolist()

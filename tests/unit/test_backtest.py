from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.errors import BacktestError
from portfolio_optimizer.evaluation.backtest import BacktestConfig, equal_weight_strategy, run_backtest


@pytest.fixture
def replay_inputs(price_dataset, data_request):
    config = AnalysisConfig(min_observations=2)
    return prepare_data(price_dataset, data_request, config), data_request, config


def test_known_equity_curve_and_drifting_weights(replay_inputs):
    prepared, request, config = replay_inputs
    result = run_backtest(prepared, request, config, BacktestConfig(2, 21, 0), strategy=equal_weight_strategy)
    # Execute on Jan 7, then earn Jan 8/9 returns. Execution-day gains excluded.
    np.testing.assert_allclose(result.equity, [100_000, 105_000, 110_000])
    np.testing.assert_allclose(result.net_returns, [0, .05, 110/105 - 1])
    np.testing.assert_allclose(result.end_of_day_weights.iloc[1], [100/210, 110/210])
    assert result.metrics.total_return == pytest.approx(.1)
    assert len(result.rebalances) == 1
    record = result.rebalances[0]
    assert record.signal_date.isoformat() == "2025-01-06"
    assert record.execution_date.isoformat() == "2025-01-07"
    assert record.signal_date < record.execution_date
    assert record.observations == 2


def test_self_financing_entry_cost(replay_inputs):
    prepared, request, config = replay_inputs
    result = run_backtest(prepared, request, config, BacktestConfig(2, 21, 100), strategy=equal_weight_strategy)
    entry_wealth = 100_000 / 1.01
    assert result.equity.iloc[0] == pytest.approx(entry_wealth)
    assert result.equity.iloc[-1] == pytest.approx(entry_wealth * 1.1)
    assert result.total_cost == pytest.approx(100_000 - entry_wealth)
    assert result.rebalances[0].cost == pytest.approx(.01 * result.rebalances[0].traded_notional)


def test_rebalancing_charges_actual_buy_sell_notional(replay_inputs):
    prepared, request, config = replay_inputs
    result = run_backtest(prepared, request, config, BacktestConfig(2, 1, 100), strategy=equal_weight_strategy)
    assert len(result.rebalances) == 2  # No unnecessary terminal trade.
    assert result.rebalances[1].turnover > 0
    for record in result.rebalances:
        assert record.cost == pytest.approx(.01 * record.traded_notional)
        assert record.signal_date < record.execution_date
    assert result.total_cost == pytest.approx(sum(r.cost for r in result.rebalances))


def test_strategy_sees_only_mature_past_returns(replay_inputs):
    prepared, request, config = replay_inputs
    seen = []
    def spy(training, window_request, analysis_config):
        seen.append((training.copy(), window_request))
        return equal_weight_strategy(training, window_request, analysis_config)
    result = run_backtest(prepared, request, config, BacktestConfig(2, 1, 0), strategy=spy)
    for (training, window_request), record in zip(seen, result.rebalances):
        assert len(training) == 2
        assert training.index[-1].date() == record.signal_date
        assert training.index[-1].date() < record.execution_date
        assert window_request.end_date == record.signal_date


def test_future_price_change_does_not_change_earlier_decisions(price_dataset, data_request):
    config = AnalysisConfig(min_observations=2)
    settings = BacktestConfig(2, 1, 0)
    original = run_backtest(prepare_data(price_dataset, data_request, config), data_request, config, settings)
    prices = price_dataset.prices.copy()
    prices.iloc[-1, 0] *= 2
    changed = run_backtest(prepare_data(replace(price_dataset, prices=prices), data_request, config), data_request, config, settings)
    for a, b in zip(original.rebalances, changed.rebalances):
        pd.testing.assert_series_equal(a.target_weights, b.target_weights)
    assert original.equity.iloc[-1] != changed.equity.iloc[-1]


def test_missing_return_rejected(price_dataset, data_request):
    config = AnalysisConfig(min_observations=2)
    prices = price_dataset.prices.copy()
    prices.iloc[1, 0] = np.nan
    prepared = prepare_data(replace(price_dataset, prices=prices), data_request, config)
    with pytest.raises(BacktestError, match="consecutive"):
        run_backtest(prepared, data_request, config, BacktestConfig(2))


def test_insufficient_test_sessions(replay_inputs):
    prepared, request, config = replay_inputs
    with pytest.raises(BacktestError, match="holding session"):
        run_backtest(prepared, request, config, BacktestConfig(4))


def test_short_training_window_rejected(replay_inputs):
    prepared, request, _ = replay_inputs
    with pytest.raises(BacktestError, match="estimator minimum"):
        run_backtest(prepared, request, AnalysisConfig(min_observations=3), BacktestConfig(2))


def test_bad_strategy_weights_rejected(replay_inputs):
    prepared, request, config = replay_inputs
    def invalid(training, request, config):
        return pd.Series([.2, .2], index=list(request.tickers))
    with pytest.raises(BacktestError, match="infeasible"):
        run_backtest(prepared, request, config, BacktestConfig(2), strategy=invalid)


def test_costs_reduce_net_terminal_wealth(replay_inputs):
    prepared, request, config = replay_inputs
    free = run_backtest(prepared, request, config, BacktestConfig(2, 1, 0), strategy=equal_weight_strategy)
    paid = run_backtest(prepared, request, config, BacktestConfig(2, 1, 100), strategy=equal_weight_strategy)
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.metrics.total_return < free.metrics.total_return


def test_replay_rejects_modified_returns(replay_inputs):
    prepared, request, config = replay_inputs
    modified = prepared.returns.copy()
    modified.iloc[-1, 0] += .1
    with pytest.raises(BacktestError, match="consecutive"):
        run_backtest(replace(prepared, returns=modified), request, config, BacktestConfig(2))


def test_ml_not_silently_replaced(replay_inputs):
    prepared, request, config = replay_inputs
    with pytest.raises(BacktestError, match="historical"):
        run_backtest(prepared, replace(request, estimator="xgboost"), config, BacktestConfig(2))


@pytest.mark.parametrize("updates", [
    {"training_observations": 1}, {"training_observations": True},
    {"rebalance_every": 0}, {"rebalance_every": 1.5},
    {"transaction_cost_bps": -1}, {"transaction_cost_bps": 10_000},
    {"transaction_cost_bps": np.nan}, {"transaction_cost_bps": True},
])
def test_invalid_backtest_configuration(updates):
    with pytest.raises(ValueError):
        BacktestConfig(**updates)

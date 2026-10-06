from decimal import Decimal

import pandas as pd
import pytest

from portfolio_optimizer.errors import OptimizationError
from portfolio_optimizer.optimization import allocate_currency


def test_budget_conservation_with_cents():
    result = allocate_currency(pd.Series([1/3] * 3, index=["A", "B", "C"]), 100)
    assert result.amounts == {"A": Decimal("33.34"), "B": Decimal("33.33"), "C": Decimal("33.33")}
    assert sum(result.amounts.values()) == result.budget
    assert result.realized_weights.sum() == pytest.approx(1)
    assert result.residual_cash == 0


def test_zero_weight_receives_no_money():
    result = allocate_currency(pd.Series([0, 1], index=["A", "B"]), Decimal("123.45"))
    assert result.amounts["A"] == 0
    assert result.amounts["B"] == Decimal("123.45")


def test_small_budget_and_tie_breaking():
    result = allocate_currency(pd.Series([.5, .5], index=["B", "A"]), .01)
    assert result.amounts == {"B": Decimal(".01"), "A": Decimal("0")}


@pytest.mark.parametrize("amount", [0, -1, .001, float("inf"), float("nan"), True])
def test_invalid_budget(amount):
    with pytest.raises(OptimizationError):
        allocate_currency(pd.Series([1], index=["A"]), amount)


@pytest.mark.parametrize("values", [[.1, .1], [-.1, 1.1]])
def test_invalid_weights(values):
    with pytest.raises(OptimizationError):
        allocate_currency(pd.Series(values, index=["A", "B"]), 100)


def test_currency_rejected():
    with pytest.raises(OptimizationError):
        allocate_currency(pd.Series([1], index=["A"]), 100, currency="EUR")

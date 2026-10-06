from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Mapping

import numpy as np
import pandas as pd

from ..errors import OptimizationError
from ..forecasting.validation import numeric_values


@dataclass(frozen=True)
class CurrencyAllocation:
    currency: str
    budget: Decimal
    amounts: Mapping[str, Decimal]
    target_weights: pd.Series
    realized_weights: pd.Series
    residual_cash: Decimal


def allocate_currency(weights: pd.Series, investment_amount: float | Decimal, *,
                      currency: str = "INR") -> CurrencyAllocation:
    """Round ideal currency allocations to cents using largest remainders.

    This is a funding plan, not executable share quantities. Cents rounding can
    slightly change realized asset weights; these are returned explicitly.
    """
    if currency not in {"INR", "USD"}:
        raise OptimizationError("Allocation currency must be INR or USD")
    if not isinstance(weights, pd.Series) or weights.empty or weights.index.has_duplicates or any(not isinstance(t, str) or not t for t in weights.index):
        raise OptimizationError("Allocation requires unique ticker weights")
    values = numeric_values(weights)
    if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, rtol=0, atol=1e-8):
        raise OptimizationError("Allocation weights must be long-only and sum to one")
    try:
        if isinstance(investment_amount, bool):
            raise InvalidOperation
        budget = Decimal(str(investment_amount))
        if not budget.is_finite() or budget <= 0 or budget != budget.quantize(Decimal("0.01")):
            raise InvalidOperation
        cents = int(budget * 100)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise OptimizationError("Budget must be positive and finite with at most two decimal places") from exc
    decimal_weights = [Decimal(str(value)) for value in values]
    total = sum(decimal_weights)
    exact = [Decimal(cents) * weight / total for weight in decimal_weights]
    rounded = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in exact]
    remainder = cents - sum(rounded)
    order = sorted(range(len(values)), key=lambda i: exact[i] - rounded[i], reverse=True)
    for index in order[:remainder]:
        rounded[index] += 1
    amounts = {ticker: Decimal(value) / 100 for ticker, value in zip(weights.index, rounded)}
    realized = pd.Series([value / cents for value in rounded], index=weights.index, name="realized_weight")
    return CurrencyAllocation(currency, budget, amounts, weights.copy(), realized, Decimal("0.00"))

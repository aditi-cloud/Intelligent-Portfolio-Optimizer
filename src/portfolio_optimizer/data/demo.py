"""Fixed synthetic demo universe shared by examples and the dashboard."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .contracts import PriceDataset

DEMO_TICKERS = ("DEMO-A", "DEMO-B", "DEMO-C")
DEMO_START = date(2025, 1, 1)
DEMO_END = date(2025, 9, 10)


def demo_dataset() -> PriceDataset:
    dates = pd.bdate_range(DEMO_START, DEMO_END)
    rng = np.random.default_rng(42)
    returns = rng.normal([.0004, .0002, .0003], [.015, .01, .012], size=(len(dates) - 1, 3))
    values = 100 * np.vstack([np.ones(3), np.cumprod(1 + returns, axis=0)])
    return PriceDataset(pd.DataFrame(values, index=dates, columns=DEMO_TICKERS),
                        dict.fromkeys(DEMO_TICKERS, "INR"), "synthetic_demo",
                        datetime.now(timezone.utc))

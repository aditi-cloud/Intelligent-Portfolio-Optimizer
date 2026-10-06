from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json

import streamlit as st

from ..config import AnalysisConfig
from ..contracts import PortfolioRequest
from ..data.demo import DEMO_END, DEMO_START, DEMO_TICKERS
from ..evaluation.backtest import BacktestConfig
from ..services.analysis import AnalysisOptions

DEFAULT_LIVE_TICKERS = ("BHARTIARTL.NS", "SBIN.NS", "BEL.NS")
_PREVIOUS_LIVE_TICKERS = ("RELIANCE.NS", "TCS.NS", "HDFCBANK.NS")


@dataclass(frozen=True)
class DashboardInputs:
    mode: str
    request: PortfolioRequest
    config: AnalysisConfig
    options: AnalysisOptions

    def identity(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def read_inputs(today: date) -> tuple[DashboardInputs, bool]:
    with st.sidebar:
        st.header("Build your portfolio")
        mode = st.radio("Data source", ["Demo (synthetic)", "Yahoo Finance"], key="data_source")
        demo = mode == "Demo (synthetic)"
        if demo:
            st.caption("Fixed synthetic prices. Explore the dashboard without a market-data connection.")
            tickers = tuple(st.multiselect("Stocks", DEMO_TICKERS, default=list(DEMO_TICKERS), key="demo_tickers"))
            currency = "INR"
        else:
            default_symbols = ", ".join(DEFAULT_LIVE_TICKERS)
            # Migrate the old automatic basket once, while preserving custom selections.
            if st.session_state.get("live_defaults_version") != 2:
                previous = st.session_state.get("live_tickers")
                if previous is not None and tuple(t.strip().upper() for t in previous.split(",")) == _PREVIOUS_LIVE_TICKERS:
                    st.session_state["live_tickers"] = default_symbols
                st.session_state["live_defaults_version"] = 2
            text = st.text_input("Stock symbols", default_symbols, key="live_tickers",
                                 help="Comma-separated Yahoo symbols. Use .NS for NSE equities; select stocks in one currency.")
            st.caption("Example basket: Bharti Airtel, State Bank of India and Bharat Electronics. Selected using positive historical results through 6 October 2026.")
            tickers = tuple(value.strip().upper() for value in text.split(",") if value.strip())
            currency = st.selectbox("Currency", ["INR", "USD"], key="live_currency")
        amount = st.number_input(f"Investment amount ({currency})", min_value=0.0, max_value=1_000_000_000.0,
                                 value=100_000.0, step=1_000.0, format="%.2f", key="investment_amount")
        risk = st.select_slider("Risk appetite", ["low", "medium", "high"], value="medium", key="risk_level")
        cap = st.slider("Maximum weight per stock (%)", 1, 100, 100, key="weight_cap",
                        help="This cap applies to target weights. The selected stocks must be able to hold the full investment.") / 100
        horizon = st.number_input("Scenario horizon (trading days)", 1, 252, 21, key="horizon_days")
        with st.expander("Analysis settings"):
            date_key = "demo" if demo else "live"
            start = st.date_input("History start", DEMO_START if demo else today - timedelta(days=730),
                                  key=f"{date_key}_start", min_value=DEMO_START if demo else date(1990, 1, 1),
                                  max_value=DEMO_END if demo else today)
            end = st.date_input("History end", DEMO_END if demo else today, key=f"{date_key}_end",
                                min_value=DEMO_START if demo else date(1990, 1, 1), max_value=DEMO_END if demo else today)
            risk_free = st.number_input("Annual risk-free rate (%)", min_value=-10.0, max_value=50.0,
                                        value=4.0, step=.25, key="risk_free_rate") / 100
            simulation = st.checkbox("Include Monte Carlo scenarios", value=True, key="include_simulation")
            scenarios = st.selectbox("Number of scenarios", [500, 1_000, 2_000, 5_000], index=1, key="scenarios")
            replay = st.checkbox("Include historical backtest", value=True, key="include_backtest")
            training = st.number_input("Training sessions", 60, 504, 60, key="training_sessions")
            rebalance = st.number_input("Rebalance every (sessions)", 1, 126, 21, key="rebalance_sessions")
            cost = st.number_input("Transaction cost (bps)", 0.0, 100.0, 10.0, step=1.0, key="transaction_cost")
            seed = st.number_input("Scenario seed", 0, 2_147_483_647, 42, key="random_seed")
        submitted = st.button("Analyze portfolio", type="primary", width="stretch", key="analyze")
        st.caption("Historical return estimates · long-only portfolios · one currency")
    request = PortfolioRequest(tickers, amount, start, end, currency, horizon, risk, cap)
    config = AnalysisConfig(annual_risk_free_rate=risk_free, random_seed=seed)
    options = AnalysisOptions(frontier_points=20, include_simulation=simulation, simulation_scenarios=scenarios,
                              backtest=BacktestConfig(training, rebalance, cost) if replay else None)
    return DashboardInputs(mode, request, config, options), submitted

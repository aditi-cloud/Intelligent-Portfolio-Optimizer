# Intelligent Portfolio Optimizer

A modular portfolio analysis application. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the component boundaries and staged implementation plan.

## Implemented

Stages 1–7: request validation and configuration, a price-provider interface,
Yahoo Finance adapter, versioned local JSON cache, adjusted-price validation,
and aligned daily returns with a data-quality report, historical return
estimation, Ledoit-Wolf covariance, expected portfolio metrics and an
equal-weight benchmark, constrained optimization, an efficient frontier and
currency allocations, walk-forward backtests, realized performance metrics and
correlated Monte Carlo scenarios, an application service, downloadable reports
and a Streamlit dashboard. Request validation has
no runtime dependencies; the data component uses NumPy, pandas and yfinance.
The historical baseline is integrated. Advanced ML estimators are the next stage.

## Setup

Use Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data,analysis,optimization,ui]'
python -m pytest
```

This workspace now has a Python 3.12 virtual environment. Run the existing
tests directly with `.venv/bin/python -m pytest`, or activate `.venv` first.
The underlying managed interpreter is currently stored in
`/private/tmp/portfolio-python`; if temporary files are cleared, recreate the
environment using a persistent Python installation. `/usr/bin/python3` still
requires missing Apple developer tools.

## Request example

```python
from datetime import date
from portfolio_optimizer import PortfolioRequest, validate_request

request = validate_request(
    PortfolioRequest(
        tickers=("RELIANCE.NS", "TCS.NS"),
        investment_amount=100_000,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 1, 1),
        max_asset_weight=0.7,
    ),
    today=date.today(),  # Pass the user's local date at the UI boundary.
)
```

Validation checks request structure; it cannot establish ticker existence,
asset currency or sufficient trading history without the data component.
Risk aversion values are initial configurable settings, not calibrated risk
guarantees. The risk-free rate defaults to zero and should be configured for
the analysis currency when evaluating Sharpe ratios.

## Fetch and prepare prices

```python
from pathlib import Path
from portfolio_optimizer import AnalysisConfig
from portfolio_optimizer.data import prepare_data
from portfolio_optimizer.data.cache import CachedPriceProvider
from portfolio_optimizer.data.yahoo import YahooPriceProvider

provider = CachedPriceProvider(YahooPriceProvider(), Path(".cache/prices"))
dataset = provider.fetch_prices(request)
prepared = prepare_data(dataset, request, AnalysisConfig(min_observations=60))
print(prepared.returns.tail())
print(prepared.quality)
```

The live example requires network access. The adapter explicitly requests
adjusted daily closes, translates our inclusive end date to Yahoo's exclusive
end date, and reads asset currencies from provider metadata. See the
[official download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
Mixed/unknown currencies, invalid prices and missing tickers are rejected.
No forward filling is performed: a missing price excludes both that day's
return and the next available row's return. Weekend/holiday rows are not
inserted. Since exchange calendars are not yet integrated, an entirely absent
trading session cannot currently be distinguished from a market holiday.

The cache expires after six hours. Its key includes symbols, date window,
currency, provider version and adjustment policy. Corrupt or expired entries
are fetched again; provider failures propagate rather than serving stale data.
Cache directory write errors are translated to a `ProviderError` by the
application service. Cache writes currently remain necessary when using the
cache adapter; analysis does not silently retry without caching.

## Testing

```sh
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/unit/test_preparation.py
.venv/bin/python -m pytest -m integration
```

The suite currently has 307 passing tests, including Streamlit acceptance tests. All data-provider calls in tests are
mocked; there are no live Yahoo calls. `tests/fixtures/prices.csv` contains
synthetic prices with hand-calculable returns, not observed market prices.
The integration test exercises the actual Yahoo adapter, JSON cache and price
preparation using an injected download response. Live Yahoo fetching was separately verified on 6 October 2026 for the candidate Indian stocks; these network checks are outside the deterministic test suite.

## Estimate returns, risk and the equal-weight reference

```python
from portfolio_optimizer.forecasting import estimate_market
from portfolio_optimizer.evaluation import equal_weight_metrics

config = AnalysisConfig(annual_risk_free_rate=0.04)
estimates = estimate_market(prepared, request, config)
benchmark = equal_weight_metrics(
    estimates, annual_risk_free_rate=config.annual_risk_free_rate,
)
print(estimates.expected_returns)
print(estimates.covariance)
print(benchmark)
```

The baseline uses annual arithmetic mean returns (`daily_mean * trading_days`)
and annual covariance (`daily_covariance * trading_days`). They are annualized
daily moments, not compounded one-year wealth forecasts. The scaling assumes
the daily moments remain applicable and does not model serial correlation.
Volatility is the square root of portfolio variance; expected Sharpe uses the
configured annual risk-free rate. All metrics are labeled **expected** and do
not demonstrate realized investment performance. A zero-volatility portfolio
returns an undefined Sharpe (`None`) with an explanation.

Covariance uses the centered
[scikit-learn Ledoit-Wolf estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html),
with its population normalization and recorded shrinkage coefficient. Shrinkage
can give a constant asset positive estimated variance in a mixed basket.
The estimates record ticker order, data cutoff, observation count and numerical
conventions. Invalid matrices and misaligned assets are rejected. Selecting an
unimplemented ML estimator raises an error rather than substituting the baseline.

Run a complete offline example using deterministic synthetic prices:

```sh
.venv/bin/python examples/baseline.py
```

The example prepares 180 daily returns, estimates market moments and prints the
equal-weight benchmark, optimized weights, allocations, frontier, historical
replay and simulation summaries as JSON. It uses no live market calls and makes no
claims about observed investment performance.

## Optimize and allocate

```python
from portfolio_optimizer.optimization import (
    optimize_portfolio, build_frontier, allocate_currency,
)

optimized = optimize_portfolio(estimates, request, config)
frontier = build_frontier(
    estimates, max_asset_weight=request.max_asset_weight,
    annual_risk_free_rate=config.annual_risk_free_rate,
)
allocation = allocate_currency(
    optimized.weights, request.investment_amount, currency=request.currency,
)
print(optimized.weights)
print(allocation.amounts)
```

The optimizer maximizes `expected_return - risk_aversion * variance` with
nonnegative weights, full investment and the requested per-asset cap. The
objective does not maximize Sharpe ratio. Low/medium/high risk settings map
to configurable risk-aversion values. Amount scales allocations rather than
changing normalized optimal weights.

The default solver is CLARABEL through CVXPY. Only `optimal` status is accepted;
inaccurate or infeasible results and solver failures raise `OptimizationError`.
See [CVXPY's solver status documentation](https://www.cvxpy.org/tutorial/intro/).
Weights are independently checked, then tiny residuals are removed by projection
onto the capped simplex. Numerically tiny negative covariance eigenvalues are
clipped for solving and recorded in result warnings; materially invalid
covariances fail validation.

The frontier traces minimum variance portfolios from the global minimum
variance return to the maximum feasible return under the same asset cap.
Targets are lower bounds; a fixed portfolio or an identical-return basket
produces a single point. Failed solves propagate instead of silently dropping
frontier points.

Currency allocations use Decimal amounts and largest-remainder rounding so the
allocations sum exactly to the budget. Budgets require at most two decimal
places. Target and realized weights are both returned because currency rounding
can slightly change weights and exceed an ideal cap by a rounding amount.
These are funding allocations; whole-share quantities and executable quotes
are not implemented yet.

## Replay historical performance

```python
from portfolio_optimizer.evaluation.backtest import BacktestConfig, compare_backtests

comparison = compare_backtests(
    prepared, request, config,
    BacktestConfig(training_observations=60, rebalance_every=21, transaction_cost_bps=10),
)
print(comparison.optimized.metrics)
print(comparison.equal_weight.metrics)
```

Both strategies use the same asset universe, evaluation dates, rebalance schedule
and fee rate. Each signal uses a rolling window ending at close t. It executes
at the next observed close t+1 and first earns the return from t+1 to t+2.
Existing holdings earn the execution-day return before rebalancing. The initial
execution-day record contains entry fees and zero cash return. No terminal
liquidation is charged. Holdings drift between rebalances; the asset cap applies
to rebalance targets, not to weights after subsequent price moves.

Fees apply to actual buy plus sell notional. Each rebalance solves the
self-financing equation `cost = rate * sum(abs(new_holdings - old_holdings))`,
then invests remaining wealth. At entry this gives invested wealth
`budget / (1 + rate)`. Records include training start, signal/execution dates,
weights before and after the trade, notional, turnover and cost.

The replay uses fractional holdings and adjusted-close total-return accounting.
It does not model intraday prices, slippage or taxes. Complete sequential price
rows are required; histories with missing prices/removed return observations
are rejected. Fully absent market sessions remain indistinguishable from
holidays without an exchange calendar.

Realized metrics include compounded total return, annualized geometric and
arithmetic returns, sample volatility, realized Sharpe and negative maximum
drawdown. Sharpe uses daily excess returns against a compounded daily risk-free
rate. Drawdown includes starting wealth before entry fees. Annualization uses
session count. Short replay windows can produce unstable annualized statistics.
ML backtesting is deferred until causal feature/label maturity rules are added.

## Simulate portfolio scenarios

```python
from portfolio_optimizer.evaluation.simulation import simulate_portfolio

simulation = simulate_portfolio(
    estimates, optimized.weights, request.investment_amount,
    horizon_days=request.horizon_days, scenarios=2000, seed=config.random_seed,
)
print(simulation.terminal_quantiles)
print(simulation.model_loss_probability)
```

The simulator matches daily simple-return means and covariance with correlated
lognormal gross returns. For gross means `g`, the log covariance is
`log(1 + daily_covariance / outer(g, g))`; the log mean is
`log(g) - diagonal(log_covariance)/2`. It rejects moments that cannot yield a
valid positive semidefinite log covariance. Only numerical negative eigenvalues
are corrected, with warnings. The lognormal construction follows the
[SciPy lognormal parameter convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html).

Scenarios assume independent sessions, fixed moments and a buy-and-hold portfolio
whose asset weights drift. Fees and taxes are excluded. Results include full
wealth paths, p05/p25/p50/p75/p95 path and terminal quantiles, terminal asset
returns, seed, input weights, data cutoff and assumptions. Loss frequency is
conditional on these model assumptions. Stored scenario values are limited to
10 million to bound memory use; overflow and underflow to nonpositive wealth
raise an explicit error.

Tests check known equity curves, delayed execution, future-data perturbations,
drifting weights, self-financing costs and matched moments. Stochastic sample
checks use sampling-error bounds rather than requiring exact simulated moments.

## Analyze through the application service

```python
from portfolio_optimizer.services import AnalysisOptions, PortfolioAnalysisService

service = PortfolioAnalysisService(provider, config)
result = service.analyze(
    request,
    AnalysisOptions(
        frontier_points=20,
        simulation_scenarios=2000,
        backtest=BacktestConfig(training_observations=60, rebalance_every=21, transaction_cost_bps=10),
    ),
    today=date.today(),
)
```

The service validates inputs, fetches prices once, runs core components and
returns an `AnalysisResult`. Streamlit will consume this result directly.
Backtesting is disabled by default; simulation is enabled by default. Options
control whether either evaluation is requested. Known optional evaluation
failures preserve the core result and appear in `result.issues`; setting
`allow_partial_evaluation=False` makes those failures fatal. Core preparation,
estimation, optimization and frontier failures always propagate. Unexpected
programming errors are never silently converted into missing results.
No estimator fallback is performed. The caller supplies the user's local date
when rejecting future historical windows.

## Downloadable reports

```python
from portfolio_optimizer.services import report_html, report_json, report_bundle

html_bytes = report_html(result)
json_bytes = report_json(result)
zip_bytes = report_bundle(result)
```

Exports operate on the result alone, without market calls or recalculation.
The printable HTML report includes allocation and comparison tables, inline SVG
frontier/replay/scenario charts, data quality, assumptions and explicit component
issues. It requires no external scripts, stylesheets or images. Provider text
and warnings are escaped. Undefined Sharpe values are represented as null in
JSON and as "Undefined" in HTML; non-finite numeric report data is rejected.

The ZIP contains `report.html`, `analysis.json`, `allocations.csv` and
`frontier.csv`, plus `backtest.csv` and `scenario_quantiles.csv` when available.
JSON includes inputs, configuration, provenance, estimates, results and rebalance
records. Simulation exports include quantile summaries rather than every path
to keep downloads manageable. Export functions return bytes for download buttons;
report generation failures leave the original analysis result available.

Generate reviewable sample files through the complete offline service:

```sh
.venv/bin/python examples/service_report.py
```

Files are written under `artifacts/demo-report/`: `report.html`, `analysis.json`
and `report.zip`. All figures use synthetic prices. Open the HTML report in a
browser or print it to PDF. Generated artifacts are excluded from version control.

## Streamlit dashboard

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`, keep **Demo (synthetic)** selected, and click
**Analyze portfolio**. The demo uses the same fixed synthetic universe as the
offline examples and makes no network calls. You can select subsets of the
three demo stocks, adjust the budget, risk setting, weight cap and horizon,
and configure simulation/backtest settings.

**Yahoo Finance** mode accepts comma-separated stock symbols and a selected
currency. Fetching requires network access; live fetching was checked on 6 October 2026 for the current Indian-stock defaults. The adapter rejects missing or mixed currencies. This mode reuses
the local price cache. Dates use Asia/Kolkata at the UI boundary to reject future
historical windows. Only the historical estimator is currently exposed.

The dashboard labels the arithmetic estimate "Historical mean return (annualized)"
and explicitly identifies the historical baseline. Nonpositive means and means
below the configured reference rate receive a prominent assessment explaining
the full-investment constraint. Reports include the same assessment. Positive
historical averages are not treated as validated forecasts. Risk appetite
changes the variance penalty; it does not imply positive returns. Cash allocation
is not yet supported, so a constrained stock allocation may still have a negative
historical mean. Calculation values are preserved rather than clipped or replaced.

The dashboard has allocation, risk/return, scenario, backtest and data/assumption
tabs. Plotly charts display existing result data. Report buttons download HTML,
JSON or ZIP. Analysis runs only on explicit submission; ordinary reruns reuse
the result and prepared download bytes in session state. Any changed analysis
input clears the previous result and downloads until a new submission. Failed
submissions do not leave stale results visible. Report generation can be retried
without rerunning financial analysis.

The app follows Streamlit's
[download button](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)
behavior to prevent a download click from triggering a rerun. Acceptance tests
use [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
to check demo results, input invalidation, provider errors, optional evaluation,
report retries and rerun reuse:

```sh
.venv/bin/python -m pytest -m ui
```

Local server startup and its health endpoint were verified. A headless Chrome
check submitted the demo and captured `artifacts/dashboard-preview.png`.

The Yahoo default basket is now BHARTIARTL.NS, SBIN.NS and BEL.NS. Adjusted prices were checked through 6 October 2026 over the default two-year window. All three had positive annualized historical means. With a 60-session training window, 21-session rebalancing, 10 bps costs and 4% configured reference rate, optimized net backtest geometric returns were approximately 10.46% annualized for medium risk and 8.26% for high risk. This example basket was selected after inspecting the same historical period, so these are descriptive results with selection hindsight. Changing the window or constraints changes the results. A live-data report is available in artifacts/live-defaults/. The former automatic basket is migrated once; custom stock selections are preserved.

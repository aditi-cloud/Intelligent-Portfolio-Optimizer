"""Application boundary; each analysis fetches once and produces one result."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

from ..config import AnalysisConfig
from ..contracts import EstimatorChoice, PortfolioRequest
from ..data.contracts import PreparedData
from ..data.preparation import prepare_data
from ..data.provider import PriceProvider
from ..errors import PortfolioError, ProviderError, RequestValidationError
from ..evaluation.backtest import BacktestComparison, BacktestConfig, compare_backtests
from ..evaluation.metrics import PortfolioMetrics, equal_weight_metrics
from ..evaluation.assessment import PortfolioAssessment, assess_baseline
from ..evaluation.simulation import SimulationResult, simulate_portfolio
from ..forecasting import MarketEstimates, estimate_market
from ..optimization import CurrencyAllocation, EfficientFrontier, OptimizationResult, allocate_currency, build_frontier, optimize_portfolio
from ..validation import validate_request


@dataclass(frozen=True)
class AnalysisOptions:
    frontier_points: int = 20
    include_simulation: bool = True
    simulation_scenarios: int = 2_000
    backtest: BacktestConfig | None = None
    allow_partial_evaluation: bool = True

    def __post_init__(self):
        if type(self.frontier_points) is not int or self.frontier_points < 2:
            raise ValueError("Frontier point count must be an integer >= 2")
        if type(self.simulation_scenarios) is not int or self.simulation_scenarios < 2:
            raise ValueError("Simulation scenario count must be an integer >= 2")
        if type(self.include_simulation) is not bool or type(self.allow_partial_evaluation) is not bool:
            raise ValueError("Evaluation flags must be booleans")
        if self.backtest is not None and not isinstance(self.backtest, BacktestConfig):
            raise ValueError("Backtest options must be a BacktestConfig")


@dataclass(frozen=True)
class ComponentIssue:
    component: str
    message: str


@dataclass(frozen=True)
class AnalysisResult:
    request: PortfolioRequest
    config: AnalysisConfig
    options: AnalysisOptions
    generated_at: datetime
    prepared: PreparedData
    estimates: MarketEstimates
    portfolio: OptimizationResult
    benchmark: PortfolioMetrics
    frontier: EfficientFrontier
    allocation: CurrencyAllocation
    simulation: SimulationResult | None
    backtest: BacktestComparison | None
    issues: tuple[ComponentIssue, ...]
    warnings: tuple[str, ...]
    assessment: PortfolioAssessment


class PortfolioAnalysisService:
    def __init__(self, provider: PriceProvider, config: AnalysisConfig | None = None, *, clock: Callable | None = None):
        self.provider = provider
        self.config = config or AnalysisConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def analyze(self, request: PortfolioRequest, options: AnalysisOptions | None = None, *,
                today: date | None = None) -> AnalysisResult:
        request = validate_request(request, today=today)
        options = AnalysisOptions() if options is None else options
        if not isinstance(options, AnalysisOptions):
            raise RequestValidationError("Expected AnalysisOptions")
        if request.estimator is not EstimatorChoice.HISTORICAL:
            raise RequestValidationError("Only historical estimation is currently available")
        if options.include_simulation and options.simulation_scenarios * (request.horizon_days + 1 + len(request.tickers)) > 10_000_000:
            raise RequestValidationError("Simulation exceeds the stored-value limit")
        if options.backtest and options.backtest.training_observations < self.config.min_observations:
            raise RequestValidationError("Backtest training window is shorter than the estimator minimum")
        # Check currency precision before spending time on a fetch or solve.
        from ..evaluation.metrics import equal_weights
        allocate_currency(equal_weights(request.tickers), request.investment_amount, currency=request.currency)
        try:
            dataset = self.provider.fetch_prices(request)
        except OSError as exc:
            raise ProviderError("Price provider or local cache I/O failed") from exc
        prepared = prepare_data(dataset, request, self.config)
        estimates = estimate_market(prepared, request, self.config)
        portfolio = optimize_portfolio(estimates, request, self.config)
        benchmark = equal_weight_metrics(estimates, annual_risk_free_rate=self.config.annual_risk_free_rate)
        frontier = build_frontier(estimates, max_asset_weight=request.max_asset_weight,
                                  points=options.frontier_points, annual_risk_free_rate=self.config.annual_risk_free_rate)
        allocation = allocate_currency(portfolio.weights, request.investment_amount, currency=request.currency)
        issues = []
        simulation, backtest = None, None
        if options.include_simulation:
            try:
                simulation = simulate_portfolio(estimates, portfolio.weights, request.investment_amount,
                                                 horizon_days=request.horizon_days, scenarios=options.simulation_scenarios,
                                                 seed=self.config.random_seed)
            except PortfolioError as exc:
                if not options.allow_partial_evaluation:
                    raise
                issues.append(ComponentIssue("simulation", str(exc)))
        if options.backtest:
            try:
                backtest = compare_backtests(prepared, request, self.config, options.backtest)
            except PortfolioError as exc:
                if not options.allow_partial_evaluation:
                    raise
                issues.append(ComponentIssue("backtest", str(exc)))
        generated = self._clock()
        if not isinstance(generated, datetime) or generated.utcoffset() is None:
            raise ValueError("Analysis clock must return a timezone-aware datetime")
        warnings = list(portfolio.warnings + portfolio.metrics.warnings + benchmark.warnings)
        if prepared.quality.excluded_return_dates:
            warnings.append("Incomplete price rows excluded from estimation; see the data-quality report")
        if simulation:
            warnings.extend(simulation.warnings)
        if backtest:
            warnings.extend(backtest.optimized.metrics.warnings + backtest.equal_weight.metrics.warnings)
        return AnalysisResult(request, self.config, options, generated, prepared, estimates, portfolio,
                              benchmark, frontier, allocation, simulation, backtest, tuple(issues),
                              tuple(dict.fromkeys(warnings)), assess_baseline(estimates, portfolio.metrics))

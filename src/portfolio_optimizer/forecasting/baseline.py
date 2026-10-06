from ..config import AnalysisConfig
from ..contracts import EstimatorChoice, PortfolioRequest
from ..data.contracts import PreparedData
from ..errors import EstimationError
from ..validation import validate_request
from .contracts import MarketEstimates
from .covariance import estimate_covariance
from .returns import estimate_returns
from .validation import validate_returns


def estimate_market(prepared: PreparedData, request: PortfolioRequest,
                    config: AnalysisConfig | None = None) -> MarketEstimates:
    """Build the historical baseline without silently substituting an ML model."""
    config = config or AnalysisConfig()
    request = validate_request(request)
    if request.estimator is not EstimatorChoice.HISTORICAL:
        raise EstimationError("Only the historical estimator is implemented")
    returns = validate_returns(prepared.returns, config)
    if tuple(returns.columns) != request.tickers:
        raise EstimationError("Prepared assets do not match request order")
    if returns.index[0].date() < request.start_date or returns.index[-1].date() > request.end_date:
        raise EstimationError("Returns fall outside requested estimation window")
    covariance = estimate_covariance(returns, config)
    return MarketEstimates(
        expected_returns=estimate_returns(returns, config), covariance=covariance.annual_covariance,
        cutoff=returns.index[-1].date(), observations=len(returns),
        trading_days_per_year=config.trading_days_per_year, shrinkage=covariance.shrinkage,
    )

"""Errors the application service can translate into user-facing messages."""


class PortfolioError(Exception):
    """Base class for expected application failures."""


class RequestValidationError(PortfolioError, ValueError):
    """A request violates the supported input contract."""


class DataQualityError(PortfolioError):
    """Prices cannot support the requested analysis."""


class ProviderError(PortfolioError):
    """An external data provider failed."""


class EstimationError(PortfolioError):
    """An estimator cannot produce valid estimates."""


class OptimizationError(PortfolioError):
    """The optimizer cannot produce a feasible portfolio."""


class BacktestError(PortfolioError):
    """Historical replay cannot be performed under the requested assumptions."""


class SimulationError(PortfolioError):
    """Estimates or resource limits cannot support the requested scenarios."""


class ReportError(PortfolioError):
    """A result could not be exported to a report."""

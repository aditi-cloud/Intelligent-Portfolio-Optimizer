"""Portable report exports built solely from an existing AnalysisResult."""

import csv
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from html import escape
from io import BytesIO, StringIO
import json
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ..errors import ReportError
from .analysis import AnalysisResult
from .report_charts import line_chart


def _safe(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, ".2f")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return {"index": [_safe(item) for item in value.index], "columns": list(value.columns),
                "values": [[_safe(item) for item in row] for row in value.to_numpy()]}
    if is_dataclass(value):
        return {field.name: _safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def report_data(result: AnalysisResult) -> dict:
    source = result.prepared.source
    data = {
        "report_version": 1, "generated_at": result.generated_at,
        "request": result.request, "configuration": result.config, "options": result.options,
        "data": {"provider": source.provider, "retrieved_at": source.retrieved_at,
                 "adjustment_policy": source.adjustment_policy, "currencies": source.currencies,
                 "quality": result.prepared.quality},
        "estimates": result.estimates, "portfolio": result.portfolio,
        "equal_weight_expected_metrics": result.benchmark, "frontier": result.frontier,
        "allocation": result.allocation, "issues": result.issues, "warnings": result.warnings,
        "assessment": result.assessment,
        "simulation": None, "backtest": result.backtest,
    }
    if result.simulation:
        sim = result.simulation
        data["simulation"] = {
            "horizon_days": sim.horizon_days, "scenarios": sim.scenarios, "seed": sim.seed,
            "initial_wealth": sim.initial_wealth, "weights": sim.weights, "estimation_cutoff": sim.estimation_cutoff,
            "trading_days_per_year": sim.trading_days_per_year, "terminal_quantiles": sim.terminal_quantiles,
            "mean_terminal_wealth": sim.mean_terminal_wealth, "model_loss_probability": sim.model_loss_probability,
            "path_quantiles": sim.path_quantiles, "assumptions": sim.assumptions, "warnings": sim.warnings,
        }
    return _safe(data)


def report_json(result: AnalysisResult) -> bytes:
    try:
        return json.dumps(report_data(result), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ReportError("Result contains data that cannot be serialized to a report") from exc


def _table(headers, rows):
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _percent(value):
    return f"{value:.2%}"


def _sharpe(value):
    return "Undefined" if value is None else f"{value:.3f}"


def report_html(result: AnalysisResult) -> bytes:
    """Self-contained, printable HTML with escaped text and inline SVG charts."""
    try:
        report_json(result)  # Reject non-finite report data before formatting.
        sections = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Portfolio analysis</title>',
                    '<style>body{font:16px system-ui,sans-serif;max-width:1000px;margin:36px auto;padding:0 20px;color:#172033}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #cbd5e1;padding:9px;text-align:left}th{background:#f1f5f9}figure{margin:24px 0}figcaption{font-weight:600}svg{width:100%;height:auto}svg text{font:12px system-ui}li{margin:6px 0}@media print{figure,table{break-inside:avoid}}</style></head><body>',
                    '<h1>Intelligent Portfolio Optimizer</h1>',
                    f'<p>Generated {escape(result.generated_at.isoformat())}. Prices from {escape(result.prepared.source.provider)}, retrieved {escape(result.prepared.source.retrieved_at.isoformat())}.</p>',
                    f'<h2>{escape(result.assessment.headline)}</h2><p>{escape(result.assessment.explanation)}</p>',
                    '<p>Current model: historical baseline. Future-return forecasting is not available in this version.</p>',
                    '<h2>Inputs and assumptions</h2>',
                    _table(["Setting", "Value"], [
                        ("Currency / budget", f"{result.request.currency} {result.allocation.budget:.2f}"),
                        ("Historical window", f"{result.request.start_date} to {result.request.end_date}"),
                        ("Risk setting / weight cap", f"{result.request.risk_level.value} / {result.request.max_asset_weight:.2%}"),
                        ("Annualization / risk-free rate", f"{result.config.trading_days_per_year} sessions / {result.config.annual_risk_free_rate:.2%}"),
                        ("Estimation cutoff / observations", f"{result.estimates.cutoff} / {result.estimates.observations}"),
                        ("Methods", f"{result.estimates.return_method}; {result.estimates.covariance_method}"),
                        ("Price policy", result.prepared.source.adjustment_policy),
                        ("Solver / status", f"{result.portfolio.solver} / {result.portfolio.status}"),
                    ]),
                    '<p>Historical baseline metrics use annualized daily arithmetic moments. The objective balances estimated return against variance and requires full investment. Currency allocations are a funding plan; whole-share execution is not included.</p>',
                    '<h2>Allocation</h2>',
                    _table(["Ticker", "Target weight", "Currency amount", "Rounded weight"],
                           [(ticker, _percent(result.portfolio.weights[ticker]), f"{result.allocation.amounts[ticker]:.2f}",
                             _percent(result.allocation.realized_weights[ticker])) for ticker in result.request.tickers]),
                    '<h2>Historical baseline comparison</h2>',
                    _table(["Portfolio", "Historical mean return (annualized)", "Historical volatility (annualized)", "Historical Sharpe"],
                           [(name, _percent(metrics.expected_annual_return), _percent(metrics.annual_volatility), _sharpe(metrics.expected_sharpe_ratio))
                            for name, metrics in [("Optimized", result.portfolio.metrics), ("Equal weight", result.benchmark)]]),
                    line_chart("Efficient Frontier", {
                        "Frontier": [(p.metrics.annual_volatility * 100, p.metrics.expected_annual_return * 100) for p in result.frontier.points],
                        "Optimized": [(result.portfolio.metrics.annual_volatility * 100, result.portfolio.metrics.expected_annual_return * 100)],
                        "Equal weight": [(result.benchmark.annual_volatility * 100, result.benchmark.expected_annual_return * 100)],
                    }, x_label="Historical volatility (annualized, %)", y_label="Historical mean return (annualized, %)"),
                    '<h2>Data quality</h2>', _table(["Statistic", "Value"], [
                        ("Input price rows", result.prepared.quality.input_price_rows),
                        ("Complete price rows", result.prepared.quality.complete_price_rows),
                        ("Aligned return observations", result.prepared.quality.observations),
                        ("Missing prices by ticker", dict(result.prepared.quality.missing_prices_by_ticker)),
                        ("Excluded return dates", ", ".join(result.prepared.quality.excluded_return_dates) or "None"),
                    ])]
        if result.backtest:
            runs = [result.backtest.optimized, result.backtest.equal_weight]
            sections.extend(['<h2>Realized historical replay</h2>',
                             _table(["Strategy", "Total return", "Realized Sharpe", "Max drawdown", "Costs"],
                                    [(run.name, _percent(run.metrics.total_return), _sharpe(run.metrics.realized_sharpe_ratio),
                                      _percent(run.metrics.max_drawdown), f"{run.total_cost:.2f}") for run in runs]),
                             f'<p>Evaluation sessions: {runs[0].equity.index[0].date()} to {runs[0].equity.index[-1].date()}. Training window: {runs[0].settings.training_observations}; rebalance every {runs[0].settings.rebalance_every} sessions; fees {runs[0].settings.transaction_cost_bps:g} bps.</p>',
                             line_chart("Historical wealth after costs", {run.name: list(enumerate(run.equity.tolist())) for run in runs},
                                        x_label="Evaluation session", y_label=f"Wealth ({result.request.currency})"),
                             "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in runs[0].assumptions) + "</ul>"])
        else:
            sections.append('<h2>Historical replay</h2><p>' + ("Unavailable; see component issues." if result.options.backtest else "Not requested.") + '</p>')
        if result.simulation:
            sim = result.simulation
            sections.extend(['<h2>Model-based scenarios</h2>',
                             f'<p>Horizon: {sim.horizon_days} trading days; scenarios: {sim.scenarios}; seed: {sim.seed}. Model loss frequency: {sim.model_loss_probability:.2%}.</p>',
                             _table(["Terminal quantile", f"Wealth ({result.request.currency})"], [(key, f"{value:.2f}") for key, value in sim.terminal_quantiles.items()]),
                             line_chart("Simulated wealth quantiles", {label: list(enumerate(sim.path_quantiles[label].tolist())) for label in ["p05", "p50", "p95"]},
                                        x_label="Trading day", y_label=f"Wealth ({result.request.currency})"),
                             "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in sim.assumptions) + "</ul>"])
        else:
            sections.append('<h2>Scenarios</h2><p>' + ("Unavailable; see component issues." if result.options.include_simulation else "Not requested.") + '</p>')
        sections.extend(['<h2>Warnings and component issues</h2>', '<ul>' + "".join(
            f'<li>{escape(message)}</li>' for message in list(result.warnings) + [f"{issue.component}: {issue.message}" for issue in result.issues]) + '</ul>',
            '<p>Educational decision support. Historical replay and conditional model scenarios do not guarantee future outcomes.</p></body></html>'])
        return "".join(sections).encode("utf-8")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ReportError("HTML report could not be generated") from exc


def _csv(headers, rows) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def report_bundle(result: AnalysisResult) -> bytes:
    """Create a ZIP for download without writing to disk or refetching data."""
    files = {"report.html": report_html(result), "analysis.json": report_json(result),
             "allocations.csv": _csv(["ticker", "target_weight", "amount", "currency", "rounded_weight"],
                                     [(t, result.portfolio.weights[t], f"{result.allocation.amounts[t]:.2f}", result.request.currency,
                                       result.allocation.realized_weights[t]) for t in result.request.tickers]),
             "frontier.csv": _csv(["expected_annual_return", "annual_volatility", "expected_sharpe"],
                                  [(p.metrics.expected_annual_return, p.metrics.annual_volatility, p.metrics.expected_sharpe_ratio)
                                   for p in result.frontier.points])}
    if result.backtest:
        runs = [result.backtest.optimized, result.backtest.equal_weight]
        files["backtest.csv"] = _csv(["date", "strategy", "wealth", "net_return"],
                                    [(day.date().isoformat(), run.name, value, run.net_returns.loc[day])
                                     for run in runs for day, value in run.equity.items()])
    if result.simulation:
        sim = result.simulation
        files["scenario_quantiles.csv"] = _csv(["trading_day", *sim.path_quantiles.columns],
                                               [(int(day), *row) for day, row in sim.path_quantiles.iterrows()])
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return output.getvalue()

import csv
from dataclasses import replace
from decimal import Decimal
from io import BytesIO, StringIO
import json
from zipfile import ZipFile

import pytest

from portfolio_optimizer.errors import ReportError
from portfolio_optimizer.services import report_bundle, report_html, report_json


def test_json_matches_authoritative_result(analysis_result):
    data = json.loads(report_json(analysis_result))
    assert data["report_version"] == 1
    assert data["allocation"]["budget"] == "100000.00"
    assert data["portfolio"]["weights"] == analysis_result.portfolio.weights.to_dict()
    assert data["portfolio"]["metrics"]["expected_annual_return"] == analysis_result.portfolio.metrics.expected_annual_return
    assert data["backtest"]["optimized"]["metrics"]["total_return"] == analysis_result.backtest.optimized.metrics.total_return
    assert data["simulation"]["seed"] == analysis_result.config.random_seed
    assert "wealth_paths" not in data["simulation"]
    assert data["data"]["quality"]["observations"] == 5


def test_html_has_charts_assumptions_and_figures(analysis_result):
    html = report_html(analysis_result).decode()
    assert html.startswith("<!doctype html>")
    assert html.count("<svg") == 3
    assert "RELIANCE.NS" in html and "TCS.NS" in html
    assert "annualized daily arithmetic" in html
    assert "Historical wealth after costs" in html
    assert "Model loss frequency" in html
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html


def test_html_escapes_untrusted_provenance(analysis_result):
    source = replace(analysis_result.prepared.source, provider='<script>alert("x")</script>')
    modified = replace(analysis_result, prepared=replace(analysis_result.prepared, source=source),
                       warnings=("<img src=x onerror=alert(1)>",))
    html = report_html(modified).decode()
    assert "<script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html


def test_bundle_contents_and_budget(analysis_result):
    with ZipFile(BytesIO(report_bundle(analysis_result))) as archive:
        assert set(archive.namelist()) == {"report.html", "analysis.json", "allocations.csv", "frontier.csv", "backtest.csv", "scenario_quantiles.csv"}
        rows = list(csv.DictReader(StringIO(archive.read("allocations.csv").decode())))
        assert sum(Decimal(row["amount"]) for row in rows) == analysis_result.allocation.budget
        assert all(row["currency"] == "INR" for row in rows)
        assert json.loads(archive.read("analysis.json"))["request"]["tickers"] == ["RELIANCE.NS", "TCS.NS"]


def test_unavailable_evaluation_is_visible(analysis_result):
    result = replace(analysis_result, simulation=None, backtest=None)
    assert report_html(result).decode().count("Unavailable; see component issues.") == 2
    with ZipFile(BytesIO(report_bundle(result))) as archive:
        assert "backtest.csv" not in archive.namelist()
        assert "scenario_quantiles.csv" not in archive.namelist()


def test_null_sharpe_serializes_without_nan(analysis_result):
    portfolio = replace(analysis_result.portfolio, metrics=replace(analysis_result.portfolio.metrics, expected_sharpe_ratio=None))
    result = replace(analysis_result, portfolio=portfolio)
    assert json.loads(report_json(result))["portfolio"]["metrics"]["expected_sharpe_ratio"] is None
    assert "Undefined" in report_html(result).decode()


def test_nonfinite_report_data_rejected(analysis_result):
    result = replace(analysis_result, benchmark=replace(analysis_result.benchmark, annual_volatility=float("nan")))
    with pytest.raises(ReportError):
        report_json(result)


def test_report_failure_does_not_discard_analysis(analysis_result):
    from unittest.mock import patch
    with patch("portfolio_optimizer.services.reports.line_chart", side_effect=ValueError("chart failed")):
        with pytest.raises(ReportError):
            report_html(analysis_result)
    assert analysis_result.portfolio.status == "optimal"
    assert json.loads(report_json(analysis_result))["portfolio"]["status"] == "optimal"


def test_disabled_features_are_distinct_from_failures(analysis_result):
    options = replace(analysis_result.options, include_simulation=False, backtest=None)
    result = replace(analysis_result, options=options, simulation=None, backtest=None)
    html = report_html(result).decode()
    assert html.count("Not requested.") == 2
    assert "Unavailable; see component issues." not in html


def test_negative_assessment_in_reports(analysis_result):
    from portfolio_optimizer.evaluation.assessment import assess_baseline
    metrics = replace(analysis_result.portfolio.metrics, expected_annual_return=-.0319)
    result = replace(analysis_result, portfolio=replace(analysis_result.portfolio, metrics=metrics),
                     assessment=assess_baseline(analysis_result.estimates, metrics))
    html = report_html(result).decode()
    assert "no positive historical return estimate" in html
    assert "-3.19%" in html
    assert "Historical mean return (annualized)" in html
    assert json.loads(report_json(result))["assessment"]["status"] == "nonpositive_historical_mean"

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from portfolio_optimizer.errors import ProviderError, ReportError
from portfolio_optimizer.services.analysis import ComponentIssue
from portfolio_optimizer.ui.dashboard import create_service

APP = Path(__file__).resolve().parents[2] / "app.py"
pytestmark = pytest.mark.ui


def dashboard():
    return AppTest.from_file(str(APP), default_timeout=20).run()


def test_initial_screen_does_not_fetch_or_analyze():
    with patch("portfolio_optimizer.ui.dashboard.create_service") as factory:
        app = dashboard()
    assert not app.exception
    assert app.title[0].value == "Intelligent Portfolio Optimizer"
    assert app.radio(key="data_source").value == "Demo (synthetic)"
    factory.assert_not_called()
    assert "analysis_result" not in app.session_state


def test_demo_analysis_renders_results_and_downloads():
    app = dashboard()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["analysis_result"]
    assert result.prepared.source.provider == "synthetic_demo"
    assert result.simulation is not None and result.backtest is not None
    assert len(app.get("plotly_chart")) == 4
    assert len(app.get("download_button")) == 3
    assert app.session_state["downloads"]["zip"].startswith(b"PK")
    assert app.metric[0].value == f"{result.portfolio.metrics.expected_annual_return:.2%}"


def test_rerun_reuses_analysis_and_report_bytes():
    with patch("portfolio_optimizer.ui.dashboard.create_service", wraps=create_service) as factory:
        app = dashboard()
        app.button(key="analyze").click().run()
        generated = app.session_state["analysis_result"].generated_at
        downloads = app.session_state["downloads"].copy()
        app.run()
    assert not app.exception
    assert factory.call_count == 1
    assert app.session_state["analysis_result"].generated_at == generated
    assert app.session_state["downloads"] == downloads


def test_input_change_invalidates_result_and_downloads():
    with patch("portfolio_optimizer.ui.dashboard.create_service", wraps=create_service) as factory:
        app = dashboard()
        app.button(key="analyze").click().run()
        app.number_input(key="investment_amount").set_value(200_000).run()
    assert not app.exception
    assert factory.call_count == 1
    assert "analysis_result" not in app.session_state
    assert "downloads" not in app.session_state
    assert len(app.get("download_button")) == 0
    assert any("Inputs changed" in info.value for info in app.info)


def test_source_change_invalidates_demo_results_without_fetch():
    with patch("portfolio_optimizer.ui.dashboard.create_service", wraps=create_service) as factory:
        app = dashboard()
        app.button(key="analyze").click().run()
        app.radio(key="data_source").set_value("Yahoo Finance").run()
    assert not app.exception
    assert factory.call_count == 1
    assert "analysis_result" not in app.session_state
    assert app.text_input(key="live_tickers").value == "BHARTIARTL.NS, SBIN.NS, BEL.NS"


def test_invalid_amount_shows_validation_error():
    app = dashboard()
    app.number_input(key="investment_amount").set_value(0)
    app.button(key="analyze").click().run()
    assert not app.exception
    assert any("positive and finite" in message.value for message in app.error)
    assert "analysis_result" not in app.session_state


def test_infeasible_cap_shows_error():
    app = dashboard()
    app.slider(key="weight_cap").set_value(10)
    app.button(key="analyze").click().run()
    assert not app.exception
    assert any("cap is too small" in message.value for message in app.error)


def test_provider_failure_does_not_show_stale_results():
    app = dashboard()
    app.button(key="analyze").click().run()
    failed = Mock()
    failed.analyze.side_effect = ProviderError("Market data unavailable")
    with patch("portfolio_optimizer.ui.dashboard.create_service", return_value=failed):
        app.button(key="analyze").click().run()
    assert not app.exception
    assert "analysis_result" not in app.session_state
    assert "downloads" not in app.session_state
    assert any("Market data unavailable" in message.value for message in app.error)


def test_disabled_evaluation_has_clear_empty_states():
    app = dashboard()
    app.checkbox(key="include_simulation").set_value(False)
    app.checkbox(key="include_backtest").set_value(False)
    app.button(key="analyze").click().run()
    assert not app.exception
    assert app.session_state["analysis_result"].simulation is None
    assert len(app.get("plotly_chart")) == 2
    assert any("Scenarios were not requested" in message.value for message in app.info)
    assert any("Backtesting was not requested" in message.value for message in app.info)


def test_optional_failure_visible_without_losing_core_result(analysis_result):
    result = replace(analysis_result, simulation=None, issues=(ComponentIssue("simulation", "incompatible moments"),))
    service = Mock()
    service.analyze.return_value = result
    with patch("portfolio_optimizer.ui.dashboard.create_service", return_value=service):
        app = dashboard()
        app.button(key="analyze").click().run()
    assert not app.exception
    assert any("Simulation unavailable" in message.value for message in app.warning)
    assert "analysis_result" in app.session_state


def test_report_failure_keeps_result_and_retry_does_not_analyze():
    with patch("portfolio_optimizer.ui.dashboard.create_service", wraps=create_service) as factory:
        app = dashboard()
        with patch("portfolio_optimizer.ui.dashboard.report_html", side_effect=ReportError("Export failed")):
            app.button(key="analyze").click().run()
        assert "analysis_result" in app.session_state
        assert "downloads" not in app.session_state
        assert any("Report generation failed" in message.value for message in app.error)
        app.button(key="retry_report").click().run()
    assert not app.exception
    assert factory.call_count == 1
    assert "downloads" in app.session_state


def test_empty_stock_selection_shows_error():
    app = dashboard()
    app.multiselect(key="demo_tickers").set_value([])
    app.button(key="analyze").click().run()
    assert not app.exception
    assert any("Select at least one stock" in message.value for message in app.error)


def test_reversed_dates_show_error():
    app = dashboard()
    from portfolio_optimizer.data.demo import DEMO_END, DEMO_START
    app.date_input(key="demo_start").set_value(DEMO_END)
    app.date_input(key="demo_end").set_value(DEMO_START)
    app.button(key="analyze").click().run()
    assert not app.exception
    assert any("Start date must be before" in message.value for message in app.error)


def test_negative_return_cannot_look_like_positive_recommendation(analysis_result):
    from portfolio_optimizer.evaluation.assessment import assess_baseline
    metrics = replace(analysis_result.portfolio.metrics, expected_annual_return=-.0319)
    portfolio = replace(analysis_result.portfolio, metrics=metrics)
    result = replace(analysis_result, portfolio=portfolio, assessment=assess_baseline(analysis_result.estimates, metrics))
    service = Mock()
    service.analyze.return_value = result
    with patch("portfolio_optimizer.ui.dashboard.create_service", return_value=service):
        app = dashboard()
        app.button(key="analyze").click().run()
    assert not app.exception
    assert app.metric[0].label == "Historical mean return (annualized)"
    assert app.metric[0].value == "-3.19%"
    assert any("no positive historical return estimate" in message.value for message in app.warning)
    assert any("Future-return forecasting is not available" in message.value for message in app.caption)


def test_existing_old_default_migrates():
    app = dashboard()
    app.session_state["live_tickers"] = "RELIANCE.NS, TCS.NS, HDFCBANK.NS"
    app.radio(key="data_source").set_value("Yahoo Finance").run()
    assert not app.exception
    assert app.text_input(key="live_tickers").value == "BHARTIARTL.NS, SBIN.NS, BEL.NS"


def test_custom_stock_selection_is_preserved():
    app = dashboard()
    app.session_state["live_tickers"] = "ICICIBANK.NS, LT.NS"
    app.radio(key="data_source").set_value("Yahoo Finance").run()
    assert not app.exception
    assert app.text_input(key="live_tickers").value == "ICICIBANK.NS, LT.NS"
    app.radio(key="data_source").set_value("Demo (synthetic)").run()
    app.radio(key="data_source").set_value("Yahoo Finance").run()
    assert app.text_input(key="live_tickers").value == "ICICIBANK.NS, LT.NS"

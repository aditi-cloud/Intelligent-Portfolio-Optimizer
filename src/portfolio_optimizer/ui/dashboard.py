from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from ..data.cache import CachedPriceProvider
from ..data.demo import demo_dataset
from ..data.memory import InMemoryPriceProvider
from ..data.yahoo import YahooPriceProvider
from ..errors import PortfolioError, ReportError
from ..services import PortfolioAnalysisService, report_bundle, report_html, report_json
from .inputs import DashboardInputs, read_inputs
from .results import render_results


def create_service(inputs: DashboardInputs) -> PortfolioAnalysisService:
    provider = (InMemoryPriceProvider(demo_dataset()) if inputs.mode == "Demo (synthetic)"
                else CachedPriceProvider(YahooPriceProvider(), Path(".cache/prices")))
    return PortfolioAnalysisService(provider, inputs.config)


def _prepare_downloads(result):
    try:
        st.session_state["downloads"] = {
            "html": report_html(result), "json": report_json(result), "zip": report_bundle(result),
        }
        st.session_state.pop("report_error", None)
    except ReportError as exc:
        st.session_state.pop("downloads", None)
        st.session_state["report_error"] = str(exc)


def main():
    st.set_page_config(page_title="Intelligent Portfolio Optimizer", page_icon="📈", layout="wide")
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    inputs, submitted = read_inputs(today)
    # Invalidate results/reports produced by earlier analysis contracts.
    identity = "historical-baseline-v2:" + inputs.identity()
    old_identity = st.session_state.get("analysis_identity")
    changed = old_identity is not None and old_identity != identity
    if changed:
        for key in ("analysis_result", "analysis_identity", "downloads", "report_error"):
            st.session_state.pop(key, None)
    st.caption("PORTFOLIO LAB")
    st.title("Intelligent Portfolio Optimizer")
    st.write("Explore allocations, understand risk, and compare strategies before committing capital.")
    st.caption("Current model: historical baseline. ML return and volatility forecasting are still to be implemented.")
    if inputs.mode == "Demo (synthetic)":
        st.info("Demo mode uses synthetic prices. Every chart and report is an illustration, with no live market calls.")
    else:
        st.caption("Yahoo Finance mode uses historical adjusted prices for stocks in your selected currency.")
    if changed and not submitted:
        st.info("Inputs changed. Select Analyze portfolio to calculate new results.")
    if submitted:
        for key in ("analysis_result", "analysis_identity", "downloads", "report_error"):
            st.session_state.pop(key, None)
        try:
            with st.spinner("Analyzing prices, allocations and risk…"):
                result = create_service(inputs).analyze(inputs.request, inputs.options, today=today)
            st.session_state["analysis_result"] = result
            st.session_state["analysis_identity"] = identity
            _prepare_downloads(result)
        except (PortfolioError, ValueError) as exc:
            st.error(str(exc))
    result = st.session_state.get("analysis_result")
    if result is None:
        st.subheader("Start with a portfolio")
        st.write("Choose your stocks and budget in the sidebar, set your risk appetite, then select Analyze portfolio.")
        columns = st.columns(3)
        for column, title, text in zip(columns,
                                      ["Allocate your budget", "Explore risk & return", "Test the strategy"],
                                      ["See target weights and currency amounts for every stock.",
                                       "Compare your portfolio with equal weighting and explore possible outcomes.",
                                       "Review a historical replay with transaction costs and download the analysis."]):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                st.write(text)
    else:
        render_results(result)
        st.subheader("Download your analysis")
        if "report_error" in st.session_state:
            st.error(f"Report generation failed: {st.session_state['report_error']}")
            if st.button("Retry report generation", key="retry_report"):
                _prepare_downloads(result)
        downloads = st.session_state.get("downloads")
        if downloads:
            columns = st.columns(3)
            for column, kind, label, extension, mime in zip(columns,
                ["html", "json", "zip"], ["Printable report", "Analysis JSON", "Complete report bundle"],
                ["html", "json", "zip"], ["text/html", "application/json", "application/zip"]):
                column.download_button(label, downloads[kind], file_name=f"portfolio-analysis.{extension}",
                                       mime=mime, on_click="ignore", width="stretch", key=f"download_{kind}")
    st.divider()
    st.caption("Educational decision support. Expected metrics, historical replay and model scenarios do not guarantee future outcomes.")

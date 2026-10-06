"""Generate reviewable report artifacts through the application service.

Run: .venv/bin/python examples/service_report.py
"""

import argparse
from pathlib import Path

from baseline import synthetic_inputs
from portfolio_optimizer.data.memory import InMemoryPriceProvider
from portfolio_optimizer.evaluation.backtest import BacktestConfig
from portfolio_optimizer.services import AnalysisOptions, PortfolioAnalysisService, report_bundle, report_html, report_json


def main():
    parser = argparse.ArgumentParser(description="Export a report from synthetic prices; no network calls")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo-report"))
    args = parser.parse_args()
    config, request, dataset = synthetic_inputs()
    service = PortfolioAnalysisService(InMemoryPriceProvider(dataset), config)
    result = service.analyze(request, AnalysisOptions(frontier_points=12, simulation_scenarios=1_000,
                                                     backtest=BacktestConfig(60, 21, 10)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {"report.html": report_html(result), "analysis.json": report_json(result), "report.zip": report_bundle(result)}
    for name, content in files.items():
        path = args.output_dir / name
        path.write_bytes(content)
        print(f"Created {path} ({len(content):,} bytes)")
    for issue in result.issues:
        print(f"Unavailable {issue.component}: {issue.message}")


if __name__ == "__main__":
    main()

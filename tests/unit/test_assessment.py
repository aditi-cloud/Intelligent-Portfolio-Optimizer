from dataclasses import replace

from portfolio_optimizer.evaluation.assessment import assess_baseline


def test_negative_stock_basket_is_explicit(analysis_result):
    estimates = replace(analysis_result.estimates, expected_returns=analysis_result.estimates.expected_returns * 0 - .03)
    metrics = replace(analysis_result.portfolio.metrics, expected_annual_return=-.0319)
    assessment = assess_baseline(estimates, metrics)
    assert assessment.status == "nonpositive_historical_mean"
    assert "Every selected stock" in assessment.explanation
    assert "requires 100% investment" in assessment.explanation


def test_positive_mean_below_reference_flagged(analysis_result):
    metrics = replace(analysis_result.portfolio.metrics, expected_annual_return=.02, annual_risk_free_rate=.04)
    assessment = assess_baseline(analysis_result.estimates, metrics)
    assert assessment.status == "below_reference_rate"
    assert "2.00%" in assessment.explanation
    assert "4.00%" in assessment.explanation


def test_positive_mean_is_still_baseline(analysis_result):
    assessment = assess_baseline(analysis_result.estimates, analysis_result.portfolio.metrics)
    assert assessment.status == "historical_baseline"
    assert "no ML forecast" in assessment.explanation

"""financial_analyzer.py의 비율 추출/성장률 계산 로직을 검증한다."""
from src.analytics.financial_analyzer import (
    compute_revenue_growth,
    margins,
    profitability,
    summarize,
    valuation_multiples,
)


def test_margins_extracts_available_fields():
    info = {"grossMargins": 0.74, "operatingMargins": 0.66, "profitMargins": 0.63}
    assert margins(info) == {"gross_margin": 0.74, "operating_margin": 0.66, "net_margin": 0.63}


def test_margins_returns_none_for_missing_fields():
    assert margins({})["gross_margin"] is None


def test_valuation_multiples_falls_back_to_forward_pe_when_trailing_missing():
    info = {"forwardPE": 16.1, "pegRatio": 0.56, "enterpriseToEbitda": 30.0, "debtToEquity": 6.5}
    assert valuation_multiples(info)["per"] == 16.1


def test_valuation_multiples_prefers_trailing_pe_when_available():
    info = {"trailingPE": 31.2, "forwardPE": 16.1}
    assert valuation_multiples(info)["per"] == 31.2


def test_profitability_extracts_available_fields():
    info = {"returnOnEquity": 0.19, "returnOnAssets": 0.10, "revenueGrowth": 0.69, "freeCashflow": 1000}
    result = profitability(info)
    assert result["roe"] == 0.19
    assert result["fcf"] == 1000


def test_summarize_combines_all_three_groups():
    info = {"grossMargins": 0.5, "trailingPE": 10.0, "returnOnEquity": 0.2}
    result = summarize(info)
    assert result["gross_margin"] == 0.5
    assert result["per"] == 10.0
    assert result["roe"] == 0.2


def test_compute_revenue_growth_matches_expected_formula():
    assert compute_revenue_growth(current_revenue=124, prior_revenue=100) == 0.24


def test_compute_revenue_growth_returns_none_when_no_prior_data():
    assert compute_revenue_growth(100, 0) is None
    assert compute_revenue_growth(100, None) is None

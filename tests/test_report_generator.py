"""report_generator.py의 컨텍스트 조립/포맷팅/HTML 렌더링을 검증한다.

실제 PDF가 제대로 렌더링되는지(레이아웃, 폰트 등)는 NVDA 실 데이터로 생성해 수동으로 확인했다
(README 참고). 여기서는 WeasyPrint 호출 없이 build_context/render_html까지만 검증한다.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from src.report.report_generator import (
    _investment_opinion,
    build_context,
    generate_business_analysis,
    generate_company_overview,
    generate_pdf,
    render_html,
)


def _minimal_thesis():
    return {
        "investment_thesis": ["강력한 매출 성장세를 보이고 있다."],
        "bear_case": ["밸류에이션 부담이 존재한다."],
        "catalysts": ["신제품 출시"],
        "risks": ["경쟁 심화"],
        "target_price": 300.0,
        "target_price_rationale": "Comps 중앙값 기준",
    }


def test_build_context_formats_money_pct_and_multiple():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={"roe": 0.5, "per": 30.123, "fcf": 1000.0},
        dcf_result={"implied_share_price": 100.0, "enterprise_value": 5000.0},
        comps_result={"implied_price_from_ev_ebitda": 400.0, "implied_price_from_pe": 350.0},
        football_field_table=None, thesis=_minimal_thesis(),
    )
    assert ctx["current_price"] == "$200.00"
    assert ctx["target_price"] == "$300.00"
    assert ctx["financial_ratios"]["ROE"] == "50.0%"
    assert ctx["financial_ratios"]["PER"] == "30.1x"
    assert ctx["financial_ratios"]["잉여현금흐름(FCF)"] == "$1,000.00"


def test_build_context_formats_debt_to_equity_as_prescaled_percent_not_multiple():
    # yfinance의 debtToEquity는 이미 퍼센트로 스케일된 값이다(예: 15.174 = 15.174%, D/E 비율 0.15).
    # "15.2x"(배수)로 잘못 포맷하면 실제보다 100배 과장된 부채비율로 보이게 된다.
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={"debt_to_equity": 15.174}, dcf_result={}, comps_result={},
        football_field_table=None, thesis=_minimal_thesis(),
    )
    assert ctx["financial_ratios"]["부채비율"] == "15.2%"


def test_build_context_labels_financial_ratio_keys_in_korean():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={"gross_margin": 0.6, "unknown_ratio": 1.0}, dcf_result={}, comps_result={},
        football_field_table=None, thesis=_minimal_thesis(),
    )
    assert ctx["financial_ratios"]["매출총이익률"] == "60.0%"
    assert "gross_margin" not in ctx["financial_ratios"]
    assert ctx["financial_ratios"]["unknown_ratio"] == 1.0  # 매핑에 없는 키는 원래 키를 그대로 라벨로 사용


def test_build_context_computes_upside_pct():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=100.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),  # target_price=300.0
    )
    assert ctx["upside_pct"] == "200.0%"


def test_build_context_executive_summary_includes_first_thesis_bullet():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),
    )
    assert "강력한 매출 성장세를 보이고 있다." in ctx["executive_summary"]
    assert "NVIDIA" in ctx["executive_summary"]


def test_build_context_defaults_optional_sections_to_na():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),
    )
    assert ctx["company_overview"] == "N/A"
    assert ctx["business_analysis"] == "N/A"


def test_build_context_converts_football_field_dataframe_to_records():
    ff = pd.DataFrame([{"method": "DCF", "low": 10, "high": 20, "midpoint": 15}])
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=ff,
        thesis=_minimal_thesis(),
    )
    assert ctx["football_field"] == [{"method": "DCF", "low": 10, "high": 20, "midpoint": 15}]


def test_generate_company_overview_returns_none_when_summary_missing():
    assert generate_company_overview({}) is None


def test_generate_company_overview_summarizes_via_llm():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="  한글 요약문.  ")

    result = generate_company_overview({"longBusinessSummary": "English summary text."}, client=client)

    assert result == "한글 요약문."
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "English summary text." in prompt


def test_generate_business_analysis_includes_sector_and_ratios_in_prompt():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="  분석문.  ")

    result = generate_business_analysis(
        {"sector": "Technology", "industry": "Semiconductors", "fullTimeEmployees": 76907},
        {"roe": 0.4}, client=client,
    )

    assert result == "분석문."
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "Technology" in prompt and "Semiconductors" in prompt and "0.4" in prompt


def test_generate_business_analysis_includes_peer_multiples_when_comps_result_given():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="분석문")

    generate_business_analysis(
        {"sector": "Technology"}, {}, comps_result={"median_ev_ebitda": 15.4, "median_pe": 73.5}, client=client,
    )

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "15.4" in prompt and "73.5" in prompt


def test_investment_opinion_is_buy_when_upside_at_or_above_10_percent():
    assert _investment_opinion(0.10) == "매수"
    assert _investment_opinion(0.25) == "매수"


def test_investment_opinion_is_sell_when_downside_at_or_beyond_10_percent():
    assert _investment_opinion(-0.10) == "매도"
    assert _investment_opinion(-0.30) == "매도"


def test_investment_opinion_is_neutral_between_thresholds():
    assert _investment_opinion(0.0) == "중립"
    assert _investment_opinion(0.05) == "중립"
    assert _investment_opinion(-0.05) == "중립"


def test_investment_opinion_is_na_when_upside_unavailable():
    assert _investment_opinion(None) == "N/A"


def test_build_context_includes_investment_opinion():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=100.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),  # target_price=300.0 -> upside 200%
    )
    assert ctx["investment_opinion"] == "매수"


def test_render_html_shows_fiscal_quarter_next_to_earnings_call_summary():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),
        earnings_call_summaries={"guidance": {"summary": "가이던스 상향", "citations": []}},
        fiscal_quarter="Q2 2026",
    )
    html = render_html(ctx)
    assert "실적발표 컨퍼런스콜 요약 (Q2 2026)" in html


def test_render_html_omits_fiscal_quarter_parens_when_not_given():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),
        earnings_call_summaries={"guidance": {"summary": "가이던스 상향", "citations": []}},
    )
    html = render_html(ctx)
    assert "실적발표 컨퍼런스콜 요약</h3>" in html


def test_render_html_produces_expected_sections():
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={"roe": 0.5}, dcf_result={"implied_share_price": 100.0},
        comps_result={}, football_field_table=None, thesis=_minimal_thesis(),
        news_summary={"tone_counts": {"Positive": 5}, "high_impact_negative_titles": ["headline"]},
    )
    html = render_html(ctx)
    assert "NVIDIA" in html
    assert "Investment Thesis" in html
    assert "강력한 매출 성장세를 보이고 있다." in html
    assert "Positive: 5건" in html
    assert "{'Positive'" not in html  # dict repr가 그대로 새어나오면 안 된다


def test_generate_pdf_calls_weasyprint_with_rendered_html(tmp_path):
    ctx = build_context(
        company_name="NVIDIA", ticker="NVDA", currency="USD", current_price=200.0,
        financial_ratios={}, dcf_result={}, comps_result={}, football_field_table=None,
        thesis=_minimal_thesis(),
    )
    output_path = str(tmp_path / "report.pdf")
    mock_html_cls = MagicMock()

    with patch.dict("sys.modules", {"weasyprint": MagicMock(HTML=mock_html_cls)}):
        result_path = generate_pdf(ctx, output_path=output_path)

    assert result_path == output_path
    mock_html_cls.assert_called_once()
    mock_html_cls.return_value.write_pdf.assert_called_once_with(output_path)

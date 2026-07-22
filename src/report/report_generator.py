"""⑩ PDF Report — 지금까지 만든 모든 컴포넌트(④~⑨)의 결과를 Jinja2 템플릿에 채워 WeasyPrint로
PDF를 만든다. 섹션 구성: Executive Summary, Company Overview, Business Analysis, Financial Analysis,
Valuation, Investment Thesis, Bear Case, Catalyst, Risk, Appendix.

Executive Summary는 별도 LLM 호출 없이 이미 생성된 thesis_agent 결과(investment_thesis 첫 문장 +
target_price)로 조합한다 — 비용 추가가 없다. Company Overview/Business Analysis는 ①~⑨ 어디서도
만들지 않는 콘텐츠라 build_context()에 문자열을 그대로 전달하는 선택적 파라미터로 남겨두고, 안 주면
"N/A"로 표시한다 — 실제 텍스트는 generate_company_overview()/generate_business_analysis()로 만들어서
호출 측(노트북/에이전트)에서 넘겨준다(사업보고서/10-K 본문을 파싱하는 별도 컴포넌트는 아직 없어서,
yfinance info에 이미 있는 데이터만 근거로 삼는다 — 그 이상은 추측하지 않는다).
"""
import json
import os
from datetime import date

from jinja2 import Environment, FileSystemLoader

from src.analytics._common import get_llm_client
from src.config import OUTPUT_DIR, PROJECT_ROOT

_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")

_CURRENCY_SYMBOLS = {"USD": "$", "KRW": "₩"}

_RATIO_FORMATTERS = {
    "gross_margin": "pct", "operating_margin": "pct", "net_margin": "pct",
    "roe": "pct", "roa": "pct", "revenue_growth": "pct",
    "per": "multiple", "peg": "multiple", "ev_to_ebitda": "multiple",
    "debt_to_equity": "pct_prescaled",  # yfinance가 이미 퍼센트로 준다(예: 15.174 = 15.174%) — financial_analyzer.valuation_multiples() 참고
    "fcf": "money",
}

_RATIO_LABELS = {
    "gross_margin": "매출총이익률", "operating_margin": "영업이익률", "net_margin": "순이익률",
    "per": "PER", "peg": "PEG", "ev_to_ebitda": "EV/EBITDA", "debt_to_equity": "부채비율",
    "roe": "ROE", "roa": "ROA", "revenue_growth": "매출성장률", "fcf": "잉여현금흐름(FCF)",
}


def _fmt_money(value, currency="USD"):
    if value is None:
        return "N/A"
    symbol = _CURRENCY_SYMBOLS.get(currency, "")
    return f"{symbol}{value:,.2f}"


def _fmt_pct(value):
    return f"{value:.1%}" if value is not None else "N/A"


def _fmt_multiple(value):
    return f"{value:.1f}x" if value is not None else "N/A"


def _fmt_pct_prescaled(value):
    """이미 퍼센트 단위로 스케일된 값(예: 15.174 = 15.174%)을 그대로 표시한다 — _fmt_pct처럼 100을
    다시 곱하면 안 된다."""
    return f"{value:.1f}%" if value is not None else "N/A"


_BUY_THRESHOLD = 0.10
_SELL_THRESHOLD = -0.10


def _investment_opinion(upside_pct, buy_threshold=_BUY_THRESHOLD, sell_threshold=_SELL_THRESHOLD):
    """목표주가-현재주가 괴리율(upside_pct)로 투자의견을 결정론적으로 산출한다 — LLM이 별도로 지어내게
    하지 않고, 이미 계산된 숫자를 임계값으로 나눠 매수/매도/중립을 정한다."""
    if upside_pct is None:
        return "N/A"
    if upside_pct >= buy_threshold:
        return "매수"
    if upside_pct <= sell_threshold:
        return "매도"
    return "중립"


def generate_company_overview(info, model="gemini-flash-latest", client=None):
    """yfinance의 longBusinessSummary(영문 원문)를 한글로 요약한다. 원문에 없는 내용은 추가하지
    않는다 — 새로운 사실을 만들어내는 게 아니라 이미 있는 내용을 한글로 옮기는 것뿐이다. 원문이
    없으면(일부 종목은 필드 자체가 비어 있음) None을 반환한다."""
    summary = info.get("longBusinessSummary")
    if not summary:
        return None

    client = client or get_llm_client()
    prompt = (
        "다음은 yfinance에서 가져온 회사 사업개요(영문)다. 이 내용만 근거로 3~4문장의 한글로 자연스럽게 "
        "요약하라. 원문에 없는 내용을 추가하거나 추측하지 마라.\n\n" + summary
    )
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


def generate_business_analysis(info, financial_ratios, comps_result=None, model="gemini-flash-latest", client=None):
    """섹터/산업/직원 수 같은 구조화 데이터와 재무비율(+피어 비교 배수, 있으면)을 근거로 사업
    분석 텍스트를 생성한다. 데이터에 없는 내용(예: 구체적 사업부문별 매출 비중)은 추측하지 않는다."""
    facts = {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "full_time_employees": info.get("fullTimeEmployees"),
        "financial_ratios": financial_ratios,
    }
    if comps_result:
        facts["peer_median_multiples"] = {
            "median_ev_ebitda": comps_result.get("median_ev_ebitda"),
            "median_pe": comps_result.get("median_pe"),
        }

    client = client or get_llm_client()
    prompt = (
        "다음은 한 회사의 사업 개요·재무비율·(있다면) 피어 그룹 대비 밸류에이션 배수 데이터다. "
        "이 데이터만 근거로 이 회사의 사업 특징과 재무적 포지셔닝을 2~3문단의 한글로 분석하라. "
        "데이터에 없는 내용(사업부문별 매출 비중, 구체적 제품 라인업 등)은 추측하거나 지어내지 마라.\n\n"
        + json.dumps(facts, ensure_ascii=False, indent=2, default=str)
    )
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


def _format_financial_ratios(ratios, currency):
    """리포트에 보여줄 형태로 값을 포맷하고, 항목명도 영문 snake_case 키 대신 한글 라벨로 바꾼다
    (_RATIO_LABELS 참고) — 매핑에 없는 키는 원래 키를 그대로 라벨로 쓴다."""
    formatted = {}
    for key, value in ratios.items():
        kind = _RATIO_FORMATTERS.get(key)
        if kind == "pct":
            formatted_value = _fmt_pct(value)
        elif kind == "pct_prescaled":
            formatted_value = _fmt_pct_prescaled(value)
        elif kind == "multiple":
            formatted_value = _fmt_multiple(value)
        elif kind == "money":
            formatted_value = _fmt_money(value, currency)
        else:
            formatted_value = value if value is not None else "N/A"
        label = _RATIO_LABELS.get(key, key)
        formatted[label] = formatted_value
    return formatted


def build_context(
    company_name, ticker, currency,
    current_price, financial_ratios, dcf_result, comps_result, football_field_table,
    thesis, news_summary=None, earnings_call_summaries=None,
    company_overview=None, business_analysis=None, fiscal_quarter=None,
):
    """각 컴포넌트의 결과물을 report_template.html이 바로 쓸 수 있는 형태로 조합한다.

    financial_ratios: financial_analyzer.summarize()의 결과
    dcf_result: dcf.run_dcf()의 결과, comps_result: comps.comps_valuation()의 결과
    football_field_table: football_field.build_football_field()의 결과(DataFrame)
    thesis: thesis_agent.generate_thesis()의 결과
    news_summary: news_analyzer.summarize_classification()의 결과
    earnings_call_summaries: {"guidance": {...}, "risk": {...}} 형태 — earnings_call_analyzer.summarize_topic() 결과들
    fiscal_quarter: 실적발표 대본의 분기(예: "Q2 2026", transcript_df["fiscal_quarter"].iloc[0]) — Appendix에
        어느 분기 실적발표를 요약한 건지 표시하기 위함
    """
    target_price = thesis.get("target_price")
    upside_pct = (target_price / current_price - 1) if target_price and current_price else None

    thesis_bullets = thesis.get("investment_thesis") or []
    executive_summary = (
        f"{company_name}({ticker}) 현재가는 {_fmt_money(current_price, currency)}, 목표주가는 "
        f"{_fmt_money(target_price, currency)}({_fmt_pct(upside_pct)} "
        f"{'상승' if (upside_pct or 0) >= 0 else '하락'} 여력)로 산정됨. "
        + (thesis_bullets[0] if thesis_bullets else "")
    )

    return {
        "company_name": company_name,
        "ticker": ticker,
        "report_date": date.today().strftime("%Y-%m-%d"),
        "executive_summary": executive_summary,
        "company_overview": company_overview or "N/A",
        "business_analysis": business_analysis or "N/A",
        "current_price": _fmt_money(current_price, currency),
        "target_price": _fmt_money(target_price, currency),
        "upside_pct": _fmt_pct(upside_pct),
        "investment_opinion": _investment_opinion(upside_pct),
        "financial_ratios": _format_financial_ratios(financial_ratios or {}, currency),
        "dcf_implied_price": _fmt_money((dcf_result or {}).get("implied_share_price"), currency),
        "dcf_enterprise_value": _fmt_money((dcf_result or {}).get("enterprise_value"), currency),
        "comps_price_ev_ebitda": _fmt_money((comps_result or {}).get("implied_price_from_ev_ebitda"), currency),
        "comps_price_pe": _fmt_money((comps_result or {}).get("implied_price_from_pe"), currency),
        "football_field": football_field_table.to_dict("records") if football_field_table is not None else [],
        "investment_thesis": thesis_bullets,
        "bear_case": thesis.get("bear_case", []),
        "catalysts": thesis.get("catalysts", []),
        "risks": thesis.get("risks", []),
        "target_price_rationale": thesis.get("target_price_rationale"),
        "news_summary": news_summary or {},
        "earnings_call_summaries": earnings_call_summaries or {},
        "fiscal_quarter": fiscal_quarter,
    }


def render_html(context):
    """report_template.html에 context 데이터를 채워 HTML 문자열을 반환한다."""
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))
    template = env.get_template("report_template.html")
    return template.render(**context)


def generate_pdf(context, output_path=None):
    """context로 HTML을 렌더링하고 WeasyPrint로 PDF 파일을 생성한다. 저장된 경로를 반환한다."""
    from weasyprint import HTML  # 선택적 의존성 — PDF 생성 시에만 필요

    html_content = render_html(context)

    if output_path is None:
        ticker_slug = context.get("ticker", "report").replace(".", "_")
        output_path = os.path.join(OUTPUT_DIR, "reports", f"{ticker_slug}_{date.today():%Y%m%d}.pdf")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    HTML(string=html_content, base_url=_TEMPLATE_DIR).write_pdf(output_path)
    return output_path

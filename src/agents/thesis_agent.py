"""⑧ Investment Thesis Generator — 지금까지 만든 컴포넌트(Valuation Engine, Financial Analyzer,
Earnings Call Analyzer, News Analyzer)의 결과를 근거로 Investment Thesis/Bear Case/Catalyst/Risk/
Target Price를 LLM으로 생성한다.

Target Price는 LLM이 임의로 지어내면 안 되므로, context["valuation"]에 이미 계산된 DCF/Comps/52주
범위 숫자를 넘기고 "이 범위 안에서, 근거를 들어" 산출하도록 프롬프트에서 강제한다 — 그 외 필드가 없어도
동작은 하지만, valuation 근거가 부실하면 target_price 신뢰도도 그만큼 낮아진다.
"""
import json

from src.analytics._common import get_llm_client

_DCF_WEIGHT = 0.6
_COMPS_WEIGHT = 0.4

_SCHEMA_INSTRUCTIONS = """다음 JSON 스키마로만 답하라 (다른 텍스트 없이 JSON만):
{
  "investment_thesis": ["근거1", "근거2", "근거3"],
  "bear_case": ["반박근거1", "반박근거2", "반박근거3"],
  "catalysts": ["향후 촉매1", "향후 촉매2"],
  "risks": ["리스크1", "리스크2"],
  "target_price": 숫자,
  "target_price_rationale": "target_price를 어떻게 산출했는지 한두 문장"
}
investment_thesis/bear_case는 정확히 3개씩, catalysts/risks는 근거가 뒷받침되는 만큼만 적어라.
target_price는 반드시 context의 valuation 섹션에 나온 범위/숫자에 근거해야 한다 — 거기 없는 숫자를
임의로 지어내지 마라. valuation.weighted_anchor가 있으면 이건 DCF/Comps 범위의 중점을 가중평균한
기준점이다(DCF 60%, Comps 40%) — target_price는 이 기준점을 기본값으로 삼고, 뉴스/실적발표 등
정성적 근거로 조정할 명확한 이유가 있을 때만 벗어나되, 그 경우 target_price_rationale에 왜
기준점에서 벗어났는지 반드시 설명하라."""


def _weighted_target_price(valuation, dcf_weight=_DCF_WEIGHT, comps_weight=_COMPS_WEIGHT):
    """DCF/Comps 범위의 중점을 가중평균해 목표주가의 기준점(anchor)을 계산한다.

    이전엔 LLM이 넓은 범위 안에서 근거 없이 극단값을 고르는 식이라 실행마다 결과가 크게 튀었다
    (같은 종목인데 한 번은 DCF 하단, 다음번은 52주 상단을 고르는 등). 실제 리서치 리포트처럼
    DCF/Comps에 명시적 가중치를 줘서 결정론적 기준점을 먼저 계산해두면, LLM은 그 기준점에서
    정성적 근거로 조정하는 역할만 하게 되어 훨씬 일관성 있다. 52주 범위는 밸류에이션 모델이 아니라
    시장 참고용 밴드라 가중치 계산에는 포함하지 않는다.
    """
    dcf_range = valuation.get("dcf_range")
    comps_range = valuation.get("comps_range")
    dcf_mid = sum(dcf_range) / 2 if dcf_range else None
    comps_mid = sum(comps_range) / 2 if comps_range else None

    if dcf_mid is None:
        return comps_mid
    if comps_mid is None:
        return dcf_mid
    return dcf_weight * dcf_mid + comps_weight * comps_mid


def _format_context(context):
    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


def generate_thesis(company_name, ticker, context, model="gemini-flash-latest", client=None):
    """context: 조립된 리서치 근거 dict. 예:

    {
        "valuation": {"dcf_range": [15, 61], "comps_range": [559, 763], "52w_range": [164, 237],
                      "current_price": 207.29},
        "financials": {"revenue_growth": 0.85, "operating_margin": 0.66, "roe": 1.14},
        "earnings_call": {"guidance_summary": "...", "risk_summary": "..."},
        "news": {"tone_counts": {...}, "high_impact_negative_titles": [...]},
    }
    비어 있는 섹션이 있어도 동작하지만, 근거가 적을수록 결과 신뢰도도 낮아진다.
    """
    context = dict(context)
    valuation = dict(context.get("valuation") or {})
    weighted_anchor = _weighted_target_price(valuation)
    if weighted_anchor is not None:
        valuation["weighted_anchor"] = weighted_anchor
    context["valuation"] = valuation

    prompt = (
        f"'{company_name}'({ticker})에 대한 아래 리서치 근거를 바탕으로 투자 리포트용 Investment "
        f"Thesis를 작성하라. 근거에 없는 내용은 추측하지 마라.\n\n"
        f"=== 리서치 근거 ===\n{_format_context(context)}\n\n{_SCHEMA_INSTRUCTIONS}"
    )

    client = client or get_llm_client()
    response = client.models.generate_content(
        model=model, contents=prompt, config={"response_mime_type": "application/json"},
    )
    return json.loads(response.text)

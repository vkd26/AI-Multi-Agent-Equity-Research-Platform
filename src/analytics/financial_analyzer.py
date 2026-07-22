"""⑥ Financial Analyzer — Revenue Growth, Margin, ROE, PER, PEG, EV/EBITDA, Debt/Equity, FCF를 계산한다.

yfinance 기업정보(market.py.get_company_info)가 이미 이 비율들을 한미 공통으로 정규화해서 제공하므로
이를 그대로 쓴다. ROIC은 시도해봤으나 뺐다 — yfinance의 operatingMargins 필드가 TTM이 아니라 최신
분기 단독치라(totalRevenue/profitMargins 등 다른 필드는 TTM인 것과 달리) 시점이 안 맞는 값을 곱하게
되어 부풀려진 값이 나왔고(TSM 실 데이터로 검증: 계산값 70.3% vs 실제 재무제표 기준 54.2%), 이 오차를
바로잡으려면 분기별 재무제표를 별도로 받아와야 해서 지표 하나의 정밀도 대비 배보다 배꼽이 커진다고
판단했다.
"""


def margins(info):
    """yfinance company info에서 마진 지표를 뽑는다. 필드가 없으면 None으로 남긴다."""
    return {
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
    }


def valuation_multiples(info):
    """debt_to_equity는 yfinance가 이미 퍼센트로 스케일해 반환한다(예: 15.174는 "15.174%",
    D/E 비율로는 0.15) — 다른 pct 필드(0~1 소수)와 스케일이 다르므로 그대로 배수(x)나 소수 퍼센트로
    포맷하면 안 된다. report_generator.py에서 별도 포맷터로 처리한다."""
    return {
        "per": info.get("trailingPE") or info.get("forwardPE"),
        "peg": info.get("pegRatio"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "debt_to_equity": info.get("debtToEquity"),
    }


def profitability(info):
    return {
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "revenue_growth": info.get("revenueGrowth"),
        "fcf": info.get("freeCashflow"),
    }


def summarize(info):
    """yfinance company info 하나로 뽑을 수 있는 표준 지표를 한 번에 모은다."""
    result = {}
    result.update(margins(info))
    result.update(valuation_multiples(info))
    result.update(profitability(info))
    return result


def compute_revenue_growth(current_revenue, prior_revenue):
    """지정한 두 기간 사이의 매출 성장률을 계산한다 — yfinance의 canned revenueGrowth와 달리 원하는
    두 시점(예: DART 분기 공시의 당기/전년동기)을 직접 비교할 수 있다."""
    if not prior_revenue:
        return None
    return current_revenue / prior_revenue - 1

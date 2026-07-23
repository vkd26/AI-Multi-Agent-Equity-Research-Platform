"""⑦ Valuation Engine — DCF. WACC 계산 -> FCF 투영 -> 터미널가치 -> 기업/자기자본가치 -> 내재주가.

거시 가정(무위험금리, 에퀴티리스크프리미엄, 세율 등)은 시장·시점마다 달라지는 값이라 함수 인자로
받는다 — 하드코딩하지 않고 호출 측(노트북/에이전트)에서 최신값을 넣도록 한다. get_risk_free_rate()가
그 값을 실제로 가져오는 헬퍼다.
"""

_KR_RISK_FREE_RATE_FALLBACK = 0.0439  # 한국 국고채 10년물, 2026-07-22 확인(4년 만에 최고치) — 폴백 사유는 get_risk_free_rate() 참고

# Damodaran(NYU Stern) implied ERP 데이터 기준, 2026-07 확인. 국가마다 다르게 잡아야 한다 — 전에는
# 한미 동일하게 5.5%를 썼는데 출처 없는 대략치였다. 미국은 forward-looking implied ERP, 한국은
# "미국 mature-market ERP + 한국 country risk premium(무디스 Aa2 등급, +0.64%p)"이 Damodaran이 직접
# 제공하는 한국 총 ERP 수치다(https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html).
_EQUITY_RISK_PREMIUM = {"US": 0.0445, "KR": 0.0487}


def get_equity_risk_premium(market):
    """에퀴티 리스크 프리미엄을 소수로 가져온다. Damodaran 데이터는 API가 아니라 주기적으로 갱신되는
    발행 자료라 실시간 조회는 못 하고, 확인 시점 값을 그대로 쓴다 — 주기적으로 재확인 권장."""
    return _EQUITY_RISK_PREMIUM.get(market, _EQUITY_RISK_PREMIUM["US"])


def get_risk_free_rate(market):
    """무위험금리(10년물 국채 수익률)를 소수(예: 0.046 = 4.6%)로 가져온다.

    미국은 yfinance의 ^TNX(CBOE 10년물 국채 수익률 지수) 티커로 실시간 조회한다 — Close 값 자체가
    이미 퍼센트 단위(예: 4.642는 4.642%)라 100으로 나눠서 소수로 바꾼다.

    한국은 yfinance에 국채수익률 티커가 없어서(KR10Y=RR, ^KTB10Y 등 여러 후보를 직접 시도해봤지만
    전부 404/데이터 없음) 대신 KRX Open API(국채전문유통시장 일별매매정보, `bon/kts_bydd_trd`)로
    국고채 10년 지표금리를 실시간 조회한다 — `_fetch_kr_10y_treasury_yield()` 참고. `KRX_API_KEY`가
    설정돼 있지 않거나 조회에 실패하면 확인 시점의 값을 폴백으로 쓴다.
    """
    if market == "US":
        import yfinance as yf
        hist = yf.Ticker("^TNX").history(period="5d")
        if hist.empty:
            raise ValueError("^TNX에서 미국 10년물 국채 수익률을 가져오지 못했다.")
        return hist["Close"].iloc[-1] / 100
    return _fetch_kr_10y_treasury_yield()


def _fetch_kr_10y_treasury_yield():
    """KRX Open API로 국고채 10년 지표금리(benchmark yield)를 가져온다.

    응답에는 만기 10년으로 표시된 종목이 두 개 섞여 나온다 — 하나는 물가연동국고채(종목명이 "물가"로
    시작, 실질금리라 훨씬 낮게 나온다)이고 실제로 원하는 건 종목명이 "국고"로 시작하는 명목 국고채다.
    이 둘을 구분하지 않고 그냥 만기(BND_EXP_TP_NM)만 보고 집으면 물가채 실질금리를 명목 무위험금리로
    잘못 쓰게 된다(실제로 라이브 응답을 까보고서야 발견한 문제).

    주말/공휴일은 그날 데이터가 없으므로 최근 영업일을 찾을 때까지 며칠 전으로 거슬러 올라간다.
    `KRX_API_KEY`가 없으면(선택적 기능이라 필수 설정은 아님) 폴백값을 쓴다.
    """
    import os
    from datetime import date, timedelta

    import requests
    from dotenv import load_dotenv

    from src.config import PROJECT_ROOT

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
    api_key = os.getenv("KRX_API_KEY")
    if not api_key:
        return _KR_RISK_FREE_RATE_FALLBACK

    for days_back in range(7):
        bas_dd = (date.today() - timedelta(days=days_back)).strftime("%Y%m%d")
        response = requests.get(
            "https://data-dbg.krx.co.kr/svc/apis/bon/kts_bydd_trd",
            params={"AUTH_KEY": api_key, "basDd": bas_dd}, timeout=10,
        )
        for row in response.json().get("OutBlock_1", []):
            if (
                row.get("BND_EXP_TP_NM") == "10"
                and row.get("GOVBND_ISU_TP_NM") == "지표"
                and row.get("ISU_NM", "").startswith("국고")
            ):
                return float(row["CLSPRC_YD"]) / 100

    return _KR_RISK_FREE_RATE_FALLBACK


_GROWTH_RATE_FALLBACK = 0.10  # revenueGrowth 필드가 없을 때 쓰는 보수적 기본값


def get_growth_rate_estimate(info, fallback=_GROWTH_RATE_FALLBACK):
    """향후 성장률 추정치를 가져온다. yfinance의 revenueGrowth(최근 TTM 매출 성장률)를 쓴다.

    실적발표 대본에서 회사 가이던스 텍스트를 파싱하는 방법도 고려했으나 범용성이 떨어져 버렸다 —
    transcript.py는 미국 대형·주목주(Motley Fool이 대본을 만드는 종목)만 되고 한국 종목은 아예 안
    된다. revenueGrowth는 market.get_company_info() 하나로 어떤 시장·종목이든 구조화된 숫자를 바로
    준다. 다만 이건 최근 추세를 그대로 연장하는 것뿐이라 향후 5년의 정교한 전망은 아니라는 한계는
    있다 — 그래도 하드코딩된 임의의 값보다는 실제 데이터에 근거한다.
    """
    growth = info.get("revenueGrowth")
    return growth if growth is not None else fallback


def compute_wacc(info, risk_free_rate, equity_risk_premium, tax_rate, cost_of_debt=None):
    """CAPM 자기자본비용과 세후 타인자본비용을 시가총액/부채 비중으로 가중평균한다.

    cost_of_debt를 안 주면 무위험금리 + 150bp(신용스프레드 근사치)를 쓴다.
    beta나 marketCap이 없으면(yfinance 필드 누락) None을 반환한다.
    """
    beta = info.get("beta")
    market_cap = info.get("marketCap")
    if beta is None or not market_cap:
        return None

    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    total_debt = info.get("totalDebt") or 0
    total_value = market_cap + total_debt

    weight_equity = market_cap / total_value
    weight_debt = total_debt / total_value
    cost_of_debt = cost_of_debt if cost_of_debt is not None else risk_free_rate + 0.015

    return weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)


def project_fcf(base_fcf, growth_rate, years=5):
    """base_fcf가 매년 growth_rate로 성장한다고 가정하고 향후 years년 FCF를 투영한다."""
    return [base_fcf * (1 + growth_rate) ** t for t in range(1, years + 1)]


def terminal_value(final_year_fcf, wacc, terminal_growth_rate):
    """Gordon Growth 모델 터미널가치. WACC가 터미널 성장률보다 작으면(현재가치 발산) 에러를 낸다."""
    if wacc <= terminal_growth_rate:
        raise ValueError(
            f"WACC({wacc})는 터미널 성장률({terminal_growth_rate})보다 커야 한다 — 안 그러면 "
            "터미널가치가 무한대로 발산한다."
        )
    return final_year_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)


def discount_cash_flows(cash_flows, wacc):
    """미래현금흐름 리스트를 wacc로 할인해 현재가치 합을 구한다. cash_flows[0]이 1년 후 현금흐름이다."""
    return sum(cf / (1 + wacc) ** t for t, cf in enumerate(cash_flows, start=1))


def run_dcf(base_fcf, wacc, growth_rate, terminal_growth_rate, years=5, net_debt=0, shares_outstanding=None):
    """DCF 파이프라인: FCF 투영 -> PV(FCF) + PV(터미널가치) -> 기업가치 -> 자기자본가치 -> (옵션)내재주가."""
    projected = project_fcf(base_fcf, growth_rate, years)
    tv = terminal_value(projected[-1], wacc, terminal_growth_rate)

    pv_fcf = discount_cash_flows(projected, wacc)
    pv_terminal_value = tv / (1 + wacc) ** years

    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value - net_debt

    result = {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "pv_fcf": pv_fcf,
        "pv_terminal_value": pv_terminal_value,
    }
    if shares_outstanding:
        result["implied_share_price"] = equity_value / shares_outstanding
    return result

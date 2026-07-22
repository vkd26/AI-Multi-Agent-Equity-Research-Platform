"""LLM에게 등록할 도구(tool) 정의. planner.run_agent_loop()에 넘길 {이름: 함수} 레지스트리를 만든다.

이 프로젝트의 실제 로직(dart/sec/market 수집, financial_analyzer, valuation, retrieval 등)은 이미
①~⑩ 단계에서 만들고 검증했다 — 여기서는 그걸 LLM이 호출하기 좋은 "단순 타입 인자/반환값" 인터페이스로
얇게 감싸기만 한다. google-genai는 함수의 타입힌트+독스트링으로 도구 스키마를 자동 추론하므로, 인자와
반환값을 str/float/list/dict처럼 단순한 타입으로 유지해야 한다(DataFrame을 그대로 노출하면 스키마
추론이 안 된다).

"전체 리포트를 만들어줘" 같은 요청은 8~9개 도구를 LLM이 일일이 순서를 재발견하게 하는 것보다,
이미 검증된 고정 파이프라인(generate_full_report)을 통째로 도구 하나로 노출하는 게 더 안정적이다 —
나머지는 가벼운 질문에 빠르게 답하기 위한 세분화된 조회용 도구들이다.
"""
from src.analytics.earnings_call_analyzer import summarize_topic
from src.analytics.financial_analyzer import summarize as summarize_financials
from src.analytics.news_analyzer import classify_news, summarize_classification
from src.collector import dart, market, news, sec, transcript
from src.config import market_of
from src.rag.chunking import chunk_dataframe
from src.rag.embedding import embed_chunks
from src.rag.vector_store import VectorStore
from src.valuation.comps import PeerSelectionError, comps_valuation, find_peer_tickers
from src.valuation.dcf import compute_wacc, get_equity_risk_premium, get_growth_rate_estimate, get_risk_free_rate, run_dcf


def get_stock_price_and_info(ticker: str) -> dict:
    """티커의 현재가, 시가총액, 통화 등 기본 시세 정보를 조회한다."""
    info = market.get_company_info(ticker)
    return {
        "ticker": ticker,
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
    }


def get_financial_ratios(ticker: str) -> dict:
    """매출성장률, 마진, ROE, PER, PEG, EV/EBITDA, Debt/Equity, FCF 등 표준 재무비율을 조회한다."""
    info = market.get_company_info(ticker)
    return summarize_financials(info)


def get_recent_material_disclosures(ticker: str) -> list:
    """최근 중요 공시 목록을 조회한다 (한국: 주요사항보고, 미국: 8-K). 각 항목은 제목/날짜를 담은 dict."""
    if market_of(ticker) == "KR":
        stock_code = ticker.split(".")[0]
        df = dart.download_disclosures(stock_code=stock_code, kind="B")
        return df[["report_nm", "rcept_dt"]].rename(columns={"report_nm": "title", "rcept_dt": "date"}).to_dict("records")
    df = sec.download_filings(ticker, forms=("8-K",))
    return df[["form", "filingDate"]].rename(columns={"form": "title", "filingDate": "date"}).to_dict("records")


def get_news_tone_summary(ticker: str, company_name: str) -> dict:
    """최근 뉴스를 논조(Positive/Negative/Neutral)와 영향도(High/Medium/Low)로 분류해 요약한다."""
    if market_of(ticker) == "KR":
        news_df = news.search_naver_news(company_name)
        title_col = "title"
    else:
        news_df = news.search_finnhub_news(ticker, from_date=_last_30_days(), to_date=_today())
        title_col = "headline"

    classified = classify_news(news_df, company_name, title_col=title_col, desc_col=("description" if title_col == "title" else "summary"))
    return summarize_classification(classified, title_col=title_col)


def search_earnings_call(ticker: str, question: str) -> dict:
    """가장 최근 실적발표 컨퍼런스콜 대본에서 question과 관련된 발언을 찾아 근거 기반으로 요약한다.

    미국 상장 종목만 지원한다(한국 기업은 대본 소스가 없다 — README 참고).
    """
    if market_of(ticker) != "US":
        return {"summary": None, "citations": [], "error": "한국 종목은 실적발표 대본을 지원하지 않는다."}

    raw = transcript.download_transcript(ticker)
    chunks = chunk_dataframe(
        raw, text_col="text", metadata_cols=["speaker", "speaker_role", "section", "fiscal_quarter"],
        extra_metadata={"ticker": ticker}, merge_adjacent=True, speaker_col="speaker",
        section_col="section", speaker_type_col="speaker_type",
    )
    embedded = embed_chunks(chunks)
    store = VectorStore(dim=len(embedded.iloc[0]["embedding"]))
    store.add(embedded)
    return summarize_topic(store, question, ticker=ticker)


def run_dcf_valuation(ticker: str, terminal_growth_rate: float, growth_rate: float | None = None, years: int = 5) -> dict:
    """DCF로 내재주가를 계산한다. growth_rate/terminal_growth_rate는 소수(예: 0.15 = 15%)로 준다.

    growth_rate를 안 주면 yfinance의 revenueGrowth(최근 TTM 매출 성장률)로 자동 추정한다
    (get_growth_rate_estimate() 참고) — 특정 시나리오(예: "성장률 20%로 가정하면?")를 테스트하고
    싶을 때만 명시적으로 값을 넘기면 된다.

    무위험금리/에퀴티리스크프리미엄/세율 전부 시장(한국/미국)에 따라 다르게 적용한다 — 한국 종목에
    미국 가정을 그대로 쓰면 안 된다. 한국 세율 27.5%는 2026년 기준 과세표준 3천억원 초과 구간(대기업)
    실효세율(국세 25% + 지방소득세 2.5%). 무위험금리는 get_risk_free_rate()(미국은 yfinance ^TNX
    실시간 조회, 한국은 폴백값), 에퀴티리스크프리미엄은 get_equity_risk_premium()(Damodaran 데이터
    기준)으로 가져온다 — 둘 다 dcf.py 독스트링 참고.
    """
    info = market.get_company_info(ticker)
    market_name = market_of(ticker)
    risk_free_rate = get_risk_free_rate(market_name)
    equity_risk_premium = get_equity_risk_premium(market_name)
    tax_rate = 0.275 if market_name == "KR" else 0.21
    if growth_rate is None:
        growth_rate = get_growth_rate_estimate(info)
    wacc = compute_wacc(info, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, tax_rate=tax_rate)
    if wacc is None:
        return {"error": "beta 또는 시가총액 정보가 없어 WACC를 계산할 수 없다."}

    net_debt = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)
    result = run_dcf(
        base_fcf=info.get("freeCashflow"), wacc=wacc, growth_rate=growth_rate,
        terminal_growth_rate=terminal_growth_rate, years=years, net_debt=net_debt,
        shares_outstanding=info.get("sharesOutstanding"),
    )
    result["wacc"] = wacc
    return result


def run_comps_valuation(ticker: str, peer_tickers: list[str] | None = None) -> dict:
    """피어 그룹의 EV/EBITDA·PER 중앙값을 적용해 내재주가를 추정한다.

    peer_tickers를 안 주면 Gemini 검색으로 사업모델이 유사한 경쟁사를 찾고 yfinance로 실존 여부를
    교차검증해 자동 선정한다(find_peer_tickers 참고) — 특정 피어 그룹을 직접 테스트하고 싶을 때만
    명시적으로 값을 넘기면 된다.
    """
    target_info = market.get_company_info(ticker)
    if peer_tickers is None:
        company_name = target_info.get("longName") or target_info.get("shortName") or ticker
        try:
            peer_tickers = find_peer_tickers(company_name, ticker)
        except PeerSelectionError as e:
            return {"error": str(e)}
    peer_infos = {peer: market.get_company_info(peer) for peer in peer_tickers}
    result = comps_valuation(target_info, peer_infos)
    result.pop("peer_table", None)  # DataFrame은 도구 반환값에 못 담는다(LLM 스키마상 단순 타입만 가능)
    return result


def _today():
    from datetime import date
    return date.today().strftime("%Y-%m-%d")


def _last_30_days():
    from datetime import date, timedelta
    return (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")


TOOLS = {
    "get_stock_price_and_info": get_stock_price_and_info,
    "get_financial_ratios": get_financial_ratios,
    "get_recent_material_disclosures": get_recent_material_disclosures,
    "get_news_tone_summary": get_news_tone_summary,
    "search_earnings_call": search_earnings_call,
    "run_dcf_valuation": run_dcf_valuation,
    "run_comps_valuation": run_comps_valuation,
}

"""⑨ Portfolio Monitor — 관심종목을 매일 점검해 공시/뉴스/실적발표/목표주가 변동 중 하나라도 "이상"이
감지되면 플래그를 올린다.

목표주가는 매일 LLM으로 재생성하지 않는다. DCF/Comps의 숫자 부분(현재가·피어 배수는 매일 바뀌지만
계산 자체는 순수 함수라 비용이 거의 없다)만 호출 측에서 미리 재계산해 cheap_target_price로 넘기면,
이 모듈은 그 값을 이전 저장값과 비교만 한다. 네 트리거 중 하나라도 걸리면 need_full_regeneration=True를
반환하는데, 실제로 thesis_agent.generate_thesis()(LLM, 비쌈)를 호출할지는 호출 측이 결정한다 — 이
모듈 자체는 LLM을 쓰지 않는다.

상태(마지막으로 본 공시 번호, 뉴스 id, 분기, 목표주가)는 data/processed/portfolio_state/{ticker}.json에
보존해서, 다음 실행 때 "새로 생긴 것"을 판단하는 기준으로 쓴다.
"""
import json
import os

from src.collector import dart, sec, transcript
from src.config import DATA_DIR_PROCESSED, market_of

_STATE_DIR = os.path.join(DATA_DIR_PROCESSED, "portfolio_state")


def _state_path(ticker):
    return os.path.join(_STATE_DIR, f"{ticker.upper().replace('.', '_')}.json")


def load_state(ticker):
    path = _state_path(ticker)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(ticker, state):
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_state_path(ticker), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def check_material_disclosures(ticker, market, last_seen_id=None):
    """중요 공시(한국: 주요사항보고 kind="B", 미국: 8-K)가 새로 올라왔는지 확인한다.

    반환: (새 공시 DataFrame, 최신 id). id는 시간순 정렬이 되는 문자열(rcept_no/accessionNumber)이라
    문자열 비교로 "이전보다 최신인지"를 판단할 수 있다.
    """
    if market == "KR":
        stock_code = ticker.split(".")[0]
        df = dart.download_disclosures(stock_code=stock_code, kind="B", use_cache=False)
        id_col = "rcept_no"
    else:
        df = sec.download_filings(ticker, forms=("8-K",), use_cache=False)
        id_col = "accessionNumber"

    if df.empty:
        return df, last_seen_id

    df = df.sort_values(id_col)
    latest_id = df.iloc[-1][id_col]
    new_items = df[df[id_col] > last_seen_id] if last_seen_id else df
    return new_items, latest_id


def check_high_impact_news(classified_news_df, last_seen_ids=None, id_col="link"):
    """news_analyzer.classify_news() 결과에서 impact="High"인 기사 중 이전에 못 본 것만 뽑는다."""
    last_seen_ids = last_seen_ids or set()
    high_impact = classified_news_df[classified_news_df["impact"] == "High"]
    new_items = high_impact[~high_impact[id_col].isin(last_seen_ids)]
    seen_ids = set(classified_news_df[id_col].dropna())
    return new_items, seen_ids


def check_new_earnings_call(ticker, last_seen_quarter=None):
    """최근 실적발표 목록에서 이전에 못 본 분기의 대본이 새로 올라왔는지 확인한다.

    아직 실적발표가 없거나(find_transcript_url이 못 찾음) fool.com에 없는 경우는 "새 발표 없음"으로
    처리한다 — 이 함수가 죽으면 다른 세 트리거 체크까지 막히면 안 되기 때문이다.
    """
    try:
        url = transcript.find_transcript_url(ticker)
    except ValueError:
        return False, last_seen_quarter

    latest_quarter = transcript._parse_fiscal_quarter(url)
    is_new = latest_quarter is not None and latest_quarter != last_seen_quarter
    return is_new, (latest_quarter or last_seen_quarter)


def check_target_price_drift(new_target_price, last_target_price, threshold=0.10):
    """새로 계산한 목표주가가 이전 저장값 대비 threshold(기본 10%) 이상 벌어졌는지 확인한다.

    이전 저장값이 없으면(처음 체크하는 종목) 무조건 "이상"으로 취급해 최초 기록을 남긴다.
    """
    if not last_target_price:
        return True, None
    pct_change = new_target_price / last_target_price - 1
    return abs(pct_change) >= threshold, pct_change


def run_daily_check(ticker, cheap_target_price, classified_news_df=None, news_id_col=None):
    """네 가지 트리거(공시/뉴스/실적발표/목표주가)를 한 번에 확인하고 상태를 갱신한다.

    classified_news_df를 안 주면 뉴스 트리거는 항상 False로 남는다(호출 측에서 아직 뉴스를 안
    돌렸을 수 있으므로 — news_analyzer.classify_news()가 LLM 호출이라 매번 자동으로 돌리진 않는다).

    news_id_col: classified_news_df에서 기사 고유 식별자로 쓸 컬럼명. 안 주면 시장에 따라 자동으로
    고른다 — 한국(네이버)은 "link", 미국(Finnhub)은 "url" 컬럼을 쓴다(news.py 참고, 둘이 스키마가
    다르다).
    """
    state = load_state(ticker)
    market = market_of(ticker)
    news_id_col = news_id_col or ("link" if market == "KR" else "url")

    new_disclosures, latest_disclosure_id = check_material_disclosures(
        ticker, market, last_seen_id=state.get("last_disclosure_id")
    )

    seen_news_ids = set(state.get("seen_news_ids", []))
    if classified_news_df is not None:
        new_high_impact_news, seen_news_ids = check_high_impact_news(
            classified_news_df, last_seen_ids=seen_news_ids, id_col=news_id_col
        )
    else:
        new_high_impact_news = None

    has_new_earnings, latest_quarter = check_new_earnings_call(
        ticker, last_seen_quarter=state.get("last_fiscal_quarter")
    )

    price_drifted, pct_change = check_target_price_drift(cheap_target_price, state.get("last_target_price"))

    triggers = {
        "material_disclosure": not new_disclosures.empty,
        "high_impact_news": new_high_impact_news is not None and not new_high_impact_news.empty,
        "new_earnings_call": has_new_earnings,
        "target_price_drift": price_drifted,
    }

    save_state(ticker, {
        "last_disclosure_id": latest_disclosure_id or state.get("last_disclosure_id"),
        "seen_news_ids": list(seen_news_ids),
        "last_fiscal_quarter": latest_quarter,
        "last_target_price": cheap_target_price,
    })

    return {
        "ticker": ticker,
        "triggers": triggers,
        "need_full_regeneration": any(triggers.values()),
        "new_disclosures": new_disclosures,
        "new_high_impact_news": new_high_impact_news,
        "target_price_pct_change": pct_change,
    }

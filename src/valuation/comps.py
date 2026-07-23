"""⑦ Valuation Engine — Comparable Company Analysis. 피어 그룹의 밸류에이션 배수(중앙값)를 대상 기업의
실적(EBITDA, EPS)에 적용해 내재가치를 추정한다.

피어 그룹 선정(find_peer_tickers)은 예전엔 사람이 직접 티커를 골라 하드코딩했다 — 체계적 기준이
없었다. Gemini의 google_search 도구로 사업모델이 유사한 경쟁사를 찾게 하되, LLM이 존재하지 않는
티커를 지어낼 수 있으니 yfinance로 실제 데이터를 받아올 수 있는지 다시 한번 교차검증한다.
"""
import pandas as pd


class PeerSelectionError(Exception):
    """검증을 통과한 피어가 min_peers 미만일 때 발생한다 — 상대가치평가(median multiple)는 최소
    2개 이상의 피어가 있어야 중앙값 산출 자체가 의미를 가지므로, 그 아래로는 계산을 진행하면 안 된다."""


def _validate_peer_ticker(ticker):
    """yfinance로 실제 데이터를 받아올 수 있는 티커인지 확인한다(LLM이 지어낸 티커를 걸러내기 위함)."""
    from src.collector import market
    try:
        market.get_company_info(ticker)
        return True
    except Exception:
        return False


def _is_kr_or_us_ticker(ticker):
    """한국(.KS/.KQ 접미사) 또는 미국(접미사 없는 일반 표기, 예: 'AMD') 티커인지 확인한다.

    src.config.market_of()는 ".KS"/".KQ"가 아니면 전부 "US"로 보는 이분법 함수라(원래 DART/SEC 중
    어느 수집기를 쓸지 정하는 용도) 여기 그대로 쓰면 0981.HK 같은 홍콩 티커도 "US"로 오분류돼서
    걸러지지 않는다 — 그래서 별도로 접미사 유무를 직접 확인한다.
    """
    if "." not in ticker:
        return True  # 미국은 보통 접미사 없이 표기(AMD, NVDA 등)
    return ticker.upper().endswith((".KS", ".KQ"))


def _extract_ticker_candidates(text):
    """LLM 응답의 마지막 줄(쉼표로 구분된 티커 목록)에서 후보를 뽑는다."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []
    tokens = [t.strip().strip("*").strip() for t in lines[-1].split(",")]
    return [t for t in tokens if t and " " not in t and len(t) <= 12]


def find_peer_tickers(company_name, ticker, client=None, model="gemini-flash-latest", min_peers=2, max_peers=5):
    """Gemini의 google_search 도구로 사업모델이 유사한 상장 경쟁사 후보를 찾고, yfinance로 실존
    여부를 교차검증해 거른 뒤 반환한다. 한국·미국 증시 상장 종목만 대상으로 한다 — 그 외 시장(홍콩,
    대만 등)은 yfinance 데이터 정합성(예: 통화 필드 불일치)이 검증되지 않아 제외한다.

    검증된 피어가 min_peers(기본 2) 미만이면 PeerSelectionError를 낸다 — 상대가치평가는 피어가
    최소 2개는 있어야 중앙값 배수를 산출하는 의미가 있다.
    """
    from google.genai import types

    from src.analytics._common import get_llm_client

    client = client or get_llm_client()
    prompt = (
        f"'{company_name}'({ticker})와 사업모델이 가장 유사한 상장 경쟁사(Comparable Company) 티커를 "
        f"최대 {max_peers}개 찾아라. 반드시 한국 증시(코스피/코스닥) 또는 미국 증시(NYSE/NASDAQ)에 "
        f"상장된 종목만 골라라 — 그 외 거래소(홍콩, 대만, 유럽 등)는 제외하라. "
        f"실제 존재하는 야후 파이낸스 티커 형식으로 답하라(예: 미국은 'AMD', 한국은 '000660.KS'). "
        f"'{ticker}' 자기 자신은 제외하라. "
        f"마지막 줄에 쉼표로 구분된 티커 목록만 다시 한 번 적어라(다른 텍스트 없이)."
    )
    response = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )

    validated = []
    for candidate in _extract_ticker_candidates(response.text):
        if candidate.upper() == ticker.upper():
            continue
        if not _is_kr_or_us_ticker(candidate):
            continue
        if _validate_peer_ticker(candidate):
            validated.append(candidate)
        if len(validated) >= max_peers:
            break

    if len(validated) < min_peers:
        raise PeerSelectionError(
            f"검증된 피어가 {len(validated)}개뿐이다(최소 {min_peers}개 필요) — "
            "상대가치평가(median multiple)를 산출할 수 없다."
        )
    return validated


def peer_multiples_table(peer_infos, multiples=("enterpriseToEbitda", "trailingPE")):
    """peer_infos: {ticker: yfinance company info dict}. 피어별 배수를 모은 표와 평균/중앙값 요약을 만든다."""
    rows = []
    for ticker, info in peer_infos.items():
        row = {"ticker": ticker}
        row.update({m: info.get(m) for m in multiples})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("ticker")
    summary = table.agg(["mean", "median"])
    return table, summary


def implied_value_from_multiple(target_metric, peer_multiple, metric_name="지표"):
    """피어 배수 × 대상 기업의 해당 지표(EBITDA, EPS 등) = 내재 가치.

    target_metric이나 peer_multiple이 음수면 계산하지 않고 (None, 사유) 를 반환한다 — PER·EV/EBITDA
    같은 이익 기반 배수는 분자·분모 중 하나라도 적자(음수)면 수학적으로는 곱해지더라도 경제적으로
    무의미한 값(예: 마이너스 "내재주가")이 나온다. 실제로 적자 기업(Wolfspeed, EBITDA/EPS 둘 다 음수)에
    적용했더니 내재주가가 -$120~-$2,012로 나오는 걸 확인했다 — 조용히 틀린 숫자를 주는 대신 계산을
    건너뛰고 이유를 알려준다.
    """
    if target_metric is None or peer_multiple is None:
        return None, None
    if target_metric < 0:
        return None, f"대상 기업의 {metric_name}이(가) 음수(적자)라 배수 적용이 무의미해 계산하지 않았다."
    if peer_multiple < 0:
        return None, f"피어 그룹의 {metric_name} 배수 중앙값이 음수라 계산하지 않았다."
    return target_metric * peer_multiple, None


def comps_valuation(target_info, peer_infos, ebitda_multiple_col="enterpriseToEbitda", pe_multiple_col="trailingPE"):
    """피어 그룹의 EV/EBITDA·PER 중앙값을 대상 기업에 적용해 내재주가를 두 가지 방식으로 추정한다."""
    peer_table, peer_summary = peer_multiples_table(peer_infos, multiples=(ebitda_multiple_col, pe_multiple_col))
    median_ev_ebitda = peer_summary.loc["median", ebitda_multiple_col]
    median_pe = peer_summary.loc["median", pe_multiple_col]

    warnings = []
    implied_ev, ev_ebitda_warning = implied_value_from_multiple(target_info.get("ebitda"), median_ev_ebitda, "EBITDA")
    implied_price_from_pe, pe_warning = implied_value_from_multiple(target_info.get("trailingEps"), median_pe, "EPS")
    if ev_ebitda_warning:
        warnings.append(ev_ebitda_warning)
    if pe_warning:
        warnings.append(pe_warning)

    net_debt = (target_info.get("totalDebt") or 0) - (target_info.get("totalCash") or 0)
    shares = target_info.get("sharesOutstanding")
    implied_price_from_ev_ebitda = None
    if implied_ev is not None and shares:
        implied_price_from_ev_ebitda = (implied_ev - net_debt) / shares

    return {
        "peer_table": peer_table,
        "median_ev_ebitda": median_ev_ebitda,
        "median_pe": median_pe,
        "implied_price_from_ev_ebitda": implied_price_from_ev_ebitda,
        "implied_price_from_pe": implied_price_from_pe,
        "warnings": warnings,
    }

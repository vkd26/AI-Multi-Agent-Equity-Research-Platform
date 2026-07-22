"""yfinance를 이용한 시세·기업정보 수집 (한국/미국 공통 — 티커 형식만 다르고 API는 동일하다).

계산(수익률, 변동성, 밸류에이션 배수 등)은 하지 않는다. 원시 가격/기업정보를 받아 캐싱하는 것까지만
담당하고, 파생 지표는 src/analytics/에서 계산한다.
"""
import json
import os

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR_RAW


def download_price_history(ticker, period="1y", interval="1d", use_cache=True):
    """OHLCV 일별(또는 지정 interval) 시세를 받아온다."""
    cache_path = os.path.join(DATA_DIR_RAW, f"market_prices_{ticker.upper()}_{period}_{interval}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"{ticker}의 시세 데이터를 가져오지 못했다.")

    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    df.to_csv(cache_path)
    return df


def get_company_info(ticker, use_cache=True):
    """시가총액/섹터/통화/발행주식수 등 yfinance의 기업 개요 정보를 받아온다.

    반환하기 직전에 normalize_financial_currency()를 적용한다 — 원본 캐시 파일(디스크)에는 yfinance가
    준 그대로의 raw 값을 저장하고, 통화 환산은 매번 반환 시점에 적용해 캐시 자체는 순수 raw 데이터로
    유지한다(이중 환산될 위험이 없다).
    """
    cache_path = os.path.join(DATA_DIR_RAW, f"market_info_{ticker.upper()}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return normalize_financial_currency(json.load(f))

    info = yf.Ticker(ticker).info
    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        raise ValueError(f"{ticker}의 기업정보를 가져오지 못했다 — 티커가 유효한지 확인할 것.")

    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return normalize_financial_currency(info)


# TSM(대만 ADR) 같은 종목은 info['currency']（주가·시가총액 통화, 예: USD）와
# info['financialCurrency']（재무제표 보고 통화, 예: TWD）가 다르다 — 실제로 TSM은 currency=USD인데
# freeCashflow/ebitda/totalRevenue/totalDebt/totalCash는 전부 TWD 원본 그대로 온다. 이걸 모르고
# DCF/Comps에서 그대로 USD처럼 나누면 내재주가가 실제 주가의 20~30배로 부풀려진다(실제로 TSM에서
# 이 버그로 DCF 내재주가가 $10,136까지 나온 적이 있다 — 정상 범위는 주가와 비슷한 수백 달러대).
# margin/growth/PER/PEG 같은 비율 필드는 분자·분모가 같은 통화라 이 문제와 무관하다.
_FINANCIAL_STATEMENT_MONEY_FIELDS = (
    "freeCashflow", "operatingCashflow", "totalCash", "totalDebt", "totalRevenue", "ebitda",
)


def get_fx_rate(from_currency, to_currency):
    """from_currency 1단위가 to_currency로 얼마인지 실시간 환율을 가져온다(하드코딩하지 않음 —
    get_risk_free_rate()가 ^TNX를 실시간 조회하는 것과 같은 원칙)."""
    if from_currency == to_currency:
        return 1.0
    hist = yf.Ticker(f"{from_currency}{to_currency}=X").history(period="5d")
    if hist.empty:
        raise ValueError(f"{from_currency}->{to_currency} 환율을 가져오지 못했다.")
    return hist["Close"].iloc[-1]


def normalize_financial_currency(info):
    """financialCurrency(재무제표 보고 통화)가 currency(주가/시가총액 통화)와 다르면, 절대금액
    재무제표 필드(_FINANCIAL_STATEMENT_MONEY_FIELDS)만 currency 기준으로 환산한 새 dict를 반환한다.
    둘이 같거나 필드가 없으면 원본을 그대로 반환한다."""
    price_currency = info.get("currency")
    financial_currency = info.get("financialCurrency")
    if not price_currency or not financial_currency or price_currency == financial_currency:
        return info

    rate = get_fx_rate(financial_currency, price_currency)
    converted = dict(info)
    for field in _FINANCIAL_STATEMENT_MONEY_FIELDS:
        value = info.get(field)
        if value is not None:
            converted[field] = value * rate
    return converted

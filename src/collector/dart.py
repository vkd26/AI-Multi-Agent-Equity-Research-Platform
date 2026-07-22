"""금융감독원 OpenDART API를 이용한 재무제표·공시 수집.

사용하려면 https://opendart.fss.or.kr 에서 무료로 API 키를 발급받아 `DART_API_KEY=발급받은키` 형태로 저장해야
한다. 키는 두 곳 중 아무데나 둬도 된다 (`.env`는 둘 다 `.gitignore`에 포함되어 git에 올라가지 않는다):

- 프로젝트 루트의 `.env` — 이 프로젝트에만 적용되는 키를 쓰고 싶을 때
- 홈 디렉토리(`~/.env`, 예: `C:/Users/<사용자명>/.env`) — 다른 Korean-stock 프로젝트에서도 이 모듈을 복사해
  재사용할 때마다 키를 새로 설정하지 않도록 한 번만 저장해두는 전역 기본값. 프로젝트 루트에 `.env`가 있으면
  그쪽이 우선한다.
"""
import os

import pandas as pd
from dotenv import load_dotenv

from src.config import DART_STOCK_CODE, DATA_DIR_RAW, END_DATE, PROJECT_ROOT, START_DATE, TICKER

REPRT_CODE = {"Q1": "11013", "H1": "11012", "Q3": "11014", "FY": "11011"}


def _get_client():
    # 프로젝트 로컬 .env를 먼저 시도하고(override=False가 기본이라 이미 세팅된 값은 안 덮어씀),
    # 없으면 홈 디렉토리의 전역 .env로 폴백한다 — 다른 프로젝트에 이 모듈을 복사해도 키를 또 만들 필요가 없다.
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DART_API_KEY가 설정되어 있지 않다. https://opendart.fss.or.kr 에서 API 키를 발급받아 "
            "프로젝트 루트의 .env 또는 홈 디렉토리(~/.env)에 DART_API_KEY=발급받은키 형태로 저장한 뒤 "
            "다시 시도할 것."
        )
    import opendartreader as odr  # 선택적 의존성 — DART 기능을 쓰지 않으면 설치가 필요 없다
    return odr.OpenDartReader(api_key)


def download_quarterly_financials(ticker=TICKER, year=None, period="Q1", fs_div="CFS", use_cache=True):
    """분기/반기/사업보고서 원본 재무제표 전체(raw long-format)를 받아온다.

    period: "Q1"(1분기보고서) / "H1"(반기보고서) / "Q3"(3분기보고서) / "FY"(사업보고서, 연간)
    fs_div: "CFS"(연결재무제표) / "OFS"(별도재무제표)
    """
    stock_code = ticker.split(".")[0]
    year = year or pd.Timestamp.today().year
    reprt_code = REPRT_CODE[period]

    cache_path = os.path.join(DATA_DIR_RAW, f"dart_{stock_code}_{year}_{period}_{fs_div}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    dart = _get_client()
    df = dart.finstate_all(stock_code, year, reprt_code=reprt_code, fs_div=fs_div)
    if df is None or df.empty:
        raise ValueError(f"{ticker} {year} {period}({fs_div}) 재무제표를 찾을 수 없다 — 아직 공시 전이거나 파라미터 오류일 수 있다.")

    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def download_disclosures(stock_code=DART_STOCK_CODE, start_date=None, end_date=None, kind="", use_cache=True):
    """지정 기간 공시 목록(제목/접수일/보고서명/링크)을 받아온다.

    start_date/end_date: "YYYYMMDD" 문자열. 생략하면 config.START_DATE/END_DATE를 90일 룩백으로 사용.
    kind: DART 공시상세유형코드 (예: "A"=정기공시, "B"=주요사항보고, ""=전체)
    """
    start_date = start_date or START_DATE.strftime("%Y%m%d")
    end_date = end_date or END_DATE.strftime("%Y%m%d")

    cache_path = os.path.join(DATA_DIR_RAW, f"dart_disclosures_{stock_code}_{start_date}_{end_date}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    dart = _get_client()
    df = dart.list(stock_code, start=start_date, end=end_date, kind=kind)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["rcept_no", "report_nm", "rcept_dt", "flr_nm"])
    else:
        os.makedirs(DATA_DIR_RAW, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df

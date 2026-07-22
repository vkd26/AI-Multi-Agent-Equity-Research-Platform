"""SEC EDGAR API를 이용한 미국 기업 공시(10-K/10-Q/8-K 등) 및 XBRL 재무데이터 수집.

SEC EDGAR는 OpenDART와 달리 발급받는 API 키가 없다. 대신 요청마다 신원을 밝히는 User-Agent 헤더를
요구한다(https://www.sec.gov/os/webmaster-faq#developers) — 기본값(_DEFAULT_USER_AGENT)으로도 동작하지만,
운영 환경에서는 `SEC_USER_AGENT=앱이름 연락처이메일` 형태로 .env(프로젝트 루트 또는 ~/.env)에 저장해 본인
정보로 바꾸는 걸 권장한다 (SEC가 남용 IP를 차단할 때 식별 근거로 쓰는 값이라 실제 연락처를 넣는 편이 안전).
"""
import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import DATA_DIR_RAW, PROJECT_ROOT

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_DEFAULT_USER_AGENT = "ai-equity-research-agent research-agent@example.com"


def _get_headers():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
    return {"User-Agent": os.getenv("SEC_USER_AGENT", _DEFAULT_USER_AGENT)}


def _load_ticker_map(use_cache=True):
    cache_path = os.path.join(DATA_DIR_RAW, "sec_company_tickers.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        resp = requests.get(_TICKER_MAP_URL, headers=_get_headers(), timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        os.makedirs(DATA_DIR_RAW, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
    return pd.DataFrame(raw.values())


def _get_cik(ticker, use_cache=True):
    """티커(예: "NVDA")를 10자리 zero-padded CIK 문자열로 변환한다."""
    mapping = _load_ticker_map(use_cache=use_cache)
    match = mapping[mapping["ticker"].str.upper() == ticker.upper()]
    if match.empty:
        raise ValueError(f"SEC EDGAR에서 티커 '{ticker}'를 찾을 수 없다.")
    return f"{int(match.iloc[0]['cik_str']):010d}"


def download_filings(ticker, forms=("10-K", "10-Q"), count=10, use_cache=True):
    """최근 공시 목록(폼타입/제출일/accession number 등)을 받아온다."""
    cache_path = os.path.join(DATA_DIR_RAW, f"sec_filings_{ticker.upper()}.csv")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        cik = _get_cik(ticker, use_cache=use_cache)
        resp = requests.get(_SUBMISSIONS_URL.format(cik=cik), headers=_get_headers(), timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["filings"]["recent"])
        os.makedirs(DATA_DIR_RAW, exist_ok=True)
        df.to_csv(cache_path, index=False)

    df = df[df["form"].isin(forms)].sort_values("filingDate", ascending=False)
    return df.head(count).reset_index(drop=True)


def download_company_facts(ticker, tags=("Revenues", "NetIncomeLoss", "OperatingIncomeLoss"), use_cache=True):
    """XBRL us-gaap 태그별 분기/연간 실적 값을 tidy 포맷(long-format)으로 받아온다."""
    cache_path = os.path.join(DATA_DIR_RAW, f"sec_facts_{ticker.upper()}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    cik = _get_cik(ticker, use_cache=use_cache)
    resp = requests.get(_COMPANY_FACTS_URL.format(cik=cik), headers=_get_headers(), timeout=10)
    resp.raise_for_status()
    facts = resp.json().get("facts", {}).get("us-gaap", {})

    rows = []
    for tag in tags:
        if tag not in facts:
            continue
        for unit, entries in facts[tag]["units"].items():
            for e in entries:
                rows.append({
                    "tag": tag,
                    "unit": unit,
                    "val": e.get("val"),
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "fy": e.get("fy"),
                    "fp": e.get("fp"),
                    "form": e.get("form"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{ticker}의 XBRL 데이터에서 태그 {tags}를 찾을 수 없다.")

    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df

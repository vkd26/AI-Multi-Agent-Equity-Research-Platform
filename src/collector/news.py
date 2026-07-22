"""뉴스 수집 — 한국은 네이버 뉴스 검색 API, 미국은 Finnhub company-news를 사용한다.

네이버: https://developers.naver.com 에서 애플리케이션을 등록하면 Client ID/Secret을 무료로 발급받을 수 있다.
Finnhub: https://finnhub.io 에서 이메일만으로 무료 API 키를 발급받을 수 있다.
둘 다 dart.py와 동일한 패턴으로 프로젝트 루트의 .env 또는 홈 디렉토리(~/.env)에 저장하면 된다.
"""
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import DATA_DIR_RAW, PROJECT_ROOT

_NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"
_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_TAG_RE = re.compile(r"<[^>]+>")


def _load_env():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))


def _strip_html(text):
    """네이버 검색 API가 검색어를 <b>태그로 감싸 반환하는 것과 HTML 엔티티(&quot; 등)를 제거한다."""
    text = _TAG_RE.sub("", text)
    return (
        text.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&apos;", "'")
    )


def search_naver_news(query, display=30, sort="date", use_cache=True):
    """네이버 뉴스 검색 API로 국내 뉴스를 가져온다.

    sort: "date"(최신순) / "sim"(정확도순). display: 검색 결과 개수(최대 100)
    """
    cache_path = os.path.join(DATA_DIR_RAW, f"news_naver_{query}_{sort}_{display}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    _load_env()
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "NAVER_CLIENT_ID/NAVER_CLIENT_SECRET이 설정되어 있지 않다. https://developers.naver.com 에서 "
            "애플리케이션을 등록해 키를 발급받은 뒤 프로젝트 루트의 .env 또는 홈 디렉토리(~/.env)에 저장할 것."
        )

    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {"query": query, "display": display, "sort": sort}
    resp = requests.get(_NAVER_API_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])

    rows = [
        {
            "title": _strip_html(item["title"]),
            "description": _strip_html(item["description"]),
            "link": item["originallink"] or item["link"],
            "pub_date": item["pubDate"],
        }
        for item in items
    ]
    df = pd.DataFrame(rows, columns=["title", "description", "link", "pub_date"])
    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def search_finnhub_news(ticker, from_date, to_date, use_cache=True):
    """Finnhub company-news로 미국 상장 종목의 뉴스를 가져온다.

    from_date/to_date: "YYYY-MM-DD" 문자열
    """
    cache_path = os.path.join(DATA_DIR_RAW, f"news_finnhub_{ticker.upper()}_{from_date}_{to_date}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    _load_env()
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FINNHUB_API_KEY가 설정되어 있지 않다. https://finnhub.io 에서 무료 API 키를 발급받아 "
            "프로젝트 루트의 .env 또는 홈 디렉토리(~/.env)에 FINNHUB_API_KEY=발급받은키 형태로 저장할 것."
        )

    params = {"symbol": ticker.upper(), "from": from_date, "to": to_date, "token": api_key}
    resp = requests.get(_FINNHUB_NEWS_URL, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json()

    df = pd.DataFrame(items, columns=["datetime", "headline", "summary", "source", "url", "category"])
    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df

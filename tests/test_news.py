"""news.py의 캐싱/에러 처리/파싱 로직을 검증한다 (네트워크 미사용, requests.get은 모킹)."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector import news


def test_strip_html_removes_tags_and_entities():
    assert news._strip_html("<b>삼성전자</b> &quot;실적&quot; 발표") == '삼성전자 "실적" 발표'


def test_search_naver_news_uses_cache_without_calling_network(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "news_naver_삼성전자_date_10.csv"
    pd.DataFrame({"title": ["cached"], "description": [""], "link": [""], "pub_date": [""]}).to_csv(
        cache_path, index=False
    )

    with patch.object(news, "requests") as mock_requests:
        result = news.search_naver_news("삼성전자", display=10)

    mock_requests.get.assert_not_called()
    assert result.loc[0, "title"] == "cached"


def test_search_naver_news_raises_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "DATA_DIR_RAW", str(tmp_path))
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(news, "_load_env", lambda: None)

    with pytest.raises(RuntimeError):
        news.search_naver_news("삼성전자", use_cache=False)


def test_search_naver_news_parses_and_strips_html(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "DATA_DIR_RAW", str(tmp_path))
    monkeypatch.setattr(news, "_load_env", lambda: None)
    monkeypatch.setenv("NAVER_CLIENT_ID", "id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "secret")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [{
            "title": "<b>삼성전자</b> 실적",
            "description": "설명 &amp; 내용",
            "originallink": "https://example.com/a",
            "link": "https://news.naver.com/a",
            "pubDate": "Wed, 22 Jul 2026 13:00:00 +0900",
        }]
    }
    with patch.object(news.requests, "get", return_value=mock_response):
        result = news.search_naver_news("삼성전자", use_cache=False)

    assert result.loc[0, "title"] == "삼성전자 실적"
    assert result.loc[0, "description"] == "설명 & 내용"
    assert result.loc[0, "link"] == "https://example.com/a"


def test_search_finnhub_news_uses_cache_without_calling_network(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "news_finnhub_NVDA_2026-07-01_2026-07-22.csv"
    pd.DataFrame({"headline": ["cached"], "source": ["Yahoo"]}).to_csv(cache_path, index=False)

    with patch.object(news, "requests") as mock_requests:
        result = news.search_finnhub_news("NVDA", from_date="2026-07-01", to_date="2026-07-22")

    mock_requests.get.assert_not_called()
    assert result.loc[0, "headline"] == "cached"


def test_search_finnhub_news_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "DATA_DIR_RAW", str(tmp_path))
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(news, "_load_env", lambda: None)

    with pytest.raises(RuntimeError):
        news.search_finnhub_news("NVDA", from_date="2026-07-01", to_date="2026-07-22", use_cache=False)

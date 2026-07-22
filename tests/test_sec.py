"""sec.py의 CIK 매핑/캐싱/에러 처리 로직을 검증한다 (네트워크 미사용, requests.get은 모킹)."""
from unittest.mock import patch

import pandas as pd
import pytest

from src.collector import sec


def _synthetic_ticker_map():
    return pd.DataFrame([
        {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    ])


def test_get_cik_finds_ticker_case_insensitively(monkeypatch):
    monkeypatch.setattr(sec, "_load_ticker_map", lambda use_cache=True: _synthetic_ticker_map())
    assert sec._get_cik("nvda") == "0001045810"


def test_get_cik_raises_for_unknown_ticker(monkeypatch):
    monkeypatch.setattr(sec, "_load_ticker_map", lambda use_cache=True: _synthetic_ticker_map())
    with pytest.raises(ValueError):
        sec._get_cik("UNKNOWN")


def test_download_filings_uses_cache_without_calling_network(tmp_path, monkeypatch):
    monkeypatch.setattr(sec, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "sec_filings_NVDA.csv"
    pd.DataFrame({
        "form": ["10-K", "10-Q", "8-K"],
        "filingDate": ["2026-02-25", "2026-05-20", "2026-06-01"],
    }).to_csv(cache_path, index=False)

    with patch.object(sec, "requests") as mock_requests:
        result = sec.download_filings("NVDA", forms=("10-K", "10-Q"), count=10)

    mock_requests.get.assert_not_called()
    assert set(result["form"]) == {"10-K", "10-Q"}
    assert list(result["filingDate"]) == ["2026-05-20", "2026-02-25"]  # 최신순 정렬


def test_download_company_facts_raises_when_tags_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sec, "DATA_DIR_RAW", str(tmp_path))
    monkeypatch.setattr(sec, "_get_cik", lambda ticker, use_cache=True: "0001045810")

    mock_response = type("MockResponse", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"facts": {"us-gaap": {}}},
    })()

    with patch.object(sec.requests, "get", return_value=mock_response):
        with pytest.raises(ValueError):
            sec.download_company_facts("NVDA", tags=("Revenues",), use_cache=False)

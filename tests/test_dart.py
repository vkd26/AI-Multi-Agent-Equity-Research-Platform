"""dart.py의 캐싱/에러 처리 로직을 검증한다 (네트워크 미사용, _get_client는 모킹)."""
from unittest.mock import patch

import pandas as pd
import pytest

from src.collector import dart


def test_download_disclosures_uses_cache_without_calling_client(tmp_path, monkeypatch):
    monkeypatch.setattr(dart, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "dart_disclosures_005930_20260101_20260201.csv"
    pd.DataFrame({"rcept_no": ["1"], "report_nm": ["test"]}).to_csv(cache_path, index=False)

    with patch.object(dart, "_get_client") as mock_client:
        result = dart.download_disclosures(stock_code="005930", start_date="20260101", end_date="20260201")

    mock_client.assert_not_called()
    assert result.loc[0, "report_nm"] == "test"


def test_download_disclosures_returns_empty_df_when_api_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(dart, "DATA_DIR_RAW", str(tmp_path))
    mock_dart = type("MockDart", (), {"list": lambda self, *a, **k: None})()

    with patch.object(dart, "_get_client", return_value=mock_dart):
        result = dart.download_disclosures(
            stock_code="005930", start_date="20260101", end_date="20260201", use_cache=False
        )

    assert result.empty
    assert list(result.columns) == ["rcept_no", "report_nm", "rcept_dt", "flr_nm"]


def test_download_quarterly_financials_uses_cache_without_calling_client(tmp_path, monkeypatch):
    monkeypatch.setattr(dart, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "dart_005930_2025_FY_CFS.csv"
    pd.DataFrame({"account_id": ["ifrs-full_Revenue"], "thstrm_amount": ["1000"]}).to_csv(cache_path, index=False)

    with patch.object(dart, "_get_client") as mock_client:
        result = dart.download_quarterly_financials(ticker="005930.KS", year=2025, period="FY")

    mock_client.assert_not_called()
    assert result.loc[0, "account_id"] == "ifrs-full_Revenue"


def test_download_quarterly_financials_raises_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dart, "DATA_DIR_RAW", str(tmp_path))
    mock_dart = type("MockDart", (), {"finstate_all": lambda self, *a, **k: None})()

    with patch.object(dart, "_get_client", return_value=mock_dart):
        with pytest.raises(ValueError):
            dart.download_quarterly_financials(ticker="005930.KS", year=2025, period="FY", use_cache=False)

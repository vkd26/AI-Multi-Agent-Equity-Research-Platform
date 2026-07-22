"""market.py의 캐싱/에러 처리 로직을 검증한다 (네트워크 미사용, yf.Ticker는 모킹)."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector import market


def test_download_price_history_uses_cache_without_calling_yfinance(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "market_prices_NVDA_6mo_1d.csv"
    pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"])).to_csv(cache_path)

    with patch.object(market, "yf") as mock_yf:
        result = market.download_price_history("NVDA", period="6mo")

    mock_yf.Ticker.assert_not_called()
    assert list(result["Close"]) == [100.0, 101.0]


def test_download_price_history_raises_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "DATA_DIR_RAW", str(tmp_path))
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch.object(market.yf, "Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError):
            market.download_price_history("BADTICKER", use_cache=False)


def test_get_company_info_uses_cache_without_calling_yfinance(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "market_info_NVDA.json"
    cache_path.write_text('{"marketCap": 123, "currency": "USD"}', encoding="utf-8")

    with patch.object(market, "yf") as mock_yf:
        result = market.get_company_info("NVDA")

    mock_yf.Ticker.assert_not_called()
    assert result["marketCap"] == 123


def test_get_company_info_raises_when_ticker_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "DATA_DIR_RAW", str(tmp_path))
    mock_ticker = MagicMock()
    mock_ticker.info = {"regularMarketPrice": None, "currentPrice": None}

    with patch.object(market.yf, "Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError):
            market.get_company_info("BADTICKER", use_cache=False)


def test_get_fx_rate_returns_one_for_same_currency():
    assert market.get_fx_rate("USD", "USD") == 1.0


def test_get_fx_rate_fetches_from_yfinance_currency_pair_ticker():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": [0.03, 0.0308]})

    with patch.object(market.yf, "Ticker", return_value=mock_ticker) as mock_cls:
        rate = market.get_fx_rate("TWD", "USD")

    mock_cls.assert_called_once_with("TWDUSD=X")
    assert rate == pytest.approx(0.0308)


def test_get_fx_rate_raises_when_no_data():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch.object(market.yf, "Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError):
            market.get_fx_rate("TWD", "USD")


def test_normalize_financial_currency_returns_info_unchanged_when_currencies_match():
    info = {"currency": "USD", "financialCurrency": "USD", "freeCashflow": 100}
    assert market.normalize_financial_currency(info) is info


def test_normalize_financial_currency_returns_info_unchanged_when_currency_fields_missing():
    info = {"marketCap": 123}
    assert market.normalize_financial_currency(info) is info


def test_normalize_financial_currency_converts_money_fields_but_not_price_fields(monkeypatch):
    # TSM 같은 ADR: currency=USD(주가/시가총액), financialCurrency=TWD(재무제표) — 재무제표 절대금액
    # 필드만 환산하고, 이미 USD인 주가/시가총액/발행주식수는 건드리면 안 된다.
    monkeypatch.setattr(market, "get_fx_rate", lambda f, t: 0.03)
    info = {
        "currency": "USD", "financialCurrency": "TWD",
        "freeCashflow": 1000, "ebitda": 2000, "totalDebt": 500, "totalCash": 300, "totalRevenue": 4000,
        "currentPrice": 424.61, "marketCap": 2202228752384, "sharesOutstanding": 5186474013,
    }
    result = market.normalize_financial_currency(info)

    assert result["freeCashflow"] == pytest.approx(30)
    assert result["ebitda"] == pytest.approx(60)
    assert result["totalDebt"] == pytest.approx(15)
    assert result["totalCash"] == pytest.approx(9)
    assert result["totalRevenue"] == pytest.approx(120)
    assert result["currentPrice"] == 424.61
    assert result["marketCap"] == 2202228752384
    assert result["sharesOutstanding"] == 5186474013


def test_normalize_financial_currency_skips_missing_fields(monkeypatch):
    monkeypatch.setattr(market, "get_fx_rate", lambda f, t: 0.03)
    info = {"currency": "USD", "financialCurrency": "TWD", "freeCashflow": 1000}  # ebitda 등 없음
    result = market.normalize_financial_currency(info)
    assert result["freeCashflow"] == pytest.approx(30)
    assert "ebitda" not in result


def test_get_company_info_applies_currency_normalization_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "DATA_DIR_RAW", str(tmp_path))
    monkeypatch.setattr(market, "get_fx_rate", lambda f, t: 0.03)
    cache_path = tmp_path / "market_info_TSM.json"
    cache_path.write_text(
        '{"currency": "USD", "financialCurrency": "TWD", "freeCashflow": 1000, "marketCap": 123}',
        encoding="utf-8",
    )

    with patch.object(market, "yf") as mock_yf:
        result = market.get_company_info("TSM")

    mock_yf.Ticker.assert_not_called()
    assert result["freeCashflow"] == pytest.approx(30)
    assert result["marketCap"] == 123

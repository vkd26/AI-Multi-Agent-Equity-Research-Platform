"""comps.py의 피어 배수 집계/내재가치 계산, 피어 그룹 자동 선정을 검증한다."""
from unittest.mock import MagicMock

import pytest

from src.valuation import comps
from src.valuation.comps import (
    PeerSelectionError,
    _extract_ticker_candidates,
    _is_kr_or_us_ticker,
    comps_valuation,
    find_peer_tickers,
    implied_value_from_multiple,
    peer_multiples_table,
)


def test_peer_multiples_table_builds_table_and_median_summary():
    peers = {
        "AMD": {"enterpriseToEbitda": 100.0, "trailingPE": 150.0},
        "AVGO": {"enterpriseToEbitda": 40.0, "trailingPE": 60.0},
    }
    table, summary = peer_multiples_table(peers)
    assert table.loc["AMD", "enterpriseToEbitda"] == 100.0
    assert summary.loc["median", "enterpriseToEbitda"] == 70.0
    assert summary.loc["median", "trailingPE"] == 105.0


def test_implied_value_from_multiple_computes_product():
    value, warning = implied_value_from_multiple(target_metric=100, peer_multiple=20)
    assert value == 2000
    assert warning is None


def test_implied_value_from_multiple_returns_none_when_missing():
    assert implied_value_from_multiple(None, 20) == (None, None)
    assert implied_value_from_multiple(100, None) == (None, None)


def test_implied_value_from_multiple_returns_none_with_warning_when_target_metric_negative():
    value, warning = implied_value_from_multiple(-50, 20, metric_name="EBITDA")
    assert value is None
    assert "EBITDA" in warning and "음수" in warning


def test_implied_value_from_multiple_returns_none_with_warning_when_peer_multiple_negative():
    value, warning = implied_value_from_multiple(50, -20, metric_name="EPS")
    assert value is None
    assert "EPS" in warning and "음수" in warning


def test_comps_valuation_computes_implied_price_from_ev_ebitda_and_pe():
    target = {"ebitda": 200, "trailingEps": 5.0, "totalDebt": 50, "totalCash": 30, "sharesOutstanding": 100}
    peers = {"PEER1": {"enterpriseToEbitda": 10.0, "trailingPE": 20.0}}

    result = comps_valuation(target, peers)
    # implied_ev = 200 * 10 = 2000, net_debt = 50-30=20, equity = 1980, price = 19.8
    assert result["implied_price_from_ev_ebitda"] == pytest.approx(19.8)
    # implied_price_from_pe = 5.0 * 20 = 100 (PER 방식은 이미 주당 값이라 순부채 조정 불필요)
    assert result["implied_price_from_pe"] == pytest.approx(100.0)
    assert result["warnings"] == []


def test_comps_valuation_handles_missing_shares_or_metrics_gracefully():
    target = {"ebitda": None, "trailingEps": None}
    peers = {"PEER1": {"enterpriseToEbitda": 10.0, "trailingPE": 20.0}}
    result = comps_valuation(target, peers)
    assert result["implied_price_from_ev_ebitda"] is None
    assert result["implied_price_from_pe"] is None


def test_comps_valuation_skips_negative_ebitda_and_eps_with_warnings():
    # Wolfspeed(WOLF)처럼 EBITDA/EPS가 둘 다 적자인 기업 — 음수 "내재주가"를 조용히 내지 않아야 한다.
    target = {"ebitda": -100, "trailingEps": -2.0, "totalDebt": 50, "totalCash": 30, "sharesOutstanding": 100}
    peers = {"PEER1": {"enterpriseToEbitda": 10.0, "trailingPE": 20.0}}

    result = comps_valuation(target, peers)
    assert result["implied_price_from_ev_ebitda"] is None
    assert result["implied_price_from_pe"] is None
    assert len(result["warnings"]) == 2


def test_extract_ticker_candidates_parses_last_comma_separated_line():
    text = "이유 설명\n두번째 줄\nAMD, AVGO, 000660.KS"
    assert _extract_ticker_candidates(text) == ["AMD", "AVGO", "000660.KS"]


def test_extract_ticker_candidates_returns_empty_for_blank_text():
    assert _extract_ticker_candidates("") == []


def test_find_peer_tickers_filters_out_invalid_tickers_via_yfinance(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="AMD, FAKE_XYZ, AVGO")

    def fake_validate(ticker):
        return ticker != "FAKE_XYZ"

    monkeypatch.setattr(comps, "_validate_peer_ticker", fake_validate)
    result = find_peer_tickers("TSMC", "TSM", client=mock_client)
    assert result == ["AMD", "AVGO"]


def test_find_peer_tickers_excludes_target_ticker_itself(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="TSM, AMD, AVGO")
    monkeypatch.setattr(comps, "_validate_peer_ticker", lambda ticker: True)

    result = find_peer_tickers("TSMC", "TSM", client=mock_client)
    assert "TSM" not in result
    assert result == ["AMD", "AVGO"]


def test_find_peer_tickers_raises_when_fewer_than_min_peers_survive_validation(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="AMD, FAKE1, FAKE2")

    def fake_validate(ticker):
        return ticker == "AMD"

    monkeypatch.setattr(comps, "_validate_peer_ticker", fake_validate)
    with pytest.raises(PeerSelectionError):
        find_peer_tickers("TSMC", "TSM", client=mock_client, min_peers=2)


def test_find_peer_tickers_stops_at_max_peers(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="A, B, C, D, E, F")
    monkeypatch.setattr(comps, "_validate_peer_ticker", lambda ticker: True)

    result = find_peer_tickers("Test Co", "TEST", client=mock_client, max_peers=3)
    assert len(result) == 3


def test_is_kr_or_us_ticker_accepts_us_tickers_without_suffix():
    assert _is_kr_or_us_ticker("AMD") is True
    assert _is_kr_or_us_ticker("NVDA") is True


def test_is_kr_or_us_ticker_accepts_kr_suffixes():
    assert _is_kr_or_us_ticker("005930.KS") is True
    assert _is_kr_or_us_ticker("000660.KQ") is True


def test_is_kr_or_us_ticker_rejects_other_exchange_suffixes():
    assert _is_kr_or_us_ticker("0981.HK") is False
    assert _is_kr_or_us_ticker("2330.TW") is False


def test_find_peer_tickers_excludes_non_kr_us_exchanges(monkeypatch):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="AMD, 0981.HK, UMC, 2330.TW, 005930.KS")
    monkeypatch.setattr(comps, "_validate_peer_ticker", lambda ticker: True)

    result = find_peer_tickers("TSMC", "TSM", client=mock_client)
    assert result == ["AMD", "UMC", "005930.KS"]

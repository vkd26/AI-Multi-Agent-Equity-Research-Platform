"""dcf.py의 WACC/FCF투영/터미널가치/할인/전체파이프라인 계산을 검증한다."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.valuation.dcf import (
    _GROWTH_RATE_FALLBACK,
    _KR_RISK_FREE_RATE_FALLBACK,
    compute_wacc,
    discount_cash_flows,
    get_equity_risk_premium,
    get_growth_rate_estimate,
    get_risk_free_rate,
    project_fcf,
    run_dcf,
    terminal_value,
)


def test_get_risk_free_rate_kr_returns_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("KRX_API_KEY", raising=False)
    with patch("dotenv.load_dotenv"):  # 실제 ~/.env에 키가 있어도 테스트는 결정론적으로 동작해야 한다
        assert get_risk_free_rate("KR") == _KR_RISK_FREE_RATE_FALLBACK


def test_get_risk_free_rate_kr_picks_nominal_govbond_not_inflation_linked(monkeypatch):
    # KRX API 응답엔 만기 10년 종목이 두 개 섞여 나온다 — 하나는 물가연동국고채(종목명이 "물가"로
    # 시작, 실질금리라 훨씬 낮음), 하나는 명목 국고채("국고"로 시작) — 후자를 골라야 한다.
    monkeypatch.setenv("KRX_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "OutBlock_1": [
            {"BND_EXP_TP_NM": "10", "GOVBND_ISU_TP_NM": "지표", "ISU_NM": "물가01125-3606(26-4)", "CLSPRC_YD": "1.731"},
            {"BND_EXP_TP_NM": "10", "GOVBND_ISU_TP_NM": "지표", "ISU_NM": "국고04250-3606(26-6)", "CLSPRC_YD": "4.397"},
        ]
    }
    with patch("dotenv.load_dotenv"), patch("requests.get", return_value=mock_response) as mock_get:
        rate = get_risk_free_rate("KR")

    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["params"]["AUTH_KEY"] == "fake-key"
    assert rate == pytest.approx(0.04397)


def test_get_risk_free_rate_kr_retries_earlier_dates_when_no_data(monkeypatch):
    monkeypatch.setenv("KRX_API_KEY", "fake-key")
    empty_response = MagicMock()
    empty_response.json.return_value = {"OutBlock_1": []}
    good_response = MagicMock()
    good_response.json.return_value = {
        "OutBlock_1": [{"BND_EXP_TP_NM": "10", "GOVBND_ISU_TP_NM": "지표", "ISU_NM": "국고01125", "CLSPRC_YD": "4.2"}]
    }

    with patch("dotenv.load_dotenv"), patch("requests.get", side_effect=[empty_response, empty_response, good_response]) as mock_get:
        rate = get_risk_free_rate("KR")

    assert mock_get.call_count == 3
    assert rate == pytest.approx(0.042)


def test_get_risk_free_rate_kr_falls_back_after_a_week_of_no_data(monkeypatch):
    monkeypatch.setenv("KRX_API_KEY", "fake-key")
    empty_response = MagicMock()
    empty_response.json.return_value = {"OutBlock_1": []}

    with patch("dotenv.load_dotenv"), patch("requests.get", return_value=empty_response):
        rate = get_risk_free_rate("KR")

    assert rate == _KR_RISK_FREE_RATE_FALLBACK


def test_get_equity_risk_premium_differs_by_market():
    # 한국은 Damodaran의 country risk premium이 더해져서 미국보다 높아야 한다
    us_erp = get_equity_risk_premium("US")
    kr_erp = get_equity_risk_premium("KR")
    assert kr_erp > us_erp
    assert us_erp == pytest.approx(0.0445)
    assert kr_erp == pytest.approx(0.0487)


def test_get_equity_risk_premium_defaults_to_us_for_unknown_market():
    assert get_equity_risk_premium("XX") == get_equity_risk_premium("US")


def test_get_risk_free_rate_us_fetches_from_tnx_and_converts_to_decimal():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": [4.5, 4.6, 4.642]})

    with patch("yfinance.Ticker", return_value=mock_ticker) as mock_cls:
        rate = get_risk_free_rate("US")

    mock_cls.assert_called_once_with("^TNX")
    assert rate == pytest.approx(0.04642)


def test_get_risk_free_rate_us_raises_when_no_data():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError):
            get_risk_free_rate("US")


def test_get_growth_rate_estimate_uses_revenue_growth_when_present():
    assert get_growth_rate_estimate({"revenueGrowth": 0.42}) == pytest.approx(0.42)


def test_get_growth_rate_estimate_falls_back_when_field_missing():
    assert get_growth_rate_estimate({}) == _GROWTH_RATE_FALLBACK


def test_compute_wacc_weights_equity_and_debt_by_market_value():
    info = {"beta": 1.5, "marketCap": 800, "totalDebt": 200}
    wacc = compute_wacc(info, risk_free_rate=0.03, equity_risk_premium=0.05, tax_rate=0.2)
    cost_of_equity = 0.03 + 1.5 * 0.05  # 0.105
    cost_of_debt = 0.03 + 0.015  # 0.045 (기본값)
    expected = 0.8 * cost_of_equity + 0.2 * cost_of_debt * (1 - 0.2)
    assert wacc == pytest.approx(expected)


def test_compute_wacc_uses_explicit_cost_of_debt_when_given():
    info = {"beta": 1.0, "marketCap": 500, "totalDebt": 500}
    wacc = compute_wacc(info, risk_free_rate=0.03, equity_risk_premium=0.05, tax_rate=0.25, cost_of_debt=0.06)
    expected = 0.5 * 0.08 + 0.5 * 0.06 * 0.75
    assert wacc == pytest.approx(expected)


def test_compute_wacc_returns_none_when_beta_missing():
    assert compute_wacc({"marketCap": 100}, 0.03, 0.05, 0.2) is None


def test_compute_wacc_treats_missing_debt_as_zero():
    info = {"beta": 1.0, "marketCap": 100}  # totalDebt 없음 -> 무부채로 취급
    wacc = compute_wacc(info, risk_free_rate=0.03, equity_risk_premium=0.05, tax_rate=0.2)
    assert wacc == pytest.approx(0.03 + 1.0 * 0.05)


def test_project_fcf_compounds_growth_rate():
    result = project_fcf(base_fcf=100, growth_rate=0.10, years=3)
    assert result == pytest.approx([110, 121, 133.1])


def test_terminal_value_uses_gordon_growth_formula():
    tv = terminal_value(final_year_fcf=100, wacc=0.10, terminal_growth_rate=0.03)
    assert tv == pytest.approx(100 * 1.03 / 0.07)


def test_terminal_value_raises_when_wacc_not_greater_than_terminal_growth():
    with pytest.raises(ValueError):
        terminal_value(final_year_fcf=100, wacc=0.02, terminal_growth_rate=0.03)


def test_discount_cash_flows_discounts_each_year_correctly():
    result = discount_cash_flows([110, 121], wacc=0.10)
    assert result == pytest.approx(110 / 1.1 + 121 / 1.1**2)


def test_run_dcf_end_to_end_matches_manual_calculation():
    result = run_dcf(
        base_fcf=100, wacc=0.10, growth_rate=0.05, terminal_growth_rate=0.02,
        years=2, net_debt=50, shares_outstanding=10,
    )
    fcf1, fcf2 = 105, 110.25
    pv_fcf = fcf1 / 1.1 + fcf2 / 1.1**2
    tv = fcf2 * 1.02 / (0.10 - 0.02)
    pv_tv = tv / 1.1**2
    ev = pv_fcf + pv_tv
    equity_value = ev - 50

    assert result["pv_fcf"] == pytest.approx(pv_fcf)
    assert result["pv_terminal_value"] == pytest.approx(pv_tv)
    assert result["enterprise_value"] == pytest.approx(ev)
    assert result["equity_value"] == pytest.approx(equity_value)
    assert result["implied_share_price"] == pytest.approx(equity_value / 10)


def test_run_dcf_omits_implied_share_price_when_shares_not_given():
    result = run_dcf(base_fcf=100, wacc=0.10, growth_rate=0.05, terminal_growth_rate=0.02)
    assert "implied_share_price" not in result

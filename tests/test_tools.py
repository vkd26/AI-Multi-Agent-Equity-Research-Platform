"""tools.py의 도구 함수들이 올바른 하위 모듈을 올바른 인자로 호출하고, 결과를 LLM이 다룰 수 있는
단순 타입(dict/list)으로 변환하는지 검증한다 (실제 API 호출은 모킹).

실제 에이전트 루프(질문 -> 도구 선택 -> 실행 -> 답변)가 NVDA로 정확히 동작하는지는 Gemini 실 API로
수동 검증했다(README 참고).
"""
import pandas as pd
import pytest

from src.agents import tools


def test_get_stock_price_and_info_extracts_key_fields(monkeypatch):
    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {
        "currentPrice": 207.29, "marketCap": 5e12, "currency": "USD",
        "fiftyTwoWeekLow": 164.07, "fiftyTwoWeekHigh": 236.54,
    })
    result = tools.get_stock_price_and_info("NVDA")
    assert result == {
        "ticker": "NVDA", "current_price": 207.29, "market_cap": 5e12, "currency": "USD",
        "fifty_two_week_low": 164.07, "fifty_two_week_high": 236.54,
    }


def test_get_financial_ratios_delegates_to_financial_analyzer(monkeypatch):
    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {"trailingPE": 31.17})
    result = tools.get_financial_ratios("NVDA")
    assert result["per"] == 31.17


def test_get_recent_material_disclosures_uses_kind_b_for_kr(monkeypatch):
    monkeypatch.setattr(tools.dart, "download_disclosures", lambda **kw: pd.DataFrame({
        "report_nm": ["자기주식처분결정"], "rcept_dt": ["20260713"],
    }))
    result = tools.get_recent_material_disclosures("005930.KS")
    assert result == [{"title": "자기주식처분결정", "date": "20260713"}]


def test_get_recent_material_disclosures_uses_8k_for_us(monkeypatch):
    captured = {}

    def fake_download_filings(ticker, forms):
        captured["forms"] = forms
        return pd.DataFrame({"form": ["8-K"], "filingDate": ["2026-07-21"]})

    monkeypatch.setattr(tools.sec, "download_filings", fake_download_filings)
    result = tools.get_recent_material_disclosures("NVDA")
    assert captured["forms"] == ("8-K",)
    assert result == [{"title": "8-K", "date": "2026-07-21"}]


def test_search_earnings_call_returns_error_for_kr_tickers():
    result = tools.search_earnings_call("005930.KS", "가이던스가 뭐야?")
    assert result["error"]
    assert result["summary"] is None


def test_run_dcf_valuation_returns_error_when_wacc_unavailable(monkeypatch):
    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {})  # beta 없음
    monkeypatch.setattr(tools, "get_risk_free_rate", lambda market: 0.046)
    result = tools.run_dcf_valuation("NVDA", growth_rate=0.15, terminal_growth_rate=0.03)
    assert "error" in result


def test_run_dcf_valuation_computes_implied_price(monkeypatch):
    info = {
        "beta": 1.5, "marketCap": 1000, "totalDebt": 100, "totalCash": 50,
        "freeCashflow": 200, "sharesOutstanding": 10,
    }
    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: info)
    monkeypatch.setattr(tools, "get_risk_free_rate", lambda market: 0.046)
    result = tools.run_dcf_valuation("NVDA", growth_rate=0.10, terminal_growth_rate=0.02)
    assert "implied_share_price" in result
    assert "wacc" in result


def test_run_dcf_valuation_uses_korean_tax_rate_and_fetches_kr_risk_free_rate(monkeypatch):
    # 한국 종목에 미국 세율(21%)을 쓰면 안 된다 — 2026년 대기업 실효세율(27.5%)을 써야 한다
    captured = {}

    def fake_compute_wacc(info, risk_free_rate, equity_risk_premium, tax_rate):
        captured["risk_free_rate"] = risk_free_rate
        captured["equity_risk_premium"] = equity_risk_premium
        captured["tax_rate"] = tax_rate
        return 0.09

    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {
        "beta": 1.0, "marketCap": 1000, "totalDebt": 0, "totalCash": 0,
        "freeCashflow": 100, "sharesOutstanding": 10,
    })
    monkeypatch.setattr(tools, "compute_wacc", fake_compute_wacc)
    monkeypatch.setattr(tools, "get_risk_free_rate", lambda market: {"KR": 0.0439, "US": 0.0463}[market])
    tools.run_dcf_valuation("005930.KS", growth_rate=0.08, terminal_growth_rate=0.02)

    assert captured["tax_rate"] == 0.275
    assert captured["risk_free_rate"] == 0.0439
    assert captured["equity_risk_premium"] == 0.0487  # 한국 ERP(Damodaran) — 미국(0.0445)보다 국가리스크만큼 높아야 함


def test_run_dcf_valuation_uses_us_tax_rate_and_fetches_us_risk_free_rate(monkeypatch):
    captured = {}

    def fake_compute_wacc(info, risk_free_rate, equity_risk_premium, tax_rate):
        captured["risk_free_rate"] = risk_free_rate
        captured["equity_risk_premium"] = equity_risk_premium
        captured["tax_rate"] = tax_rate
        return 0.09

    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {
        "beta": 1.0, "marketCap": 1000, "totalDebt": 0, "totalCash": 0,
        "freeCashflow": 100, "sharesOutstanding": 10,
    })
    monkeypatch.setattr(tools, "compute_wacc", fake_compute_wacc)
    monkeypatch.setattr(tools, "get_risk_free_rate", lambda market: {"KR": 0.0439, "US": 0.0463}[market])
    tools.run_dcf_valuation("NVDA", growth_rate=0.08, terminal_growth_rate=0.02)

    assert captured["tax_rate"] == 0.21
    assert captured["risk_free_rate"] == 0.0463
    assert captured["equity_risk_premium"] == 0.0445  # 미국 ERP(Damodaran implied ERP)


def test_run_dcf_valuation_auto_derives_growth_rate_from_revenue_growth_when_not_given(monkeypatch):
    captured = {}

    def fake_run_dcf(base_fcf, wacc, growth_rate, terminal_growth_rate, years, net_debt, shares_outstanding):
        captured["growth_rate"] = growth_rate
        return {"equity_value": 1, "enterprise_value": 1, "pv_fcf": 1, "pv_terminal_value": 1}

    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {
        "beta": 1.0, "marketCap": 1000, "totalDebt": 0, "totalCash": 0,
        "freeCashflow": 100, "sharesOutstanding": 10, "revenueGrowth": 0.42,
    })
    monkeypatch.setattr(tools, "get_risk_free_rate", lambda market: 0.046)
    monkeypatch.setattr(tools, "run_dcf", fake_run_dcf)
    tools.run_dcf_valuation("NVDA", terminal_growth_rate=0.02)

    assert captured["growth_rate"] == pytest.approx(0.42)


def test_run_comps_valuation_drops_peer_table_for_llm_compatibility(monkeypatch):
    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {
        "ebitda": 100, "trailingEps": 2.0, "totalDebt": 10, "totalCash": 5, "sharesOutstanding": 10,
    })
    result = tools.run_comps_valuation("NVDA", peer_tickers=["AMD"])
    assert "peer_table" not in result
    assert "implied_price_from_ev_ebitda" in result


def test_run_comps_valuation_auto_selects_peers_when_not_given(monkeypatch):
    captured = {}

    def fake_get_company_info(ticker):
        captured.setdefault("tickers", []).append(ticker)
        return {"longName": "NVIDIA Corp", "ebitda": 100, "trailingEps": 2.0, "totalDebt": 10, "totalCash": 5, "sharesOutstanding": 10}

    def fake_find_peer_tickers(company_name, ticker):
        captured["company_name"] = company_name
        captured["ticker"] = ticker
        return ["AMD", "AVGO"]

    monkeypatch.setattr(tools.market, "get_company_info", fake_get_company_info)
    monkeypatch.setattr(tools, "find_peer_tickers", fake_find_peer_tickers)
    result = tools.run_comps_valuation("NVDA")

    assert captured["company_name"] == "NVIDIA Corp"
    assert captured["ticker"] == "NVDA"
    assert "implied_price_from_ev_ebitda" in result


def test_run_comps_valuation_returns_error_when_peer_selection_fails(monkeypatch):
    monkeypatch.setattr(tools.market, "get_company_info", lambda ticker: {"longName": "NVIDIA Corp"})

    def fake_find_peer_tickers(company_name, ticker):
        raise tools.PeerSelectionError("검증된 피어가 0개뿐이다")

    monkeypatch.setattr(tools, "find_peer_tickers", fake_find_peer_tickers)
    result = tools.run_comps_valuation("NVDA")
    assert "error" in result

"""portfolio_monitor.py의 트리거 판정/상태 영속성을 검증한다 (dart/sec/transcript 네트워크 호출은 모킹).

실제 트리거 4개(공시/뉴스/실적발표/목표주가)가 실 데이터로 정확히 발동/유지되는지는 삼성전자로
수동 검증했다(README 참고).
"""
import pandas as pd
import pytest

from src.agents import portfolio_monitor as pm


def test_state_round_trips_through_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_STATE_DIR", str(tmp_path))
    pm.save_state("NVDA", {"last_target_price": 100})
    assert pm.load_state("NVDA") == {"last_target_price": 100}


def test_load_state_returns_empty_dict_when_no_prior_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_STATE_DIR", str(tmp_path))
    assert pm.load_state("NVDA") == {}


def test_check_material_disclosures_uses_kind_b_for_kr(monkeypatch):
    captured = {}

    def fake_download_disclosures(stock_code, kind, use_cache):
        captured["stock_code"] = stock_code
        captured["kind"] = kind
        return pd.DataFrame({"rcept_no": ["20260101000001", "20260201000002"]})

    monkeypatch.setattr(pm.dart, "download_disclosures", fake_download_disclosures)
    new_items, latest_id = pm.check_material_disclosures("005930.KS", "KR", last_seen_id=None)

    assert captured["kind"] == "B"
    assert captured["stock_code"] == "005930"
    assert latest_id == "20260201000002"
    assert len(new_items) == 2


def test_check_material_disclosures_filters_to_only_new_items(monkeypatch):
    monkeypatch.setattr(pm.dart, "download_disclosures", lambda **kw: pd.DataFrame({
        "rcept_no": ["20260101000001", "20260201000002", "20260301000003"],
    }))
    new_items, latest_id = pm.check_material_disclosures("005930.KS", "KR", last_seen_id="20260201000002")
    assert list(new_items["rcept_no"]) == ["20260301000003"]
    assert latest_id == "20260301000003"


def test_check_material_disclosures_uses_8k_for_us(monkeypatch):
    captured = {}

    def fake_download_filings(ticker, forms, use_cache):
        captured["forms"] = forms
        return pd.DataFrame({"accessionNumber": ["0001-26-000001"]})

    monkeypatch.setattr(pm.sec, "download_filings", fake_download_filings)
    pm.check_material_disclosures("NVDA", "US", last_seen_id=None)
    assert captured["forms"] == ("8-K",)


def test_check_high_impact_news_filters_by_impact_and_unseen_id():
    news_df = pd.DataFrame({
        "link": ["a", "b", "c"],
        "impact": ["High", "High", "Low"],
    })
    new_items, seen_ids = pm.check_high_impact_news(news_df, last_seen_ids={"a"})
    assert list(new_items["link"]) == ["b"]
    assert seen_ids == {"a", "b", "c"}


def test_check_new_earnings_call_detects_quarter_change(monkeypatch):
    monkeypatch.setattr(pm.transcript, "find_transcript_url", lambda ticker: "https://x/q2-2026-earnings-call-transcript/")
    monkeypatch.setattr(pm.transcript, "_parse_fiscal_quarter", lambda url: "Q2 2026")

    is_new, quarter = pm.check_new_earnings_call("NVDA", last_seen_quarter="Q1 2026")
    assert is_new is True
    assert quarter == "Q2 2026"


def test_check_new_earnings_call_no_change_when_same_quarter(monkeypatch):
    monkeypatch.setattr(pm.transcript, "find_transcript_url", lambda ticker: "https://x/q2-2026-earnings-call-transcript/")
    monkeypatch.setattr(pm.transcript, "_parse_fiscal_quarter", lambda url: "Q2 2026")

    is_new, quarter = pm.check_new_earnings_call("NVDA", last_seen_quarter="Q2 2026")
    assert is_new is False


def test_check_new_earnings_call_handles_missing_transcript_gracefully(monkeypatch):
    def raise_not_found(ticker):
        raise ValueError("not found")
    monkeypatch.setattr(pm.transcript, "find_transcript_url", raise_not_found)

    is_new, quarter = pm.check_new_earnings_call("005930.KS", last_seen_quarter="Q1 2026")
    assert is_new is False
    assert quarter == "Q1 2026"


def test_check_target_price_drift_triggers_on_first_check():
    triggered, pct_change = pm.check_target_price_drift(100, last_target_price=None)
    assert triggered is True
    assert pct_change is None


def test_check_target_price_drift_triggers_above_threshold():
    triggered, pct_change = pm.check_target_price_drift(115, last_target_price=100, threshold=0.10)
    assert triggered is True
    assert pct_change == pytest.approx(0.15)


def test_check_target_price_drift_does_not_trigger_below_threshold():
    triggered, pct_change = pm.check_target_price_drift(105, last_target_price=100, threshold=0.10)
    assert triggered is False
    assert pct_change == pytest.approx(0.05)


def test_run_daily_check_combines_all_triggers_and_persists_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(pm.dart, "download_disclosures", lambda **kw: pd.DataFrame({"rcept_no": ["20260101000001"]}))
    monkeypatch.setattr(pm.transcript, "find_transcript_url", lambda ticker: "https://x/q2-2026-earnings-call-transcript/")
    monkeypatch.setattr(pm.transcript, "_parse_fiscal_quarter", lambda url: "Q2 2026")

    result = pm.run_daily_check("005930.KS", cheap_target_price=100)

    assert result["triggers"]["material_disclosure"] is True
    assert result["triggers"]["target_price_drift"] is True  # 최초 실행
    assert result["need_full_regeneration"] is True

    state = pm.load_state("005930.KS")
    assert state["last_target_price"] == 100
    assert state["last_disclosure_id"] == "20260101000001"

    # 두 번째 실행: 상태가 그대로면 트리거가 전부 꺼져야 한다
    result2 = pm.run_daily_check("005930.KS", cheap_target_price=100)
    assert not any(result2["triggers"].values())
    assert result2["need_full_regeneration"] is False


def test_run_daily_check_without_news_df_leaves_news_trigger_false(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(pm.dart, "download_disclosures", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(pm.transcript, "find_transcript_url", lambda ticker: (_ for _ in ()).throw(ValueError()))

    result = pm.run_daily_check("005930.KS", cheap_target_price=100, classified_news_df=None)
    assert result["triggers"]["high_impact_news"] is False
    assert result["new_high_impact_news"] is None


def test_run_daily_check_uses_url_column_for_us_finnhub_news_schema(tmp_path, monkeypatch):
    # Finnhub 뉴스는 "link"가 아니라 "url" 컬럼을 쓴다 — id_col을 하드코딩하면 KeyError가 난다.
    monkeypatch.setattr(pm, "_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(pm.sec, "download_filings", lambda ticker, **kw: pd.DataFrame())
    monkeypatch.setattr(pm.transcript, "find_transcript_url", lambda ticker: (_ for _ in ()).throw(ValueError()))

    news_df = pd.DataFrame({"url": ["https://a.com/1"], "impact": ["High"]})
    result = pm.run_daily_check("NVDA", cheap_target_price=100, classified_news_df=news_df)

    assert result["triggers"]["high_impact_news"] is True
    assert len(result["new_high_impact_news"]) == 1

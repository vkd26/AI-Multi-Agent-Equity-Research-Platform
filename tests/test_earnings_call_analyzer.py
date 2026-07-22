"""earnings_call_analyzer.py의 키워드 카운팅/QoQ 비교/화자 필터링/주제 요약을 검증한다.

summarize_topic()은 retrieval.search와 LLM 클라이언트를 모킹한다 — 실제 검색 품질/요약 품질은
TSMC 실 데이터로 수동 검증했다(README 참고).
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.analytics import earnings_call_analyzer as eca


def test_count_keyword_mentions_uses_word_boundaries():
    # "again"/"maintain"/"certain"에 들어있는 "ai" 부분 문자열까지 "AI"로 잘못 세면 안 된다
    df = pd.DataFrame({"text": [
        "We will maintain this strategy again, and remain certain about AI demand.",
    ]})
    counts = eca.count_keyword_mentions(df, keywords=["AI"])
    assert counts["AI"] == 1


def test_count_keyword_mentions_is_case_insensitive_and_sums_across_rows():
    df = pd.DataFrame({"text": ["CapEx is rising.", "capex guidance was raised.", "No mention here."]})
    counts = eca.count_keyword_mentions(df, keywords=["CapEx"])
    assert counts["CapEx"] == 2


def test_compare_mentions_qoq_computes_pct_change():
    prior = {"AI": 12, "Risk": 4}
    current = {"AI": 31, "Risk": 4}
    result = eca.compare_mentions_qoq(prior, current).set_index("keyword")
    assert result.loc["AI", "pct_change"] == pytest.approx(31 / 12 - 1)
    assert result.loc["Risk", "pct_change"] == 0.0


def test_compare_mentions_qoq_handles_new_keyword_with_no_prior_mentions():
    result = eca.compare_mentions_qoq({}, {"AI": 5}).set_index("keyword")
    assert result.loc["AI", "prior_count"] == 0
    assert result.loc["AI", "pct_change"] == float("inf")


def test_compare_mentions_qoq_zero_to_zero_is_no_change():
    result = eca.compare_mentions_qoq({"Risk": 0}, {"Risk": 0}).set_index("keyword")
    assert result.loc["Risk", "pct_change"] == 0.0


def test_extract_speaker_statements_filters_by_role_and_preserves_order():
    df = pd.DataFrame({
        "speaker_role": ["CEO", "Analyst", "CFO", "CEO"],
        "text": ["ceo-1", "analyst-1", "cfo-1", "ceo-2"],
    })
    assert eca.extract_speaker_statements(df, "CEO") == ["ceo-1", "ceo-2"]
    assert eca.extract_speaker_statements(df, "CFO") == ["cfo-1"]


def test_summarize_topic_returns_empty_when_no_results_found():
    store = object()
    with patch.object(eca.retrieval, "search", return_value=pd.DataFrame()):
        result = eca.summarize_topic(store, "guidance")
    assert result == {"summary": None, "citations": []}


def test_capex_topic_uses_dedicated_query_not_raw_topic_string():
    store = object()
    with patch.object(eca.retrieval, "search", return_value=pd.DataFrame()) as mock_search:
        eca.summarize_topic(store, "capex")
    assert mock_search.call_args.args[1] == eca._TOPIC_QUERIES["capex"]


def test_summarize_topic_builds_prompt_from_results_and_calls_llm():
    store = object()
    search_results = pd.DataFrame({
        "text": ["Revenue guidance is $40B."],
        "citation": ["TSM | Q2 2026 | CFO | chunk 1-2"],
    })
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="  Summary text.  ")

    with patch.object(eca.retrieval, "search", return_value=search_results) as mock_search:
        result = eca.summarize_topic(store, "guidance", ticker="TSM", client=mock_client)

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs["group_cols"] == ("ticker",)
    call_prompt = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "Revenue guidance is $40B." in call_prompt
    assert result == {"summary": "Summary text.", "citations": ["TSM | Q2 2026 | CFO | chunk 1-2"]}

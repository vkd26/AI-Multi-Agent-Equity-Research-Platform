"""news_analyzer.py의 분류 결과 매핑/검증/집계 로직을 검증한다 (LLM 클라이언트는 모킹).

실제 분류 품질(논조/영향도가 합리적인지)은 네이버·Finnhub 실 데이터로 수동 검증했다(README 참고).
"""
import json
from unittest.mock import MagicMock

import pandas as pd

from src.analytics.news_analyzer import classify_news, summarize_classification


def _mock_client(response_items):
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps(response_items))
    return client


def test_classify_news_maps_index_to_tone_and_impact():
    news_df = pd.DataFrame({"title": ["기사1", "기사2"], "description": ["설명1", "설명2"]})
    client = _mock_client([
        {"index": 0, "tone": "Positive", "impact": "High"},
        {"index": 1, "tone": "Negative", "impact": "Low"},
    ])
    result = classify_news(news_df, "삼성전자", client=client)
    assert result.loc[0, "tone"] == "Positive"
    assert result.loc[0, "impact"] == "High"
    assert result.loc[1, "tone"] == "Negative"
    assert result.loc[1, "impact"] == "Low"


def test_classify_news_ignores_invalid_tone_and_impact_values():
    news_df = pd.DataFrame({"title": ["기사1"], "description": ["설명1"]})
    client = _mock_client([{"index": 0, "tone": "매우긍정", "impact": "Critical"}])
    result = classify_news(news_df, "삼성전자", client=client)
    # 스키마에 없는 값은 버려지고 None으로 남아야 한다 (잘못된 값을 그대로 쓰면 안 됨)
    assert result.loc[0, "tone"] is None
    assert result.loc[0, "impact"] is None


def test_classify_news_ignores_out_of_range_index():
    news_df = pd.DataFrame({"title": ["기사1"], "description": ["설명1"]})
    client = _mock_client([{"index": 5, "tone": "Positive", "impact": "High"}])
    result = classify_news(news_df, "삼성전자", client=client)
    assert result.loc[0, "tone"] is None


def test_classify_news_empty_input_returns_empty_without_calling_llm():
    news_df = pd.DataFrame({"title": [], "description": []})
    client = MagicMock()
    result = classify_news(news_df, "삼성전자", client=client)
    assert result.empty
    client.models.generate_content.assert_not_called()


def test_classify_news_uses_custom_column_names_for_finnhub_schema():
    news_df = pd.DataFrame({"headline": ["NVDA headline"], "summary": ["NVDA summary"]})
    client = _mock_client([{"index": 0, "tone": "Neutral", "impact": "Low"}])
    result = classify_news(news_df, "NVIDIA", title_col="headline", desc_col="summary", client=client)
    assert result.loc[0, "tone"] == "Neutral"
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "NVDA headline" in prompt and "NVDA summary" in prompt


def test_summarize_classification_counts_and_flags_high_impact_negative():
    classified = pd.DataFrame({
        "title": ["a", "b", "c", "d"],
        "tone": ["Positive", "Negative", "Negative", "Neutral"],
        "impact": ["High", "High", "Low", "Medium"],
    })
    summary = summarize_classification(classified, title_col="title")
    assert summary["tone_counts"] == {"Positive": 1, "Negative": 2, "Neutral": 1}
    assert summary["impact_counts"] == {"High": 2, "Low": 1, "Medium": 1}
    assert summary["high_impact_negative_titles"] == ["b"]

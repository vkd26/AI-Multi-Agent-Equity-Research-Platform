"""⑤ News Analyzer — 뉴스 기사를 논조(Positive/Negative/Neutral)와 영향도(High/Medium/Low)로 분류한다.

한국(네이버)은 title/description, 미국(Finnhub)은 headline/summary로 컬럼명이 달라서 collector 스키마를
그대로 넘기지 말고 text_col/desc_col을 호출 측에서 지정한다. 기사를 한 번에 하나씩 LLM에 보내면 뉴스가
많을 때 호출 비용이 커지므로, 전체 목록을 한 번에 보내고 JSON 배열로 일괄 분류받는다.
"""
import json

import pandas as pd

from src.analytics._common import get_llm_client

VALID_TONES = ("Positive", "Negative", "Neutral")
VALID_IMPACTS = ("High", "Medium", "Low")


def classify_news(news_df, company_name, title_col="title", desc_col="description", model="gemini-flash-latest", client=None):
    """뉴스 기사 목록에 tone/impact 컬럼을 채워 반환한다.

    LLM이 일부 기사를 빠뜨리거나 유효하지 않은 값을 반환해도(모델 응답이 완벽하지 않을 수 있음)
    죽지 않고, 해당 행은 tone/impact가 None으로 남는다.
    """
    result = news_df.reset_index(drop=True).copy()
    result["tone"] = None
    result["impact"] = None
    if result.empty:
        return result

    articles = "\n".join(
        f"{i}. {row.get(title_col) or ''} - {row.get(desc_col) or ''}"
        for i, row in result.iterrows()
    )
    prompt = (
        f"다음은 '{company_name}' 관련 뉴스 기사 목록이다. 각 기사를 투자자 관점에서 분류하라.\n"
        f"- tone: {'/'.join(VALID_TONES)} 중 하나\n"
        f"- impact: 주가나 사업에 미칠 영향의 크기, {'/'.join(VALID_IMPACTS)} 중 하나\n\n"
        f"{articles}\n\n"
        '각 기사마다 {"index": 번호, "tone": "...", "impact": "..."} 형태의 JSON 배열로만 답하라. '
        "번호는 위 목록의 번호와 정확히 일치해야 한다."
    )

    client = client or get_llm_client()
    response = client.models.generate_content(
        model=model, contents=prompt, config={"response_mime_type": "application/json"},
    )
    classifications = json.loads(response.text)

    for item in classifications:
        idx = item.get("index")
        tone = item.get("tone")
        impact = item.get("impact")
        if idx is None or not (0 <= idx < len(result)):
            continue
        if tone in VALID_TONES:
            result.loc[idx, "tone"] = tone
        if impact in VALID_IMPACTS:
            result.loc[idx, "impact"] = impact

    return result


def summarize_classification(classified_df, title_col="title"):
    """논조/영향도 분포와, 우선적으로 챙겨봐야 할 고영향 부정 기사 목록을 뽑는다."""
    high_impact_negative = classified_df[
        (classified_df["tone"] == "Negative") & (classified_df["impact"] == "High")
    ]
    return {
        "tone_counts": classified_df["tone"].value_counts(dropna=True).to_dict(),
        "impact_counts": classified_df["impact"].value_counts(dropna=True).to_dict(),
        "high_impact_negative_titles": high_impact_negative[title_col].tolist(),
    }

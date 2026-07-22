"""④ Earnings Call Analyzer — 실적발표 대본에서 CEO/CFO 발언, Guidance, Risk, Margin, Demand, AI 언급,
Capex, Inventory를 뽑고 전분기 대비 변화를 계산한다.

키워드 언급 횟수(count_keyword_mentions/compare_mentions_qoq)와 화자 역할별 발언 모음
(extract_speaker_statements)은 순수 계산이라 LLM이 필요 없다. 반면 "Guidance/Risk에 대해 경영진이
뭐라고 했는지" 같은 주제별 요약(summarize_topic)은 rag/retrieval.py로 관련 발언을 하이브리드 검색+
리랭크로 찾은 뒤, 그 발언만 근거로 LLM에게 요약을 요청한다 — 전체 대본을 통째로 넣지 않아 토큰 비용도
줄고, 요약이 실제 발언에서 벗어나지 않도록 근거를 좁혀준다.
"""
import re

import pandas as pd

from src.analytics._common import get_llm_client
from src.rag import retrieval

DEFAULT_KEYWORDS = ["AI", "CapEx", "Inventory", "Margin", "Demand", "Guidance", "Risk"]

_TOPIC_QUERIES = {
    "guidance": "What guidance did management provide for the next quarter or fiscal year?",
    "risk": "What risks, headwinds, or uncertainties did management mention?",
    "margin": "What did management say about gross margin or operating margin trends?",
    "demand": "What did management say about customer demand?",
    "capex": "What did management say about capital expenditure (CapEx) plans?",
}


def count_keyword_mentions(transcript_df, keywords=DEFAULT_KEYWORDS, text_col="text"):
    """transcript_df 전체 텍스트에서 키워드별 등장 횟수를 센다 (대소문자 무시, 단어 경계 매칭).

    단어 경계(\\b) 없이 부분 문자열로만 매칭하면 "AI"가 "again", "maintain", "certain" 같은 단어 속
    "ai"에도 걸려 카운트가 크게 부풀려진다 — 실제로 TSMC 대본에서 "AI" 매칭이 부분 문자열 기준 114회,
    단어 경계 기준 45회로 2배 이상 차이났다.
    """
    full_text = " ".join(transcript_df[text_col].dropna())
    return {
        kw: len(re.findall(rf"\b{re.escape(kw)}\b", full_text, re.IGNORECASE))
        for kw in keywords
    }


def compare_mentions_qoq(prior_counts, current_counts):
    """키워드별 언급 횟수를 전분기(prior) 대비 당분기(current)로 비교한다.

    전분기에 0회였다가 이번에 나온 경우 pct_change는 계산 불가능하므로 inf로 남긴다(호출 측에서
    "신규 언급"으로 표시하면 된다).
    """
    rows = []
    for kw in current_counts:
        prior = prior_counts.get(kw, 0)
        current = current_counts[kw]
        if prior > 0:
            pct_change = current / prior - 1
        else:
            pct_change = float("inf") if current > 0 else 0.0
        rows.append({"keyword": kw, "prior_count": prior, "current_count": current, "pct_change": pct_change})
    return pd.DataFrame(rows)


def extract_speaker_statements(transcript_df, speaker_role, role_col="speaker_role", text_col="text"):
    """특정 역할(예: "CEO", "CFO")의 발언만 등장 순서대로 리스트로 모은다."""
    return transcript_df.loc[transcript_df[role_col] == speaker_role, text_col].tolist()


def summarize_topic(store, topic, ticker=None, top_k=5, model="gemini-flash-latest", client=None):
    """topic(예: "guidance", "risk", "margin", "demand" 또는 임의의 질문 문장)에 대해 RAG로 관련 발언을
    찾고, 그 발언만 근거로 LLM에게 요약을 요청한다. 관련 발언을 하나도 못 찾으면 요약 없이 빈 결과를
    돌려준다(LLM에게 근거 없이 지어내게 하지 않기 위함).
    """
    query = _TOPIC_QUERIES.get(topic, topic)
    group_cols = ("ticker",) if ticker else ()
    results = retrieval.search(store, query, top_k=top_k, group_cols=group_cols)
    if results.empty:
        return {"summary": None, "citations": []}

    context = "\n\n".join(f"[{row['citation']}]\n{row['text']}" for _, row in results.iterrows())
    prompt = (
        f"다음은 실적발표 컨퍼런스콜에서 '{topic}'와 관련된 발언 목록이다. 이 발언 내용만 근거로 2~3문장"
        f"으로 요약하라. 발언에 없는 내용은 추측하거나 언급하지 마라.\n\n{context}"
    )

    client = client or get_llm_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return {"summary": response.text.strip(), "citations": results["citation"].tolist()}

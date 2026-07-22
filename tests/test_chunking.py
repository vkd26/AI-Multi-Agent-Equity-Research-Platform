"""chunking.py의 정제/분할/병합/메타데이터 부착 로직을 검증한다.

토큰 계산은 실제 BGE-M3 토크나이저를 쓴다(허깅페이스에서 캐시됨, 전체 모델 가중치는 로드하지 않음).
"""
import pandas as pd

from src.rag.chunking import chunk_dataframe, chunk_text, clean_text, count_tokens


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n\n  world  ") == "hello world"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_chunk_text_returns_single_chunk_when_short():
    text = "This is a short sentence."
    assert chunk_text(text, chunk_size=400) == [text]


def test_chunk_text_returns_empty_list_for_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_text_by_token_count():
    sentence = "This is one sentence about quarterly earnings and revenue guidance. "
    text = sentence * 60  # 확실히 chunk_size(토큰)보다 길게
    assert count_tokens(text) > 100

    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 + 40 for c in chunks)  # 문장경계 확장분 여유
    # 문장 중간이 아니라 마침표 뒤에서 끝나야 한다
    assert chunks[0].endswith(".")


def test_chunk_dataframe_attaches_metadata_and_doc_ids():
    long_text = "This is one sentence about quarterly earnings and revenue guidance. " * 60
    df = pd.DataFrame({
        "text": ["short text", long_text],
        "speaker": ["Alice", "Bob"],
    })
    result = chunk_dataframe(
        df, text_col="text", chunk_size=100, overlap=20,
        metadata_cols=["speaker"], extra_metadata={"ticker": "NVDA"},
    )
    assert set(result.columns) == {"doc_id", "chunk_id", "text", "speaker", "ticker", "chunk_seq"}
    assert (result["ticker"] == "NVDA").all()
    assert (result[result["doc_id"] == 0]["speaker"] == "Alice").all()
    assert len(result[result["doc_id"] == 1]) > 1
    assert (result[result["doc_id"] == 1]["speaker"] == "Bob").all()


def test_chunk_dataframe_skips_empty_text_rows():
    df = pd.DataFrame({"text": ["", "real content"]})
    result = chunk_dataframe(df, text_col="text")
    assert len(result) == 1
    assert result.iloc[0]["text"] == "real content"


def test_chunk_dataframe_does_not_merge_by_default():
    df = pd.DataFrame({"text": ["short one", "short two", "short three"]})
    result = chunk_dataframe(df, text_col="text", chunk_size=400)
    # merge_adjacent=False가 기본값이므로 행마다 청크 1개씩, 서로 합쳐지면 안 된다
    assert len(result) == 3
    assert list(result["text"]) == ["short one", "short two", "short three"]


def test_chunk_dataframe_merge_adjacent_combines_short_rows_and_keeps_speaker_labels():
    df = pd.DataFrame({
        "text": ["Good afternoon everyone.", "Thanks, revenue was strong.", "Any questions?"],
        "speaker": ["Jeff Su", "Wendell Huang", "Jeff Su"],
    })
    result = chunk_dataframe(
        df, text_col="text", chunk_size=400, overlap=20,
        metadata_cols=["speaker"], merge_adjacent=True, speaker_col="speaker",
    )
    # 세 행이 다 짧으므로 하나의 청크로 병합되어야 한다
    assert len(result) == 1
    merged = result.iloc[0]
    assert "Jeff Su: Good afternoon everyone." in merged["text"]
    assert "Wendell Huang: Thanks, revenue was strong." in merged["text"]
    # 중복 없이 등장 순서대로 화자가 남아야 한다 (Jeff Su가 두 번 말해도 한 번만)
    assert merged["speaker"] == "Jeff Su, Wendell Huang"


def test_chunk_dataframe_merge_adjacent_splits_when_row_alone_exceeds_chunk_size():
    long_text = "This is one sentence about quarterly earnings and revenue guidance. " * 60
    df = pd.DataFrame({
        "text": ["short intro", long_text],
        "speaker": ["Jeff Su", "Wendell Huang"],
    })
    result = chunk_dataframe(
        df, text_col="text", chunk_size=100, overlap=20,
        metadata_cols=["speaker"], merge_adjacent=True, speaker_col="speaker",
    )
    # 첫 행은 짧아서 단독 청크, 두 번째 행은 그 자체로 chunk_size를 넘어 여러 청크로 분할된다
    assert result.iloc[0]["speaker"] == "Jeff Su"
    assert len(result) > 2
    assert (result.iloc[1:]["speaker"] == "Wendell Huang").all()


def test_chunk_dataframe_adds_sequential_chunk_seq():
    df = pd.DataFrame({"text": ["a", "b", "c"]})
    result = chunk_dataframe(df, text_col="text", chunk_size=400)
    assert list(result["chunk_seq"]) == [0, 1, 2]


def test_chunk_dataframe_qa_aware_splits_new_question_from_previous_answer():
    # Analyst A 질문 -> CFO 답변 -> Analyst B 질문 -> CEO 답변. 전부 짧아서 토큰 한도로는 안 갈리지만,
    # 새 애널리스트가 등장하면 새 청크로 끊겨야 한다 (서로 다른 주제가 섞이면 안 되므로).
    df = pd.DataFrame({
        "text": ["CapEx 전망은?", "30% 증가할 것으로 봅니다.", "중국 매출 전망은?", "불확실성이 있습니다."],
        "speaker": ["Analyst A", "CFO", "Analyst B", "CEO"],
        "speaker_type": ["analyst", "management", "analyst", "management"],
        "section": ["qa", "qa", "qa", "qa"],
    })
    result = chunk_dataframe(
        df, text_col="text", chunk_size=400, overlap=20,
        metadata_cols=["speaker"], merge_adjacent=True,
        speaker_col="speaker", section_col="section", speaker_type_col="speaker_type",
    )
    assert len(result) == 2
    assert "Analyst A" in result.iloc[0]["text"] and "CFO" in result.iloc[0]["text"]
    assert "Analyst B" in result.iloc[1]["text"] and "CEO" in result.iloc[1]["text"]
    # CapEx 질문/답변에 중국 매출 얘기가 섞이면 안 된다
    assert "중국" not in result.iloc[0]["text"]


def test_chunk_dataframe_qa_aware_does_not_split_multi_paragraph_question():
    # 한 애널리스트가 질문을 두 문단에 걸쳐 이어가는 경우 — 아직 경영진 답변을 받기 전이므로 갈리면 안 된다
    df = pd.DataFrame({
        "text": ["첫 번째 질문입니다.", "이어서 두 번째 부분입니다.", "네, 답변드리겠습니다."],
        "speaker": ["Analyst A", "Analyst A", "CFO"],
        "speaker_type": ["analyst", "analyst", "management"],
        "section": ["qa", "qa", "qa"],
    })
    result = chunk_dataframe(
        df, text_col="text", chunk_size=400, overlap=20,
        metadata_cols=["speaker"], merge_adjacent=True,
        speaker_col="speaker", section_col="section", speaker_type_col="speaker_type",
    )
    assert len(result) == 1
    assert "첫 번째 질문" in result.iloc[0]["text"]
    assert "이어서 두 번째" in result.iloc[0]["text"]
    assert "답변드리겠습니다" in result.iloc[0]["text"]


def test_chunk_dataframe_section_change_forces_new_chunk_even_if_short():
    df = pd.DataFrame({
        "text": ["prepared remarks 발언입니다.", "첫 질문입니다."],
        "speaker": ["CEO", "Analyst A"],
        "speaker_type": ["management", "analyst"],
        "section": ["prepared_remarks", "qa"],
    })
    result = chunk_dataframe(
        df, text_col="text", chunk_size=400, overlap=20,
        metadata_cols=["speaker", "section"], merge_adjacent=True,
        speaker_col="speaker", section_col="section", speaker_type_col="speaker_type",
    )
    assert len(result) == 2
    assert result.iloc[0]["section"] == "prepared_remarks"
    assert result.iloc[1]["section"] == "qa"

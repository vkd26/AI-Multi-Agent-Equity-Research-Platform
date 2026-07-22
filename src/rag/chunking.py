"""Document Processor — 수집된 문서(실적발표 대본, 뉴스, 공시 등)를 정제(Cleaning)하고 임베딩하기
좋은 크기로 분할(Chunking)하며, 검색 시 근거로 제시할 메타데이터(티커/소스/화자 등)를 붙인다.

크기는 글자 수가 아니라 임베딩 모델(BGE-M3)의 실제 토큰 수 기준이다 — embedding.py와 같은 모델을
쓰지만, 토크나이저만 필요하므로(전체 가중치 로딩 불필요) 이 모듈은 embedding.py에 의존하지 않는다.
"""
import re

import pandas as pd

_MODEL_NAME = "BAAI/bge-m3"  # embedding.py와 동일한 모델 — 토크나이저만 가볍게 로드해 쓴다
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]\s")

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return _tokenizer


def count_tokens(text):
    return len(_get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def clean_text(text):
    """중복 공백·줄바꿈을 정리하고 앞뒤 공백을 제거한다."""
    if not isinstance(text, str):
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text, chunk_size=400, overlap=50):
    """긴 텍스트를 chunk_size 토큰 단위로, overlap 토큰만큼 겹치게 분할한다.

    가능하면 문장 경계(. ! ? 줄바꿈)에서 자르려고 시도하고, 경계가 없으면 chunk_size에서 그냥 자른다.
    다음 청크의 겹침 시작 위치는 문장 경계로 늘어나기 전의 원래 chunk_size 지점 기준으로 계산한다
    (경계 탐색으로 늘어난 몇 글자 때문에 매번 재토큰화하지 않기 위한 단순화).
    """
    text = clean_text(text)
    if not text:
        return []

    encoding = _get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    if len(offsets) <= chunk_size:
        return [text]

    chunks = []
    tok_start = 0
    while tok_start < len(offsets):
        tok_end = min(tok_start + chunk_size, len(offsets))
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end - 1][1]
        if tok_end < len(offsets):
            boundary = _SENTENCE_BOUNDARY_RE.search(text[char_end:char_end + 200])
            if boundary:
                char_end += boundary.end()
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
        if tok_end >= len(offsets):
            break
        tok_start = max(tok_end - overlap, tok_start + 1)
    return chunks


def chunk_dataframe(
    df, text_col="text", chunk_size=400, overlap=50, metadata_cols=None, extra_metadata=None,
    merge_adjacent=False, speaker_col=None, section_col=None, speaker_type_col=None,
):
    """DataFrame의 text_col을 청킹하며 메타데이터를 각 청크에 붙인다. 결과에는 검색 후 인접 청크를
    가져올 수 있도록 문서 순서를 보존하는 chunk_seq(0부터 시작하는 일련번호)가 항상 붙는다.

    merge_adjacent=False(기본): 행 1개(뉴스 기사 1건, 공시 1건 등 서로 무관한 문서)를 문서 1개로 보고,
    chunk_size보다 길면 여러 청크로 쪼갠다. 서로 다른 행끼리는 절대 합치지 않는다.

    merge_adjacent=True: 대본처럼 순서가 있는 짧은 발언들이 이어지는 데이터용 — 인접한 짧은 행들을
    chunk_size 안에서 최대한 하나의 청크로 합쳐서, 화자 한 명이 한두 문장만 말한 경우에도 앞뒤 문맥이
    함께 남도록 한다. speaker_col을 주면 합쳐진 텍스트 안에 "화자: 발언" 라벨을 인라인으로 남기고,
    청크의 metadata_cols[speaker_col]에는 그 청크에 포함된 화자들을 등장 순서대로 콤마 연결해 남긴다.

    section_col/speaker_type_col을 추가로 주면(예: transcript.py가 만드는 section/speaker_type 컬럼)
    토큰 한도와 별개로 두 가지 경계에서 강제로 새 청크를 시작한다:
    - section이 바뀌는 지점(예: prepared_remarks -> qa)
    - qa 섹션에서, 경영진 답변까지 받은 뒤 새 애널리스트가 등장하는 지점(= 새 질문 시작)
    이렇게 하면 서로 다른 애널리스트의 질문-답변이 한 청크에 섞이지 않는다.
    """
    metadata_cols = metadata_cols or []
    extra_metadata = extra_metadata or {}

    if merge_adjacent:
        result = _merge_and_chunk(
            df, text_col, chunk_size, overlap, metadata_cols, extra_metadata,
            speaker_col, section_col, speaker_type_col,
        )
    else:
        rows = []
        for doc_id, row in df.iterrows():
            pieces = chunk_text(row[text_col], chunk_size=chunk_size, overlap=overlap)
            for chunk_idx, piece in enumerate(pieces):
                record = {"doc_id": doc_id, "chunk_id": f"{doc_id}-{chunk_idx}", "text": piece}
                for col in metadata_cols:
                    record[col] = row.get(col)
                record.update(extra_metadata)
                rows.append(record)
        result = pd.DataFrame(rows)

    if not result.empty:
        result = result.reset_index(drop=True)
        result["chunk_seq"] = result.index
    return result


def _format_row(row, text, speaker_col):
    if speaker_col and pd.notna(row.get(speaker_col)):
        return f"{row[speaker_col]}: {text}"
    return text


def _merge_and_chunk(
    df, text_col, chunk_size, overlap, metadata_cols, extra_metadata,
    speaker_col, section_col, speaker_type_col,
):
    rows = []
    buffer = []  # [(doc_id, formatted_text, original_row)]
    buffer_tokens = 0
    buffer_section = None
    buffer_has_management = False

    def flush():
        nonlocal buffer, buffer_tokens, buffer_section, buffer_has_management
        if not buffer:
            return
        doc_ids = [doc_id for doc_id, _, _ in buffer]
        merged_text = "\n\n".join(text for _, text, _ in buffer)
        chunk_id = f"{doc_ids[0]}-{doc_ids[-1]}"

        record = {"doc_id": chunk_id, "chunk_id": chunk_id, "text": merged_text}
        for col in metadata_cols:
            if col == speaker_col:
                seen = []
                for _, _, r in buffer:
                    val = r.get(speaker_col)
                    if pd.notna(val) and val not in seen:
                        seen.append(val)
                record[col] = ", ".join(seen)
            else:
                record[col] = buffer[0][2].get(col)
        record.update(extra_metadata)
        rows.append(record)
        buffer, buffer_tokens, buffer_section, buffer_has_management = [], 0, None, False

    for doc_id, row in df.iterrows():
        text = clean_text(row[text_col])
        if not text:
            continue
        formatted = _format_row(row, text, speaker_col)
        tok_count = count_tokens(formatted)
        row_section = row.get(section_col) if section_col else None
        row_speaker_type = row.get(speaker_type_col) if speaker_type_col else None

        # 섹션이 바뀌면(예: prepared_remarks -> qa) 무조건 새 청크로 끊는다
        if buffer and section_col and row_section != buffer_section:
            flush()

        # qa 섹션에서 경영진 답변까지 받은 뒤 새 애널리스트가 등장하면 새 질문 블록으로 본다
        if (
            buffer and speaker_type_col and row_section == "qa"
            and row_speaker_type == "analyst" and buffer_has_management
        ):
            flush()

        if tok_count > chunk_size:
            # 이 행 하나가 이미 한도를 넘으면 버퍼부터 비우고, 이 행만 따로 문장 경계 기준으로 분할한다
            flush()
            for chunk_idx, piece in enumerate(chunk_text(formatted, chunk_size=chunk_size, overlap=overlap)):
                record = {"doc_id": doc_id, "chunk_id": f"{doc_id}-{chunk_idx}", "text": piece}
                for col in metadata_cols:
                    record[col] = row.get(col)
                record.update(extra_metadata)
                rows.append(record)
            continue

        if buffer_tokens + tok_count > chunk_size:
            flush()

        if not buffer:
            buffer_section = row_section
        buffer.append((doc_id, formatted, row))
        buffer_tokens += tok_count
        if row_speaker_type == "management":
            buffer_has_management = True

    flush()
    return pd.DataFrame(rows)

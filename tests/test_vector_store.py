"""vector_store.py의 인덱싱/검색/영속성을 합성 임베딩으로 검증한다 (임베딩 모델 로딩 없음)."""
import numpy as np
import pandas as pd
import pytest

from src.rag.vector_store import VectorStore, _tokenize


def _chunks_df():
    # 서로 직교에 가까운 4차원 합성 임베딩 — 벡터 검색 결과를 예측 가능하게 하기 위함
    return pd.DataFrame({
        "chunk_id": ["a", "b", "c"],
        "text": ["TSMC CapEx guidance for next year", "quarterly revenue grew strongly", "risk factors and headwinds"],
        "speaker": ["Jeff Su", "Wendell Huang", "C.C. Wei"],
        "embedding": [
            np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
        ],
    })


def test_tokenize_lowercases_and_splits_words():
    assert _tokenize("TSMC's CapEx Guidance!") == ["tsmc", "s", "capex", "guidance"]


def test_add_populates_index_and_metadata():
    store = VectorStore(dim=4)
    store.add(_chunks_df())
    assert store.index.ntotal == 3
    assert len(store.metadata) == 3
    assert "embedding" not in store.metadata.columns


def test_add_assigns_globally_unique_row_id_even_with_colliding_chunk_ids():
    # 서로 다른 문서에서 온 두 배치가 chunk_id("a")를 재사용해도(각자 다른 문서에서 0번째 청크였다는 뜻),
    # _row_id는 스토어 전체에서 유일해야 한다 — retrieval.py의 RRF가 이걸로 매칭하기 때문
    store = VectorStore(dim=4)
    store.add(_chunks_df())
    store.add(_chunks_df())
    assert store.metadata["_row_id"].is_unique
    assert list(store.metadata["_row_id"]) == [0, 1, 2, 3, 4, 5]


def test_search_vector_returns_closest_match_first():
    store = VectorStore(dim=4)
    store.add(_chunks_df())
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    result = store.search_vector(query, top_k=2)
    assert len(result) == 2
    assert result.iloc[0]["chunk_id"] == "a"
    assert result.iloc[0]["score"] > result.iloc[1]["score"]


def test_search_bm25_matches_keyword():
    store = VectorStore(dim=4)
    store.add(_chunks_df())
    result = store.search_bm25("CapEx guidance", top_k=1)
    assert result.iloc[0]["chunk_id"] == "a"


def test_search_on_empty_store_returns_empty_dataframe():
    store = VectorStore(dim=4)
    query = np.zeros(4, dtype="float32")
    assert store.search_vector(query, top_k=5).empty
    assert store.search_bm25("anything", top_k=5).empty


def test_save_and_load_round_trip_preserves_search_results(tmp_path):
    store = VectorStore(dim=4)
    store.add(_chunks_df())
    store.save(str(tmp_path / "store"))

    loaded = VectorStore.load(str(tmp_path / "store"))
    assert loaded.index.ntotal == 3
    assert len(loaded.metadata) == 3

    query = np.array([0.0, 0.9, 0.1, 0.0], dtype="float32")
    original = store.search_vector(query, top_k=2)
    restored = loaded.search_vector(query, top_k=2)
    assert list(original["chunk_id"]) == list(restored["chunk_id"])

    bm25_restored = loaded.search_bm25("revenue", top_k=1)
    assert bm25_restored.iloc[0]["chunk_id"] == "b"

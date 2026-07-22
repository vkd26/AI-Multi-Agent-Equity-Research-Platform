"""retrieval.py의 RRF 결합/인접 청크 확장/출처 포맷팅을 검증한다 (리랭커 모델 로딩 없음, 모킹).

rerank()/hybrid_search()는 실제 임베딩·리랭커 모델이 필요해 로컬에서 수동으로 검증했다(README 참고).
여기서는 모델 의존 없이 동작하는 순수 로직만 자동화 테스트로 커버한다.
"""
from unittest.mock import patch

import pandas as pd

from src.rag import retrieval


def test_reciprocal_rank_fusion_combines_and_boosts_overlap():
    vector_results = pd.DataFrame({
        "_row_id": [1, 2, 3], "text": ["a", "b", "c"], "score": [0.9, 0.8, 0.7],
    })
    bm25_results = pd.DataFrame({
        "_row_id": [2, 1, 4], "text": ["b", "a", "d"], "score": [5.0, 4.0, 3.0],
    })
    fused = retrieval.reciprocal_rank_fusion([vector_results, bm25_results])

    # row_id 1과 2는 두 검색 모두에 등장하므로 4(BM25에만 등장)보다 순위가 높아야 한다
    assert list(fused["_row_id"])[:2] == [1, 2] or list(fused["_row_id"])[:2] == [2, 1]
    assert fused.iloc[-1]["_row_id"] == 4
    assert "score" not in fused.columns  # 원래 개별 검색 score는 rrf_score로 대체되어야 한다


def test_reciprocal_rank_fusion_empty_inputs_returns_empty():
    result = retrieval.reciprocal_rank_fusion([pd.DataFrame(), pd.DataFrame()])
    assert result.empty


def test_expand_with_neighbors_respects_group_cols_boundary():
    store = type("Store", (), {})()
    store.metadata = pd.DataFrame({
        "ticker": ["TSM", "TSM", "TSM", "BAC", "BAC"],
        "chunk_seq": [0, 1, 2, 0, 1],
        "text": ["tsm-0", "tsm-1", "tsm-2", "bac-0", "bac-1"],
    })
    results = pd.DataFrame({"ticker": ["TSM"], "chunk_seq": [1], "text": ["tsm-1"]})

    expanded = retrieval.expand_with_neighbors(store, results, window=1, group_cols=("ticker",))
    # TSM의 이웃(0,1,2)만 들어가야 하고, chunk_seq가 우연히 겹치는 BAC-0/BAC-1은 섞이면 안 된다
    assert "tsm-0" in expanded.iloc[0]["context_text"]
    assert "tsm-2" in expanded.iloc[0]["context_text"]
    assert "bac" not in expanded.iloc[0]["context_text"]


def test_expand_with_neighbors_empty_results_returns_empty():
    store = type("Store", (), {"metadata": pd.DataFrame()})()
    result = retrieval.expand_with_neighbors(store, pd.DataFrame())
    assert result.empty


def test_format_citation_includes_available_fields_only():
    row = pd.Series({
        "ticker": "TSM", "fiscal_quarter": "Q2 2026", "speaker": "C.C. Wei",
        "speaker_role": "CEO", "section": "qa", "chunk_id": "28-40",
    })
    assert retrieval.format_citation(row) == "TSM | Q2 2026 | C.C. Wei | CEO | qa | chunk 28-40"


def test_format_citation_skips_missing_fields():
    row = pd.Series({"ticker": "TSM", "chunk_id": "5-0"})
    assert retrieval.format_citation(row) == "TSM | chunk 5-0"


def test_add_citations_attaches_citation_column():
    results = pd.DataFrame({"ticker": ["TSM"], "chunk_id": ["1-0"]})
    result = retrieval.add_citations(results)
    assert result.iloc[0]["citation"] == "TSM | chunk 1-0"


def test_search_pipeline_wires_stages_together():
    # 파이프라인의 각 단계가 올바른 인자로 호출되고 결과가 이어지는지만 확인 — 실제 모델은 모킹
    store = object()
    hybrid_df = pd.DataFrame({"_row_id": [1], "text": ["candidate"], "chunk_seq": [0], "chunk_id": ["0-0"]})
    reranked_df = hybrid_df.assign(rerank_score=[0.9])
    expanded_df = reranked_df.assign(context_text=["candidate"])

    with patch.object(retrieval, "hybrid_search", return_value=hybrid_df) as mock_hybrid, \
         patch.object(retrieval, "rerank", return_value=reranked_df) as mock_rerank, \
         patch.object(retrieval, "expand_with_neighbors", return_value=expanded_df) as mock_expand:
        result = retrieval.search(store, "query", top_k=3, group_cols=("ticker",))

    mock_hybrid.assert_called_once()
    mock_rerank.assert_called_once()
    mock_expand.assert_called_once()
    assert result.iloc[0]["citation"] == "chunk 0-0"

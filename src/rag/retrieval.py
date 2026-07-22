"""Financial RAG — 검색. 벡터+BM25 하이브리드 검색(RRF) -> BGE-Reranker 재정렬 -> 인접 청크로 문맥
확장 -> 출처(citation) 포함 결과 반환까지의 파이프라인.
"""
import pandas as pd
from sentence_transformers import CrossEncoder

from src.rag.embedding import embed_texts

_RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_NAME)
    return _reranker


def reciprocal_rank_fusion(result_dfs, k=60):
    """여러 검색 결과(각각 순위대로 정렬된 DataFrame)를 RRF로 합쳐 rrf_score 기준으로 정렬해 반환한다.

    각 결과는 VectorStore.search_vector/search_bm25의 반환값이어야 한다(고유 식별자 _row_id 포함).
    같은 청크가 여러 방식에서 검색되면 순위 기반 점수가 합산되어 더 높은 순위를 받는다.
    """
    scores = {}
    meta_lookup = {}
    for df in result_dfs:
        for rank, (_, row) in enumerate(df.iterrows()):
            row_id = row["_row_id"]
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank + 1)
            meta_lookup[row_id] = row

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    rows = []
    for row_id in ranked_ids:
        record = meta_lookup[row_id].drop(labels=["score"], errors="ignore").to_dict()
        record["rrf_score"] = scores[row_id]
        rows.append(record)
    return pd.DataFrame(rows)


def hybrid_search(store, query, top_k=20, vector_top_k=20, bm25_top_k=20):
    """벡터 검색과 BM25 검색을 각각 돌려 RRF로 합친 상위 top_k개를 반환한다."""
    query_embedding = embed_texts([query])[0]
    vector_results = store.search_vector(query_embedding, top_k=vector_top_k)
    bm25_results = store.search_bm25(query, top_k=bm25_top_k)
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused.head(top_k).reset_index(drop=True)


def rerank(query, candidates, text_col="text", top_k=5):
    """BGE-Reranker(쿼리-후보 쌍을 직접 채점하는 cross-encoder)로 candidates를 재정렬한다."""
    if candidates.empty:
        return candidates
    pairs = [(query, text) for text in candidates[text_col]]
    scores = get_reranker().predict(pairs)
    result = candidates.copy()
    result["rerank_score"] = scores
    return result.sort_values("rerank_score", ascending=False).head(top_k).reset_index(drop=True)


def expand_with_neighbors(store, results, window=1, group_cols=()):
    """각 결과 청크에 같은 문서 내 앞뒤 window개 청크를 이어붙인 context_text를 추가한다.

    group_cols로 문서 경계를 지정해야 한다(예: ("ticker", "fiscal_quarter")) — 안 주면 스토어에 문서가
    하나뿐이라고 가정하고 chunk_seq만으로 이웃을 찾는다. 여러 문서가 섞인 스토어에서 group_cols 없이
    쓰면 다른 문서의 청크를 이웃으로 잘못 가져올 수 있다.
    """
    if results.empty:
        return results

    meta = store.metadata
    rows = []
    for _, row in results.iterrows():
        mask = pd.Series(True, index=meta.index)
        for col in group_cols:
            mask &= meta[col] == row[col]
        mask &= meta["chunk_seq"].between(row["chunk_seq"] - window, row["chunk_seq"] + window)
        neighbors = meta[mask].sort_values("chunk_seq")

        record = row.to_dict()
        record["context_text"] = "\n\n".join(neighbors["text"])
        rows.append(record)
    return pd.DataFrame(rows)


def format_citation(row):
    """사람이 읽을 수 있는 출처 문자열을 만든다. 있는 필드만 골라 " | "로 잇는다."""
    parts = []
    for col in ("ticker", "fiscal_quarter", "speaker", "speaker_role", "section"):
        value = row.get(col)
        if pd.notna(value) and value:
            parts.append(str(value))
    if row.get("chunk_id"):
        parts.append(f"chunk {row['chunk_id']}")
    return " | ".join(parts)


def add_citations(results):
    if results.empty:
        return results
    result = results.copy()
    result["citation"] = result.apply(format_citation, axis=1)
    return result


def search(store, query, top_k=5, rerank_candidates=20, expand_neighbors=True, group_cols=()):
    """전체 파이프라인: 하이브리드 검색 -> 리랭크 -> (옵션) 인접 청크 확장 -> 출처 부착."""
    candidates = hybrid_search(store, query, top_k=rerank_candidates)
    reranked = rerank(query, candidates, top_k=top_k)
    if expand_neighbors:
        reranked = expand_with_neighbors(store, reranked, group_cols=group_cols)
    return add_citations(reranked)

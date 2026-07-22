"""Financial RAG — 벡터 스토어. FAISS(밀집 벡터) + BM25(키워드) 인덱스를 함께 관리한다.

원래 ChromaDB(메타데이터 필터링 내장, 설정 간편)를 쓰려 했는데, 이 개발 환경(Windows + Python 3.13)에서
chromadb의 grpc 네이티브 의존성이 import 단계에서 깨져서(cygrpc DLL 로드 실패) FAISS로 바꿨다. FAISS는
메타데이터 필터링이 내장이 아니라서, 임베딩과 별개로 메타데이터를 pandas DataFrame으로 나란히 들고
있다가 검색 후 직접 필터링한다.

BM25는 하이브리드 검색(retrieval.py)에서 밀집 벡터 검색과 결합해 쓴다 — 금융 텍스트는 "2nm", "CapEx"
같은 고유명사·숫자가 많아 키워드 매칭이 의미 임베딩을 보완해준다. 토큰화는 단순 정규식 기반이라
영어에는 잘 맞지만 한국어는 형태소 분석 없이 공백 기준이라 정밀도가 떨어진다는 한계가 있다.
"""
import os
import pickle
import re

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class VectorStore:
    def __init__(self, dim=1024):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 정규화된 임베딩의 내적 = 코사인 유사도
        self.metadata = pd.DataFrame()  # FAISS 인덱스의 행 순서와 1:1로 대응
        self._tokenized_corpus = []
        self.bm25 = None

    def add(self, chunks_df, embedding_col="embedding", text_col="text"):
        """embed_chunks()의 결과(embedding 컬럼 포함)를 벡터·BM25 인덱스에 추가한다.

        chunk_id는 chunk_dataframe() 호출 단위로 새로 매겨지므로(서로 다른 문서가 같은 chunk_id를 가질
        수 있음) 스토어 전역에서 유일함을 보장하는 _row_id를 별도로 부여한다 — retrieval.py가 벡터/BM25
        검색 결과를 같은 청크로 매칭(RRF)할 때 이 값을 join key로 쓴다.
        """
        if chunks_df.empty:
            return
        embeddings = np.vstack(chunks_df[embedding_col].to_numpy()).astype("float32")
        start_id = len(self.metadata)
        self.index.add(embeddings)

        meta = chunks_df.drop(columns=[embedding_col]).reset_index(drop=True)
        meta["_row_id"] = range(start_id, start_id + len(meta))
        self.metadata = pd.concat([self.metadata, meta], ignore_index=True)

        self._tokenized_corpus.extend(_tokenize(t) for t in chunks_df[text_col])
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search_vector(self, query_embedding, top_k=10):
        """쿼리 임베딩과 코사인 유사도가 가장 높은 top_k개를 (메타데이터 + score) DataFrame으로 반환한다."""
        if self.index.ntotal == 0:
            return self.metadata.iloc[0:0].assign(score=[])
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, idx = self.index.search(query, min(top_k, self.index.ntotal))
        result = self.metadata.iloc[idx[0]].copy()
        result["score"] = scores[0]
        return result.reset_index(drop=True)

    def search_bm25(self, query, top_k=10):
        """BM25 키워드 검색 상위 top_k개를 (메타데이터 + score) DataFrame으로 반환한다."""
        if self.bm25 is None or len(self.metadata) == 0:
            return self.metadata.iloc[0:0].assign(score=[])
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        result = self.metadata.iloc[top_idx].copy()
        result["score"] = scores[top_idx]
        return result.reset_index(drop=True)

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        self.metadata.to_parquet(os.path.join(dir_path, "metadata.parquet"))
        with open(os.path.join(dir_path, "bm25_corpus.pkl"), "wb") as f:
            pickle.dump(self._tokenized_corpus, f)

    @classmethod
    def load(cls, dir_path):
        index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        store = cls(dim=index.d)
        store.index = index
        store.metadata = pd.read_parquet(os.path.join(dir_path, "metadata.parquet"))
        with open(os.path.join(dir_path, "bm25_corpus.pkl"), "rb") as f:
            store._tokenized_corpus = pickle.load(f)
        store.bm25 = BM25Okapi(store._tokenized_corpus) if store._tokenized_corpus else None
        return store

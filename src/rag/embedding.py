"""Financial RAG — 임베딩. BAAI/bge-m3(다국어, 한국어 성능 우수)로 텍스트를 벡터화한다.

API 키가 필요 없는 로컬 모델이라 재현성이 좋다(클론 후 별도 키 설정 없이 그대로 실행 가능). 대신 첫
실행 시 모델(~2GB)을 허깅페이스에서 내려받는다. 한/영 교차 검색 성능은 검증됨(README 참고).
"""
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-m3"
_model = None


def get_model():
    """모델을 프로세스당 한 번만 로드해 재사용한다 (로딩 비용이 크다)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=32):
    """텍스트 리스트를 정규화된(코사인 유사도 = 내적) 임베딩 배열로 변환한다."""
    model = get_model()
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True)


def embed_chunks(chunks_df, text_col="text", batch_size=32):
    """chunking.chunk_dataframe()의 결과에 embedding 컬럼(벡터)을 추가한다."""
    embeddings = embed_texts(chunks_df[text_col].tolist(), batch_size=batch_size)
    result = chunks_df.copy()
    result["embedding"] = list(embeddings)
    return result

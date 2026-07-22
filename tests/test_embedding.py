"""embedding.py의 인터페이스를 검증한다 (모델 로딩 없음, SentenceTransformer는 모킹).

실제 BGE-M3 모델의 임베딩 품질(한/영 교차 검색)은 로컬에서 수동으로 검증했다 — CI에서 매번 ~2GB
모델을 내려받는 건 비현실적이라 여기서는 embed_texts/embed_chunks가 모델 출력을 올바르게 다루는지만
확인한다.
"""
import numpy as np
import pandas as pd

from src.rag import embedding


def test_embed_texts_calls_model_with_normalize_and_returns_output(monkeypatch):
    mock_model = type("MockModel", (), {
        "encode": lambda self, texts, batch_size=32, normalize_embeddings=True: np.array(
            [[float(len(t)), 0.0] for t in texts]
        )
    })()
    monkeypatch.setattr(embedding, "get_model", lambda: mock_model)

    result = embedding.embed_texts(["ab", "abcd"])
    assert result.tolist() == [[2.0, 0.0], [4.0, 0.0]]


def test_embed_chunks_attaches_embedding_column(monkeypatch):
    mock_model = type("MockModel", (), {
        "encode": lambda self, texts, batch_size=32, normalize_embeddings=True: np.array(
            [[1.0, 0.0], [0.0, 1.0]]
        )
    })()
    monkeypatch.setattr(embedding, "get_model", lambda: mock_model)

    df = pd.DataFrame({"chunk_id": ["0-0", "1-0"], "text": ["first", "second"]})
    result = embedding.embed_chunks(df)

    assert "embedding" in result.columns
    assert result.iloc[0]["embedding"].tolist() == [1.0, 0.0]
    assert result.iloc[1]["embedding"].tolist() == [0.0, 1.0]
    # 원본 컬럼은 그대로 유지되어야 한다
    assert list(result["chunk_id"]) == ["0-0", "1-0"]


def test_get_model_caches_singleton(monkeypatch):
    call_count = {"n": 0}

    class DummyModel:
        pass

    def fake_ctor(name):
        call_count["n"] += 1
        return DummyModel()

    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setattr(embedding, "SentenceTransformer", fake_ctor)

    m1 = embedding.get_model()
    m2 = embedding.get_model()
    assert m1 is m2
    assert call_count["n"] == 1

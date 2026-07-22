"""analytics/_common.py의 LLM 클라이언트 초기화 에러 처리를 검증한다."""
import pytest

from src.analytics import _common


def test_get_llm_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(_common, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        _common.get_llm_client()

"""analytics 모듈들이 공유하는 LLM 클라이언트 초기화 로직."""
import os

from dotenv import load_dotenv

from src.config import PROJECT_ROOT


def get_llm_client():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되어 있지 않다. https://aistudio.google.com 에서 API 키를 발급받아 "
            "프로젝트 루트의 .env 또는 홈 디렉토리(~/.env)에 GEMINI_API_KEY=발급받은키 형태로 저장할 것."
        )
    from google import genai
    return genai.Client(api_key=api_key)

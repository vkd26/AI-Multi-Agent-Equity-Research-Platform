"""단일 진입점 Equity Research Agent — planner.py의 tool-calling 루프에 tools.py의 도구 전체를
등록해서, 자연어 질문 하나로 필요한 조사를 알아서 수행하게 한다.

원래 스펙은 research_agent/valuation_agent/report_agent를 단계별로 나눠 순차 실행하는 구조였는데,
실제 워크플로가 거의 선형이라 LangGraph 같은 그래프 프레임워크를 쓸 이유가 약했다(자세한 이유는
planner.py 참고). 대신 도구 전부를 한 에이전트에 등록해 LLM이 질문에 따라 필요한 것만 동적으로
골라 쓰게 했다 — 간단한 질문("PER 얼마야?")은 도구 1개로 끝나고, 복잡한 질문("전체 분석해줘")은
여러 도구를 체이닝한다.
"""
from src.agents.planner import run_agent_loop
from src.agents.tools import TOOLS
from src.analytics._common import get_llm_client

_SYSTEM_INSTRUCTION = (
    "너는 주식 리서치 애널리스트를 돕는 AI 에이전트다. 사용자의 질문에 답하기 위해 필요한 도구를 "
    "스스로 판단해서 호출하라. 도구 결과에 없는 내용은 추측하지 말고, 정보가 없으면 없다고 답하라. "
    "숫자를 인용할 때는 근거(어떤 조회 결과에서 나온 값인지)를 함께 제시하라."
)


def run_equity_research_agent(query, client=None, model="gemini-flash-latest", max_turns=8):
    """자연어 질문을 받아 필요한 도구를 판단해 호출하며 답변한다.

    반환: (답변 텍스트, 호출된 도구 로그) — 로그는 어떤 근거로 답했는지 추적하는 용도.
    """
    client = client or get_llm_client()
    return run_agent_loop(
        client, model, TOOLS, query, system_instruction=_SYSTEM_INSTRUCTION, max_turns=max_turns,
    )

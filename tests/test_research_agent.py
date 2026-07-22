"""research_agent.py의 진입점이 planner.run_agent_loop을 올바른 인자(전체 도구 레지스트리,
시스템 프롬프트)로 호출하는지 검증한다 (LLM 클라이언트/루프는 모킹).

실제 멀티턴 도구 체이닝(질문 -> 도구 3개 순차 호출 -> 답변)은 NVDA로 수동 검증했다(README 참고).
"""
from unittest.mock import MagicMock, patch

from src.agents.research_agent import run_equity_research_agent
from src.agents.tools import TOOLS


def test_run_equity_research_agent_passes_full_tool_registry_and_system_instruction():
    client = MagicMock()
    with patch("src.agents.research_agent.run_agent_loop", return_value=("답변", [])) as mock_loop:
        text, log = run_equity_research_agent("NVDA PER 얼마야?", client=client)

    mock_loop.assert_called_once()
    call_args = mock_loop.call_args
    assert call_args[0][0] is client
    assert call_args[0][2] == TOOLS
    assert call_args[0][3] == "NVDA PER 얼마야?"
    assert "system_instruction" in call_args.kwargs
    assert text == "답변"
    assert log == []


def test_run_equity_research_agent_creates_client_when_not_given():
    with patch("src.agents.research_agent.get_llm_client", return_value=MagicMock()) as mock_get_client, \
         patch("src.agents.research_agent.run_agent_loop", return_value=("답변", [])):
        run_equity_research_agent("질문")

    mock_get_client.assert_called_once()

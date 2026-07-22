"""planner.py의 tool-calling 루프 메커니즘을 검증한다 (LLM 클라이언트는 모킹).

실제 멀티턴 도구 호출(가격 조회 2번 -> 비교 1번 -> 최종 답변)은 Gemini 실 API로 수동 검증했다
(README 참고). 여기서는 루프 자체의 제어 흐름(반복 종료 조건, 에러 처리, 턴 제한)을 검증한다.
"""
from unittest.mock import MagicMock

import pytest

from src.agents.planner import run_agent_loop


def _mock_function_call(name, args):
    # MagicMock(name=...)는 목의 내부 디버그 이름을 설정하는 특수 인자라 .name 속성이 안 된다 —
    # 생성 후 직접 속성으로 대입해야 call.name이 실제로 "name" 값을 갖는다.
    call = MagicMock()
    call.name = name
    call.args = args
    return call


def _response_with_calls(calls, content=MagicMock()):
    resp = MagicMock()
    resp.function_calls = calls
    resp.candidates = [MagicMock(content=content)]
    return resp


def _response_with_text(text, content=MagicMock()):
    resp = MagicMock()
    resp.function_calls = None
    resp.text = text
    resp.candidates = [MagicMock(content=content)]
    return resp


def test_run_agent_loop_returns_immediately_when_no_tool_call_needed():
    client = MagicMock()
    client.models.generate_content.return_value = _response_with_text("바로 답변")

    text, log = run_agent_loop(client, "gemini-flash-latest", tools={}, user_message="안녕")

    assert text == "바로 답변"
    assert log == []
    assert client.models.generate_content.call_count == 1


def test_run_agent_loop_executes_single_tool_call_then_returns_final_answer():
    tool = MagicMock(return_value={"price": 207.29})
    call = _mock_function_call("get_price", {"ticker": "NVDA"})

    client = MagicMock()
    client.models.generate_content.side_effect = [
        _response_with_calls([call]),
        _response_with_text("NVDA는 $207.29입니다"),
    ]

    text, log = run_agent_loop(client, "gemini-flash-latest", tools={"get_price": tool}, user_message="NVDA 가격은?")

    tool.assert_called_once_with(ticker="NVDA")
    assert text == "NVDA는 $207.29입니다"
    assert log == [{"tool": "get_price", "args": {"ticker": "NVDA"}, "result": {"price": 207.29}}]
    assert client.models.generate_content.call_count == 2


def test_run_agent_loop_handles_multiple_sequential_tool_calls():
    price_tool = MagicMock(side_effect=[{"price": 207.29}, {"price": 165.50}])
    compare_tool = MagicMock(return_value={"higher": "NVDA"})

    client = MagicMock()
    client.models.generate_content.side_effect = [
        _response_with_calls([_mock_function_call("get_price", {"ticker": "NVDA"})]),
        _response_with_calls([_mock_function_call("get_price", {"ticker": "AMD"})]),
        _response_with_calls([_mock_function_call("compare", {"a": 207.29, "b": 165.50})]),
        _response_with_text("NVDA가 더 비쌉니다"),
    ]

    text, log = run_agent_loop(
        client, "gemini-flash-latest",
        tools={"get_price": price_tool, "compare": compare_tool},
        user_message="NVDA랑 AMD 중 뭐가 비싸?",
    )

    assert text == "NVDA가 더 비쌉니다"
    assert len(log) == 3
    assert [entry["tool"] for entry in log] == ["get_price", "get_price", "compare"]


def test_run_agent_loop_handles_unknown_tool_name_gracefully():
    call = _mock_function_call("nonexistent_tool", {})
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _response_with_calls([call]),
        _response_with_text("알 수 없는 도구였습니다"),
    ]

    text, log = run_agent_loop(client, "gemini-flash-latest", tools={}, user_message="test")

    assert "error" in log[0]["result"]


def test_run_agent_loop_handles_tool_exception_gracefully():
    def failing_tool(**kwargs):
        raise ValueError("API down")

    call = _mock_function_call("failing_tool", {})
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _response_with_calls([call]),
        _response_with_text("도구 실행 실패를 반영한 답변"),
    ]

    text, log = run_agent_loop(client, "gemini-flash-latest", tools={"failing_tool": failing_tool}, user_message="test")

    assert log[0]["result"] == {"error": "API down"}
    assert text == "도구 실행 실패를 반영한 답변"


def test_run_agent_loop_raises_when_max_turns_exceeded():
    call = _mock_function_call("loop_tool", {})
    tool = MagicMock(return_value={"ok": True})

    client = MagicMock()
    client.models.generate_content.return_value = _response_with_calls([call])  # 매번 또 도구를 부름

    with pytest.raises(RuntimeError):
        run_agent_loop(client, "gemini-flash-latest", tools={"loop_tool": tool}, user_message="test", max_turns=3)

    assert client.models.generate_content.call_count == 3

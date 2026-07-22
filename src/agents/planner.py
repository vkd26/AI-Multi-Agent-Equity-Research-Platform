"""LLM Tool-calling 루프를 직접 설계한 구현체.

google-genai SDK는 automatic_function_calling이라는 편의 기능으로 도구 실행을 알아서 처리해준다 —
편하지만, 그러면 "루프를 이해하고 설계"한 게 아니라 SDK 뒤에 숨는 것이므로 일부러 껐다. 대신 다음
과정을 직접 구현한다.

1. LLM에게 사용 가능한 도구 목록 + 지금까지의 대화(contents)를 보낸다.
2. 응답에 function_call이 있으면, 그 이름/인자로 실제 파이썬 함수를 실행한다.
3. 실행 결과를 "함수 응답" 파트로 만들어 대화에 추가한다.
4. 다시 LLM에게 보낸다 — function_call이 더 이상 없을 때까지(=최종 텍스트 응답) 반복한다.
5. 무한루프 방지를 위해 max_turns를 둔다.
"""
from google.genai import types


def run_agent_loop(client, model, tools, user_message, system_instruction=None, max_turns=8):
    """도구 호출 루프를 실행해 최종 응답 텍스트와 호출 로그를 반환한다.

    tools: {"함수이름": 실제 파이썬 콜러블} 딕셔너리. 함수의 타입힌트/독스트링으로 Gemini가 스키마를
    자동 추론한다(google-genai가 raw 함수를 tools로 받으면 지원하는 기능) — 손으로 JSON 스키마를 안
    써도 된다. 대신 인자/반환값은 문자열·숫자·dict·list 같은 단순 타입이어야 스키마 추론이 된다.
    반환: (최종 응답 텍스트, 실행된 tool call 로그 리스트) — 로그는 어떤 근거로 답했는지 추적하는 용도.
    """
    config = types.GenerateContentConfig(
        tools=list(tools.values()),
        system_instruction=system_instruction,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    call_log = []

    for _ in range(max_turns):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        contents.append(response.candidates[0].content)

        function_calls = response.function_calls
        if not function_calls:
            return response.text, call_log

        response_parts = []
        for call in function_calls:
            tool_fn = tools.get(call.name)
            if tool_fn is None:
                result = {"error": f"알 수 없는 도구: {call.name}"}
            else:
                try:
                    result = tool_fn(**call.args)
                except Exception as e:
                    result = {"error": str(e)}
            call_log.append({"tool": call.name, "args": dict(call.args), "result": result})
            response_parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=response_parts))

    raise RuntimeError(f"{max_turns}턴 안에 최종 응답을 받지 못했다 — 무한 루프 가능성이 있다.")

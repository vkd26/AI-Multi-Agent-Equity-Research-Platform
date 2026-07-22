"""CLI 진입점 — 자연어로 질문하면 Tool-calling Agent가 필요한 도구를 스스로 골라 호출해 답한다.

사용법:
    python main.py "TSM 최근 재무비율이랑 뉴스 논조 같이 알려줘"
    python main.py "005930.KS DCF 밸류에이션 해줘" --verbose
"""
import argparse
import sys

from src.agents.research_agent import run_equity_research_agent


def main():
    # Windows 콘솔의 기본 stdout 인코딩이 로케일에 따라 cp949 등으로 잡혀 있으면 한글 출력이 깨지거나
    # 파일로 리다이렉트했을 때 다른 도구에서 mojibake로 보일 수 있다 — UTF-8로 명시적으로 고정한다.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI Equity Research Tool-calling Agent")
    parser.add_argument("query", help="자연어 질문 (예: 'TSM DCF 밸류에이션 해줘')")
    parser.add_argument("--model", default="gemini-flash-latest", help="사용할 Gemini 모델 (기본: gemini-flash-latest)")
    parser.add_argument("--max-turns", type=int, default=8, help="도구 호출 루프 최대 반복 횟수 (기본: 8)")
    parser.add_argument("--verbose", action="store_true", help="호출된 도구 로그도 함께 출력")
    args = parser.parse_args()

    # 도구를 여러 번 체이닝하는 질문은 응답까지 수십 초가 걸릴 수 있는데, 그동안 화면에 아무 표시가
    # 없으면 멈춘 건지 처리 중인 건지 구분이 안 된다 — 시작했다는 것만이라도 즉시 알려준다.
    print("질문 처리 중... (도구 호출 여러 번 거치면 최대 1분 정도 걸릴 수 있습니다)", flush=True)

    answer, call_log = run_equity_research_agent(args.query, model=args.model, max_turns=args.max_turns)
    print("\n" + answer)

    if args.verbose:
        print("\n=== 호출된 도구 ===")
        for entry in call_log:
            print(" -", entry["tool"], entry["args"])


if __name__ == "__main__":
    main()

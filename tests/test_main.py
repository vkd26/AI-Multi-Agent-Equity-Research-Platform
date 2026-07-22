"""main.py CLI 진입점의 인자 파싱/출력을 검증한다 (에이전트 호출은 모킹)."""
from unittest.mock import patch

import main


def test_main_prints_answer_and_calls_agent_with_query(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py", "TSM PER 얼마야?"])
    with patch.object(main, "run_equity_research_agent", return_value=("답변입니다.", [])) as mock_run:
        main.main()

    mock_run.assert_called_once_with("TSM PER 얼마야?", model="gemini-flash-latest", max_turns=8)
    captured = capsys.readouterr()
    assert "답변입니다." in captured.out


def test_main_verbose_prints_call_log(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py", "질문", "--verbose"])
    call_log = [{"tool": "get_stock_price_and_info", "args": {"ticker": "TSM"}}]
    with patch.object(main, "run_equity_research_agent", return_value=("답변", call_log)):
        main.main()

    captured = capsys.readouterr()
    assert "get_stock_price_and_info" in captured.out


def test_main_without_verbose_omits_call_log(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py", "질문"])
    call_log = [{"tool": "get_stock_price_and_info", "args": {"ticker": "TSM"}}]
    with patch.object(main, "run_equity_research_agent", return_value=("답변", call_log)):
        main.main()

    captured = capsys.readouterr()
    assert "get_stock_price_and_info" not in captured.out


def test_main_respects_model_and_max_turns_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "질문", "--model", "gemini-pro", "--max-turns", "3"])
    with patch.object(main, "run_equity_research_agent", return_value=("답변", [])) as mock_run:
        main.main()

    mock_run.assert_called_once_with("질문", model="gemini-pro", max_turns=3)

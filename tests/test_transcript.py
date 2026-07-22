"""transcript.py의 화자/역할 분류, 섹션 경계, 캐싱/에러 처리를 검증한다 (네트워크 미사용, requests.get은 모킹)."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector import transcript

_SAMPLE_HTML = """
<html><body>
<div class="article-body transcript-content">
<h2>CALL PARTICIPANTS</h2>
<ul>
<li>Director of Investor Relations - Jeff Su</li>
<li>Chairman and Chief Executive Officer - C.C. Wei</li>
</ul>
<h2>Full Conference Call Transcript</h2>
<p><strong>Jeff Su:</strong> Good afternoon, welcome to the call.</p>
<p><strong>C.C. Wei:</strong> Thank you, Jeff. Revenue grew <strong>15%</strong> this quarter.</p>
<p>We continue to see strong demand.</p>
<p><strong>Operator:</strong> Please press star one to ask a question.</p>
<p><strong>Jeff Su:</strong> We'll take the first question from Sunny Lin from UBS, please.</p>
<p><strong>Sunny Lin:</strong> Thanks, what's your CapEx outlook?</p>
<p><strong>C.C. Wei:</strong> We expect it to grow.</p>
</div>
</body></html>
"""


def test_download_transcript_parses_speakers_and_carries_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "DATA_DIR_RAW", str(tmp_path))
    mock_response = MagicMock()
    mock_response.text = _SAMPLE_HTML

    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        result = transcript.download_transcript("TSM", url="https://example.com/transcript", use_cache=False)

    assert list(result["speaker"]) == [
        "Jeff Su", "C.C. Wei", "C.C. Wei", "Operator", "Jeff Su", "Sunny Lin", "C.C. Wei",
    ]
    assert result.iloc[0]["text"] == "Good afternoon, welcome to the call."
    # 라벨이 아닌 문단 중간의 <strong>(예: "15%")은 화자로 오인되면 안 되고, 텍스트에도 그대로 남아야 한다
    assert "15%" in result.iloc[1]["text"]
    # 화자 라벨이 없는 문단은 직전 화자를 그대로 이어받는다
    assert result.iloc[2]["speaker"] == "C.C. Wei"


def test_download_transcript_classifies_speaker_roles_and_types(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "DATA_DIR_RAW", str(tmp_path))
    mock_response = MagicMock()
    mock_response.text = _SAMPLE_HTML

    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        result = transcript.download_transcript("TSM", url="https://example.com/transcript", use_cache=False)

    by_speaker = result.set_index("speaker")
    assert by_speaker.loc["Jeff Su", "speaker_type"].iloc[0] == "management"
    assert by_speaker.loc["Jeff Su", "speaker_role"].iloc[0] == "IR"
    assert by_speaker.loc["C.C. Wei", "speaker_role"].iloc[0] == "CEO"
    assert by_speaker.loc["Operator", "speaker_type"] == "operator"
    assert by_speaker.loc["Sunny Lin", "speaker_type"] == "analyst"
    # "Sunny Lin from UBS" 인트로 문구에서 소속을 뽑아야 한다
    assert by_speaker.loc["Sunny Lin", "organization"] == "UBS"


def test_download_transcript_parses_fiscal_quarter_from_url(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "DATA_DIR_RAW", str(tmp_path))
    mock_response = MagicMock()
    mock_response.text = _SAMPLE_HTML
    url = "https://www.fool.com/earnings/call-transcripts/2026/07/16/tsm-tsm-q2-2026-earnings-call-transcript/"

    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        result = transcript.download_transcript("TSM", url=url, use_cache=False)

    assert (result["fiscal_quarter"] == "Q2 2026").all()


def test_download_transcript_section_boundary_is_first_true_analyst_not_operator(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "DATA_DIR_RAW", str(tmp_path))
    mock_response = MagicMock()
    mock_response.text = _SAMPLE_HTML

    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        result = transcript.download_transcript("TSM", url="https://example.com/transcript", use_cache=False)

    # Operator 발언(다이얼인 안내)과 이후 진행자의 소개 멘트까지는 prepared_remarks로 남아야 하고,
    # 실제 첫 애널리스트 질문(Sunny Lin, index 5)부터 qa다
    assert list(result["section"]) == [
        "prepared_remarks", "prepared_remarks", "prepared_remarks", "prepared_remarks", "prepared_remarks",
        "qa", "qa",
    ]


def test_download_transcript_uses_cache_without_calling_network(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "DATA_DIR_RAW", str(tmp_path))
    cache_path = tmp_path / "transcript_TSM.csv"
    pd.DataFrame({"speaker": ["Jeff Su"], "text": ["cached"]}).to_csv(cache_path, index=False)

    with patch.object(transcript, "requests") as mock_requests:
        result = transcript.download_transcript("TSM")

    mock_requests.get.assert_not_called()
    assert result.loc[0, "text"] == "cached"


def test_download_transcript_raises_when_body_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "DATA_DIR_RAW", str(tmp_path))
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>no transcript here</p></body></html>"

    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        with pytest.raises(ValueError):
            transcript.download_transcript("TSM", url="https://example.com/transcript", use_cache=False)


def test_find_transcript_url_matches_ticker_slug():
    mock_response = MagicMock()
    mock_response.text = (
        '<a href="/earnings/call-transcripts/2026/07/16/tsm-tsm-q2-2026-earnings-call-transcript/">a</a>'
        '<a href="/earnings/call-transcripts/2026/07/16/unitedhealth-unh-q2-2026-earnings-call-transcript/">b</a>'
    )
    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        url = transcript.find_transcript_url("TSM")

    assert url == "https://www.fool.com/earnings/call-transcripts/2026/07/16/tsm-tsm-q2-2026-earnings-call-transcript/"


def test_find_transcript_url_raises_when_not_found():
    mock_response = MagicMock()
    mock_response.text = '<a href="/earnings/call-transcripts/2026/07/16/unitedhealth-unh-q2-2026-earnings-call-transcript/">b</a>'
    with patch.object(transcript, "requests") as mock_requests:
        mock_requests.get.return_value = mock_response
        with pytest.raises(ValueError):
            transcript.find_transcript_url("TSM")

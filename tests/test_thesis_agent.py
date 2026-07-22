"""thesis_agent.py의 프롬프트 구성/LLM 응답 파싱을 검증한다 (LLM 클라이언트는 모킹).

실제 생성 품질(근거 기반 여부, target_price가 valuation 범위 안에서 나오는지)은 NVDA 실 데이터로
수동 검증했다(README 참고).
"""
import json
from unittest.mock import MagicMock

from src.agents.thesis_agent import _weighted_target_price, generate_thesis


def test_generate_thesis_parses_json_response():
    expected = {
        "investment_thesis": ["a", "b", "c"],
        "bear_case": ["x", "y", "z"],
        "catalysts": ["cat1"],
        "risks": ["risk1"],
        "target_price": 558.98,
        "target_price_rationale": "comps 하단값 채택",
    }
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps(expected))

    result = generate_thesis("NVIDIA", "NVDA", context={"valuation": {}}, client=client)
    assert result == expected


def test_generate_thesis_includes_company_ticker_and_context_in_prompt():
    context = {"valuation": {"dcf_range": [15, 61]}, "financials": {"roe": 1.14}}
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps({}))

    generate_thesis("NVIDIA", "NVDA", context, client=client)

    call_kwargs = client.models.generate_content.call_args.kwargs
    prompt = call_kwargs["contents"]
    assert "NVIDIA" in prompt and "NVDA" in prompt
    assert "15" in prompt and "61" in prompt  # context가 프롬프트에 실제로 들어갔는지
    assert call_kwargs["config"] == {"response_mime_type": "application/json"}


def test_generate_thesis_prompt_instructs_target_price_must_be_grounded():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps({}))
    generate_thesis("NVIDIA", "NVDA", {"valuation": {}}, client=client)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "임의로 지어내지 마라" in prompt


def test_weighted_target_price_combines_dcf_and_comps_with_60_40_weights():
    valuation = {"dcf_range": [100, 200], "comps_range": [300, 500]}
    # dcf_mid=150, comps_mid=400 -> 0.6*150 + 0.4*400 = 250
    assert _weighted_target_price(valuation) == 250


def test_weighted_target_price_falls_back_to_dcf_only_when_comps_missing():
    assert _weighted_target_price({"dcf_range": [100, 200]}) == 150


def test_weighted_target_price_falls_back_to_comps_only_when_dcf_missing():
    assert _weighted_target_price({"comps_range": [300, 500]}) == 400


def test_weighted_target_price_returns_none_when_both_missing():
    assert _weighted_target_price({}) is None


def test_generate_thesis_injects_weighted_anchor_into_prompt():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps({}))
    context = {"valuation": {"dcf_range": [100, 200], "comps_range": [300, 500]}}

    generate_thesis("NVIDIA", "NVDA", context, client=client)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "weighted_anchor" in prompt
    assert "250" in prompt
    assert "가중평균" in prompt

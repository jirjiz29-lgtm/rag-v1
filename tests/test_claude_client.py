"""Offline tests of ClaudeClient: real Anthropic SDK, mocked HTTP transport."""
import json

import anthropic
import httpx2
import pytest

from backend.prompts import ANSWER_SCHEMA
from backend.rag_pipeline import ClaudeClient, LLMError
from config import Settings


def make_client(handler, **overrides) -> ClaudeClient:
    settings = Settings(llm_provider="anthropic", anthropic_api_key="test-key", **overrides)
    client = ClaudeClient(settings)
    client.client = anthropic.Anthropic(
        api_key="test-key", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return client


def message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 80},
    }


def test_request_shape_and_parsing():
    captured = {}
    payload = {"answer": "20 days [S1]", "found_in_documents": True, "cited_sources": ["S1"], "conflicts": []}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["beta"] = request.headers.get("anthropic-beta", "")
        return httpx2.Response(200, json=message(json.dumps(payload)))

    out = make_client(handler).generate("SYS", [{"role": "user", "content": "q"}], ANSWER_SCHEMA)
    body = captured["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in captured["beta"]
    assert "temperature" not in body
    assert out["answer"] == "20 days [S1]" and out["_usage"]["cache_read_input_tokens"] == 80


def test_fallbacks_can_be_disabled():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=message('{"answer":"","found_in_documents":false,"cited_sources":[],"conflicts":[]}'))

    make_client(handler, claude_enable_fallbacks=False).generate("S", [{"role": "user", "content": "q"}], ANSWER_SCHEMA)
    assert "fallbacks" not in captured["body"]


@pytest.mark.parametrize("status,code", [(401, 503), (429, 429), (500, 502), (404, 503)])
def test_api_errors_map_to_friendly_llm_errors(status, code):
    handler = lambda r: httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": "boom"}})
    with pytest.raises(LLMError) as exc:
        make_client(handler).generate("S", [{"role": "user", "content": "q"}], ANSWER_SCHEMA)
    assert exc.value.status_code == code


def test_refusal_and_truncation_are_reported():
    with pytest.raises(LLMError, match="declined"):
        make_client(lambda r: httpx2.Response(200, json=message("", "refusal"))).generate("S", [{"role": "user", "content": "q"}], ANSWER_SCHEMA)
    with pytest.raises(LLMError, match="cut off"):
        make_client(lambda r: httpx2.Response(200, json=message("{", "max_tokens"))).generate("S", [{"role": "user", "content": "q"}], ANSWER_SCHEMA)


def test_missing_key_is_rejected():
    with pytest.raises(LLMError):
        ClaudeClient(Settings(anthropic_api_key=""))

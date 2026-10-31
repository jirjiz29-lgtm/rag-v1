"""Offline tests of GeminiClient: real google-genai types, stubbed generate_content."""
import json

import pytest
from google.genai import errors, types

from backend.gemini_client import GeminiClient
from backend.prompts import ANSWER_SCHEMA
from backend.rag_pipeline import LLMError
from config import Settings

PAYLOAD = {"answer": "7 days [S1]", "found_in_documents": True, "cited_sources": ["S1"], "conflicts": []}
MESSAGES = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
    {"role": "user", "content": "q"},
]


def response(text: str, finish: str = "STOP", model: str = "gemini-3.8-flash") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20),
        model_version=model,
    )


def api_error(code: int) -> errors.APIError:
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": "boom", "status": "X"}})


def make_client(*outcomes, **overrides):
    """Each outcome is returned (a response) or raised (an exception) by successive calls."""
    settings = Settings(llm_provider="gemini", gemini_api_key="test-key", gemini_model="main-model",
                        gemini_fallback_models=("lite-model",), **overrides)
    client = GeminiClient(settings)
    calls = []
    queue = list(outcomes)

    def fake_generate(model, contents, config):
        calls.append({"model": model, "contents": contents, "config": config})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client.client.models.generate_content = fake_generate
    return client, calls


def test_request_shape_and_parsing():
    client, calls = make_client(response(json.dumps(PAYLOAD)))
    out = client.generate("SYS", MESSAGES, ANSWER_SCHEMA)
    call = calls[0]
    assert call["model"] == "main-model"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["config"].system_instruction == "SYS"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == ANSWER_SCHEMA
    assert out["answer"] == "7 days [S1]"
    assert out["_usage"]["input_tokens"] == 100 and out["_model"] == "gemini-3.8-flash"


@pytest.mark.parametrize("code", [503, 504, 429])
def test_overloaded_model_falls_back(code):
    client, calls = make_client(api_error(code), response(json.dumps(PAYLOAD), model="lite-model"))
    out = client.generate("S", MESSAGES, ANSWER_SCHEMA)
    assert [c["model"] for c in calls] == ["main-model", "lite-model"]
    assert out["_model"] == "lite-model"


@pytest.mark.parametrize("code,status", [(401, 503), (403, 503), (404, 503), (500, 502)])
def test_errors_map_to_friendly_llm_errors(code, status):
    outcomes = [api_error(code)] if code < 500 else [api_error(code), api_error(code)]
    client, calls = make_client(*outcomes)
    with pytest.raises(LLMError) as exc:
        client.generate("S", MESSAGES, ANSWER_SCHEMA)
    assert exc.value.status_code == status
    if code < 500:
        assert len(calls) == 1  # auth / not-found errors are not retried on another model


def test_all_models_overloaded_reports_overload():
    client, _ = make_client(api_error(503), api_error(504))
    with pytest.raises(LLMError, match="overloaded"):
        client.generate("S", MESSAGES, ANSWER_SCHEMA)


def test_blocked_truncated_and_unreadable_responses():
    for finish, text, match in [("SAFETY", "", "declined"), ("MAX_TOKENS", "{", "cut off"), ("STOP", "not json", "unreadable")]:
        client, _ = make_client(response(text, finish))
        with pytest.raises(LLMError, match=match):
            client.generate("S", MESSAGES, ANSWER_SCHEMA)


def test_missing_key_is_rejected():
    with pytest.raises(LLMError):
        GeminiClient(Settings(llm_provider="gemini", gemini_api_key=""))

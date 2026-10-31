"""Gemini implementation of the pipeline's LLMClient interface (Google Gen AI SDK)."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from google import genai
from google.genai import errors, types

from config import Settings
from backend.rag_pipeline import LLMError

logger = logging.getLogger(__name__)

# Finish reasons that mean the model was blocked rather than finishing normally.
_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}
# Overloaded / deadline errors worth retrying on a fallback model.
_RETRY_ON_FALLBACK = {429, 500, 503, 504}


class GeminiClient:
    """Thin wrapper over the Google Gen AI SDK returning schema-validated JSON."""

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not configured on the server.", status_code=503)
        self.settings = settings
        self.model = settings.gemini_model
        self.models = [settings.gemini_model] + [m for m in settings.gemini_fallback_models if m != settings.gemini_model]
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            # One attempt per model: on overload we move to the next model instead of waiting again.
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    @staticmethod
    def _contents(messages: List[Dict[str, Any]]) -> List[types.Content]:
        # The pipeline speaks Anthropic-style roles; Gemini calls the assistant "model".
        return [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=str(m["content"]))])
            for m in messages
        ]

    def generate(self, system: str, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=self.settings.llm_max_tokens,
        )
        contents = self._contents(messages)
        for i, model in enumerate(self.models):
            try:
                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                break
            except errors.APIError as exc:
                if exc.code in _RETRY_ON_FALLBACK and i + 1 < len(self.models):
                    logger.warning("Gemini model %s unavailable (%s); trying %s", model, exc.code, self.models[i + 1])
                    continue
                self._raise(exc, model)
            except Exception as exc:  # network errors surface as httpx exceptions
                if i + 1 < len(self.models):
                    logger.warning("Gemini model %s failed (%s); trying %s", model, type(exc).__name__, self.models[i + 1])
                    continue
                logger.warning("Gemini request failed (%s)", type(exc).__name__)
                raise LLMError("Could not reach the AI service. Check the network connection.", 503) from exc
        return self._parse(response)

    @staticmethod
    def _raise(exc: errors.APIError, model: str) -> None:
        code = exc.code
        if code in (401, 403) or (code == 400 and "API key" in str(exc.message)):
            raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY.", 503) from exc
        if code == 404:
            raise LLMError(f"Model '{model}' was not found. Check GEMINI_MODEL / GEMINI_FALLBACK_MODELS.", 503) from exc
        if code == 429:
            raise LLMError("The AI service is rate-limited right now. Please retry in a moment.", 429) from exc
        if code in (503, 504):
            raise LLMError("The AI service is overloaded or timed out. Please retry in a moment.", 503) from exc
        if code == 400:
            logger.error("Gemini rejected the request: %s", exc.message)
            raise LLMError("The AI service rejected the request. See server logs for details.", 502) from exc
        raise LLMError(f"The AI service returned an error ({code}). Please retry.", 502) from exc

    def _parse(self, response: types.GenerateContentResponse) -> Dict[str, Any]:
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise LLMError("The AI model declined to answer this request.", 422)
        candidate = (response.candidates or [None])[0]
        finish = getattr(getattr(candidate, "finish_reason", None), "name", "") or ""
        if finish in _BLOCKED:
            raise LLMError("The AI model declined to answer this request.", 422)
        if finish == "MAX_TOKENS":
            raise LLMError("The answer was too long and got cut off. Try a narrower question.", 502)

        try:
            data = json.loads(response.text or "")
        except json.JSONDecodeError as exc:
            raise LLMError("The AI service returned an unreadable response. Please retry.", 502) from exc
        usage = response.usage_metadata
        data["_usage"] = {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
        }
        data["_model"] = response.model_version or self.model
        return data

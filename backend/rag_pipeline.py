"""RAG pipeline: question -> retrieval -> LLM (Gemini or Claude) -> grounded, cited answer."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import anthropic

from config import Settings
from backend.citations import claims_for, highlight_sentences, scrub_page_mentions
from backend.prompts import ANSWER_SCHEMA, FALLBACK_MESSAGE, SYSTEM_PROMPT, build_user_message
from backend.retriever import RetrievedChunk, Retriever, process_query

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(S\d+)\]")


class LLMError(Exception):
    """A user-presentable failure talking to the language model."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        ...


class ClaudeClient:
    """Thin wrapper over the official Anthropic SDK returning schema-validated JSON."""

    def __init__(self, settings: Settings):
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not configured on the server.", status_code=503)
        self.settings = settings
        self.model = settings.claude_model
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.claude_timeout_seconds,
            max_retries=2,
        )

    def generate(self, system: str, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.settings.claude_max_tokens,
            # Stable system prompt first so it can be served from the prompt cache.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.settings.claude_effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if self.settings.claude_enable_fallbacks:
            # On a safety-classifier decline, the API retries on Anthropic's recommended fallback model.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            response = self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            hint = self.settings.api_key_warning
            raise LLMError(f"The Anthropic API key was rejected. {hint or 'Check ANTHROPIC_API_KEY.'}", 503) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("The Anthropic API key lacks permission for this model.", 503) from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model '{self.model}' was not found. Check CLAUDE_MODEL.", 503) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The AI service is rate-limited right now. Please retry in a moment.", 429) from exc
        except anthropic.BadRequestError as exc:
            logger.error("Claude rejected the request: %s", exc.message)
            raise LLMError("The AI service rejected the request. See server logs for details.", 502) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The AI service returned an error ({exc.status_code}). Please retry.", 502) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the AI service. Check the network connection.", 503) from exc

        if response.stop_reason == "refusal":
            raise LLMError("The AI model declined to answer this request.", 422)
        if response.stop_reason == "max_tokens":
            raise LLMError("The answer was too long and got cut off. Try a narrower question.", 502)

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("The AI service returned an unreadable response. Please retry.", 502) from exc
        data["_usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        data["_model"] = response.model
        return data


@dataclass
class RAGAnswer:
    answer: str
    found: bool
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "found": self.found,
            "sources": self.sources,
            "retrieved": self.retrieved,
            "conflicts": self.conflicts,
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


def _clean_history(history: Optional[Sequence[Dict[str, str]]], max_turns: int) -> List[Dict[str, str]]:
    """Keep the last N text turns in strict user/assistant alternation, starting with a user turn."""
    turns = [
        {"role": m["role"], "content": str(m["content"])[:4000]}
        for m in (history or [])
        if m.get("role") in {"user", "assistant"} and str(m.get("content", "")).strip()
    ][-2 * max_turns :]
    cleaned: List[Dict[str, str]] = []
    for m in turns:
        if cleaned and cleaned[-1]["role"] == m["role"]:
            cleaned[-1] = m
        else:
            cleaned.append(m)
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    if cleaned and cleaned[-1]["role"] == "user":
        cleaned.pop()
    return cleaned


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: Optional[LLMClient], settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def answer(
        self,
        question: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        categories: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> RAGAnswer:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        query = process_query(question, history)
        chunks = self.retriever.retrieve(
            query, top_k=top_k, relevance_threshold=relevance_threshold, categories=categories, doc_ids=doc_ids
        )
        retrieved = [c.to_dict() for c in chunks]
        if not chunks:
            # Nothing relevant enough - answer with the fallback without calling the LLM.
            return RAGAnswer(FALLBACK_MESSAGE, False, retrieved=retrieved,
                             latency_ms=int((time.perf_counter() - started) * 1000))

        if self.llm is None:
            raise LLMError(f"{self.settings.llm_key_name} is not configured on the server.", status_code=503)

        messages = _clean_history(history, self.settings.history_turns)
        messages.append({"role": "user", "content": build_user_message(question, chunks)})
        data = self.llm.generate(SYSTEM_PROMPT, messages, ANSWER_SCHEMA)
        result = self._ground(data, chunks)
        result.retrieved = retrieved
        result.model = data.get("_model", getattr(self.llm, "model", None))
        result.usage = data.get("_usage", {})
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    @staticmethod
    def _ground(data: Dict[str, Any], chunks: List[RetrievedChunk]) -> RAGAnswer:
        """Validate the model's citations against what was actually retrieved; never surface fake ones."""
        by_id = {c.source_id: c for c in chunks}
        answer = str(data.get("answer", "")).strip()
        found = bool(data.get("found_in_documents")) and bool(answer)

        cited = [s for s in data.get("cited_sources", []) if s in by_id]
        cited += [s for s in _CITATION.findall(answer) if s in by_id and s not in cited]
        # Remove inline markers that point at sources we never provided.
        answer = _CITATION.sub(lambda m: m.group(0) if m.group(1) in by_id else "", answer)

        if not found or not cited or answer == FALLBACK_MESSAGE:
            return RAGAnswer(FALLBACK_MESSAGE, False)

        conflicts = []
        for conflict in data.get("conflicts", []) or []:
            ids = [s for s in conflict.get("sources", []) if s in by_id]
            if conflict.get("description") and ids:
                conflicts.append({
                    "description": conflict["description"],
                    "sources": ids,
                    "documents": sorted({by_id[s].document for s in ids}),
                })
        # Strip "(page N)" style references in the prose that no retrieved chunk supports.
        answer = scrub_page_mentions(answer, {c.page for c in chunks if c.page})

        ordered = sorted(set(cited), key=lambda s: int(s[1:]))
        sources = []
        for number, sid in enumerate(ordered, start=1):
            source = by_id[sid].to_dict()
            source["number"] = number  # display label: [S3] in the answer is shown as [1], [2]...
            source["highlights"] = highlight_sentences(by_id[sid].text, claims_for(answer, sid), heading=by_id[sid].section)
            sources.append(source)
        return RAGAnswer(answer, True, sources=sources, conflicts=conflicts)

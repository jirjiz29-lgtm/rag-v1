"""Suggested starter questions derived from the documents actually in the knowledge base.

The LLM proposes questions from each document's title and section headings; every proposal is then
checked against the real retriever and kept only if its source document comes back near the top, so
a suggestion never points at something the knowledge base can't answer. Results are cached until the
set of indexed documents (or the index settings) changes. Without an LLM, questions are built from
section headings instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from backend.hybrid import contextual_text, tokenize
from backend.knowledge_base import KnowledgeBaseManager
from backend.rag_pipeline import LLMClient, LLMError
from backend.retriever import Retriever

logger = logging.getLogger(__name__)

SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "document": {"type": "string", "description": "File name of the document that answers it."},
                },
                "required": ["question", "document"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write 12 starter questions for an internal company knowledge assistant. You are given each document's "
    "file name, title, category and section headings. Write short, natural questions an employee would ask "
    "(6-14 words) that the listed sections clearly answer. Cover different documents. Mix kinds: a how-to, a "
    "rule or policy check, a summary request, and one question that compares two documents. Never ask about "
    "facts a section says are not yet defined. Return the file name of the main document for each question."
)

_GENERIC_SECTIONS = {"objective", "purpose", "scope", "overview", "general"}
# Words that say nothing about *what* is asked; a suggestion must share specific words with its passage.
_GENERIC_WORDS = set(tokenize(
    "what which how who when where why does do say says said about describe described explain company our we team "
    "policy policies procedure procedures process step steps rule rules guideline guidelines document documents "
    "follow required require standard official main key core employee employees staff according between differ "
    "compare compared new use used handle handled include included"
))


class SuggestionService:
    def __init__(self, knowledge_base: KnowledgeBaseManager, retriever: Retriever, llm: Optional[LLMClient],
                 cache_path: Path, index_signature: str):
        self.kb = knowledge_base
        self.retriever = retriever
        self.llm = llm
        self.cache_path = cache_path
        self.index_signature = index_signature
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def get(self, limit: int = 6) -> Dict:
        docs = [d for d in self.kb.list_documents() if d.get("status") == "indexed"]
        if not docs:
            return {"questions": [], "source": "empty"}
        fingerprint = self._fingerprint(docs)
        with self._lock:
            cached = self._read_cache()
            if cached.get("fingerprint") == fingerprint and cached.get("questions"):
                return {"questions": cached["questions"][:limit], "source": cached.get("source", "cache")}
            questions, source = self._generate(docs)
            self._write_cache({"fingerprint": fingerprint, "source": source, "questions": questions})
        return {"questions": questions[:limit], "source": source}

    # ------------------------------------------------------------------ generation
    def _outline(self, docs: List[Dict]) -> List[Dict]:
        outline = []
        for d in docs:
            chunks = self.kb.vector_db.get_document_chunks(d["doc_id"])
            sections = list(dict.fromkeys(c["metadata"].get("section", "") for c in chunks if c["metadata"].get("section")))
            title = next((c["metadata"].get("title") for c in chunks if c["metadata"].get("title")), d["filename"])
            outline.append({"document": d["filename"], "title": title, "category": d.get("category", "General"),
                            "sections": sections[:15]})
        return outline

    def _generate(self, docs: List[Dict]):
        outline = self._outline(docs)
        if self.llm is not None:
            try:
                data = self.llm.generate(SYSTEM, [{"role": "user", "content": json.dumps({"documents": outline})}],
                                         SUGGESTION_SCHEMA)
                verified = self._verify(data.get("questions", []), outline)
                if len(verified) >= 3:
                    # Top up from headings for documents the model's questions didn't cover.
                    covered = {q["document"] for q in verified}
                    extra = [q for q in self._verify(self._from_headings(outline), outline) if q["document"] not in covered]
                    return verified + extra, "llm"
            except (LLMError, Exception) as exc:  # any failure falls back to headings; never break the page
                logger.info("Suggestion generation fell back to headings (%s)", type(exc).__name__)
        return self._verify(self._from_headings(outline), outline), "headings"

    @staticmethod
    def _from_headings(outline: List[Dict]) -> List[Dict]:
        """Deterministic fallback: one question per document from its most specific section heading."""
        out = []
        for doc in outline:
            topic = next((s for s in doc["sections"]
                          if re.sub(r"^[\d.\s]+", "", s.split(" > ")[-1]).strip().lower() not in _GENERIC_SECTIONS
                          and s != doc["title"]), None)
            if not topic:
                continue
            topic = re.sub(r"^[\d.\s]+", "", topic.split(" > ")[-1]).strip()
            short = re.sub(r"^.*?[–-]\s*", "", doc["title"]).strip() or doc["title"]
            out.append({"question": f"What does the {short} say about {topic.lower()}?", "document": doc["document"]})
        return out

    def _verify(self, proposals: List[Dict], outline: List[Dict]) -> List[Dict]:
        """Keep well-formed questions whose named document is among the top retrieved sources."""
        known = {d["document"]: d for d in outline}
        kept, seen_docs, seen_text = [], {}, set()
        for p in proposals:
            question = re.sub(r"\s+", " ", str(p.get("question", ""))).strip()
            document = str(p.get("document", "")).strip()
            key = " ".join(sorted(set(tokenize(question))))
            if document not in known or not 15 <= len(question) <= 160 or key in seen_text:
                continue
            top = [c for c in self.retriever.retrieve(question)[:3] if c.document == document]
            if not top or not self._answerable(question, top[0]):
                continue
            if seen_docs.get(document, 0) >= 2:  # keep the set varied
                continue
            seen_docs[document] = seen_docs.get(document, 0) + 1
            seen_text.add(key)
            kept.append({"question": question if question.endswith("?") else question.rstrip(".") + "?",
                         "document": document, "category": known[document]["category"]})
        return kept

    @staticmethod
    def _answerable(question: str, chunk) -> bool:
        """The passage must contain the question's specific words, not just share its topic.

        "What is the parking policy for visitors?" retrieves the employee handbook (closest topic), but
        neither "parking" nor "visitor" appears in it, so the suggestion is dropped.
        """
        specific = set(tokenize(question)) - _GENERIC_WORDS
        if not specific:
            return False
        passage = set(tokenize(contextual_text(chunk.text, chunk.metadata.get("title", ""), chunk.section or "")))
        return len(specific & passage) >= min(2, len(specific)) and len(specific & passage) / len(specific) >= 0.5

    # ------------------------------------------------------------------ cache
    def _fingerprint(self, docs: List[Dict]) -> str:
        raw = "|".join(sorted(f"{d['doc_id']}:{d.get('sha256', '')}" for d in docs)) + "|" + self.index_signature
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _read_cache(self) -> Dict:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_cache(self, data: Dict) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            logger.warning("Could not write %s", self.cache_path.name)

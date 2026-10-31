"""Semantic retrieval with metadata filtering and relevance thresholding."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.citations import location_label, page_note, valid_page, valid_section
from backend.vector_database import VectorDatabase


@dataclass
class RetrievedChunk:
    source_id: str  # "S1", "S2", ... - the label the LLM cites
    chunk_id: str
    doc_id: str
    document: str
    text: str
    score: float
    page: Optional[int] = None
    section: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_type(self) -> str:
        return str(self.metadata.get("file_type", "")).lower()

    @property
    def total_pages(self) -> Optional[int]:
        total = self.metadata.get("total_pages")
        return total if isinstance(total, int) and self.page else None

    @property
    def citation(self) -> str:
        return location_label(self.document, self.page, self.total_pages, self.section)

    def to_dict(self) -> Dict[str, Any]:
        """Citation record. Every field is copied from the chunk's stored metadata - none from the model."""
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "document": self.document,
            "file_type": self.file_type,
            "page": self.page,
            "total_pages": self.total_pages,
            "page_note": page_note(self.file_type, self.page),
            "section": self.section,
            "category": self.category,
            "chunk_index": self.metadata.get("chunk_index"),
            "chunk_total": self.metadata.get("chunk_total"),
            "uploaded_at": self.metadata.get("uploaded_at"),
            "citation": self.citation,
            "score": round(self.score, 4),
            "excerpt": self.text,
        }


def process_query(question: str, history: Optional[Sequence[Dict[str, str]]] = None) -> str:
    """Normalise the question and add context for short follow-ups ("what about contractors?")."""
    query = re.sub(r"\s+", " ", question or "").strip()
    if history and len(query.split()) <= 6:
        previous = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
        if previous:
            query = f"{previous[-1]} {query}"
    return query


def build_filter(categories: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None) -> Optional[Dict]:
    clauses = []
    if categories:
        clauses.append({"category": {"$in": list(categories)}})
    if doc_ids:
        clauses.append({"doc_id": {"$in": list(doc_ids)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Retriever:
    def __init__(self, vector_db: VectorDatabase, top_k: int = 6, relevance_threshold: float = 0.25):
        self.vector_db = vector_db
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        categories: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        if not query.strip():
            return []
        k = top_k or self.top_k
        threshold = self.relevance_threshold if relevance_threshold is None else relevance_threshold
        results = self.vector_db.search(query, k=k, where=build_filter(categories, doc_ids))
        chunks: List[RetrievedChunk] = []
        for doc, score in results:
            if score < threshold:
                continue
            meta = doc.metadata
            chunks.append(
                RetrievedChunk(
                    source_id=f"S{len(chunks) + 1}",
                    chunk_id=meta.get("chunk_id", ""),
                    doc_id=meta.get("doc_id", ""),
                    document=meta.get("source", "Unknown document"),
                    text=doc.page_content,
                    score=score,
                    page=valid_page(meta),
                    section=valid_section(meta),
                    category=meta.get("category"),
                    metadata=meta,
                )
            )
        return chunks

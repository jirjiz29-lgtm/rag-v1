"""Hybrid retrieval (semantic + keyword), cross-encoder reranking, metadata filtering and thresholds."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.citations import location_label, page_note, valid_page, valid_section
from backend.hybrid import contextual_text, rrf_fuse  # noqa: F401  (contextual_text re-exported)
from backend.reranker import CrossEncoderReranker
from backend.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


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
    rerank_score: Optional[float] = None  # cross-encoder relevance (0-1), when reranking is on
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
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "excerpt": self.text,
        }


def select_chunks(scored: Sequence[tuple], top_k: int, threshold: Optional[float] = None,
                  relative_cutoff: Optional[float] = None, per_doc_cap: Optional[int] = None,
                  must_include: Sequence[str] = ()) -> List[str]:
    """Choose the final context from (id, score, doc) candidates.

    Drops candidates below `threshold` or below `relative_cutoff` x best score, always keeps the ids
    in `must_include` (e.g. the best match for each part of a multi-part question), caps chunks per
    document so one long document can't crowd out another, and returns at most `top_k` ids by score.
    """
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    if not ranked:
        return []
    best = ranked[0][1]
    keep = [(cid, s, doc) for cid, s, doc in ranked
            if (threshold is None or s >= threshold) and (relative_cutoff is None or s >= relative_cutoff * best)]
    chosen: List[str] = []
    per_doc: Dict[str, int] = {}
    by_id = {cid: (cid, s, doc) for cid, s, doc in keep}
    for cid in list(dict.fromkeys(c for c in must_include if c in by_id)) + [c for c, _, _ in keep]:
        if cid in chosen or len(chosen) >= top_k:
            continue
        doc = by_id[cid][2]
        if per_doc_cap is not None and per_doc.get(doc, 0) >= per_doc_cap and cid not in must_include:
            continue
        chosen.append(cid)
        per_doc[doc] = per_doc.get(doc, 0) + 1
    return sorted(chosen, key=lambda c: by_id[c][1], reverse=True)


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
    """Hybrid retrieval with reranking.

    1. Semantic search (embeddings) and BM25 keyword search each return `candidate_pool` chunks;
       reciprocal-rank fusion merges them.
    2. A cross-encoder reranks the pool by how well each chunk answers the question.
    3. Chunks scoring below `keep_ratio` x the best score are dropped; at most `top_k` are kept.
    If no chunk reaches `relevance_threshold` cosine similarity, nothing is returned and the
    pipeline answers with the fallback message without calling the LLM.
    """

    def __init__(self, vector_db: VectorDatabase, top_k: int = 6, relevance_threshold: float = 0.2, *,
                 hybrid: bool = True, candidate_pool: int = 20, reranker: Optional[CrossEncoderReranker] = None,
                 keep_ratio: float = 0.001):
        self.vector_db = vector_db
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.hybrid = hybrid
        self.candidate_pool = candidate_pool
        self.reranker = reranker
        self.keep_ratio = keep_ratio

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
        pool = max(self.candidate_pool, k)

        dense = self.vector_db.search(query, k=pool, where=build_filter(categories, doc_ids))
        if not dense or max(score for _, score in dense) < threshold:
            return []  # nothing in the knowledge base is close to this question
        docs = {d.metadata["chunk_id"]: d for d, _ in dense}
        cosine = {d.metadata["chunk_id"]: score for d, score in dense}
        order = list(cosine)
        if self.hybrid:
            keyword = self.vector_db.keyword_search(query, pool, categories, doc_ids)
            for d, _ in keyword:
                docs.setdefault(d.metadata["chunk_id"], d)
            order = [cid for cid, _ in rrf_fuse([order, [d.metadata["chunk_id"] for d, _ in keyword]])][:pool]
            missing = [cid for cid in order if cid not in cosine]
            cosine.update(self.vector_db.cosine_scores(query, missing))

        rerank: Dict[str, float] = {}
        chosen = order[:k]
        if self.reranker is not None:
            try:
                rerank = dict(zip(order, self.reranker.score(query, [docs[c].page_content for c in order])))
                chosen = select_chunks([(c, rerank[c], docs[c].metadata.get("doc_id", "")) for c in order],
                                       top_k=k, relative_cutoff=self.keep_ratio or None)
            except Exception as exc:  # model unavailable (e.g. offline): keep the hybrid order
                logger.warning("Reranker failed (%s); using hybrid ranking", type(exc).__name__)
                rerank, chosen = {}, order[:k]

        chunks: List[RetrievedChunk] = []
        for cid in chosen:
            meta = docs[cid].metadata
            chunks.append(
                RetrievedChunk(
                    source_id=f"S{len(chunks) + 1}",
                    chunk_id=meta.get("chunk_id", ""),
                    doc_id=meta.get("doc_id", ""),
                    document=meta.get("source", "Unknown document"),
                    text=docs[cid].page_content,
                    score=cosine.get(cid, 0.0),
                    rerank_score=rerank.get(cid),
                    page=valid_page(meta),
                    section=valid_section(meta),
                    category=meta.get("category"),
                    metadata=meta,
                )
            )
        return chunks

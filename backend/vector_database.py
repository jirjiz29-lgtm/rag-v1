"""Persistent ChromaDB vector store (via langchain-chroma)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.hybrid import BM25Index, contextual_text

_ALLOWED_META_TYPES = (str, int, float, bool)


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma only accepts scalar metadata values; drop None and stringify the rest."""
    clean = {}
    for key, value in meta.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, _ALLOWED_META_TYPES) else str(value)
    return clean


class VectorDatabase:
    def __init__(self, persist_dir: Path, collection_name: str, embeddings: Embeddings):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embeddings = embeddings
        self._version = 0  # bumped on every change; invalidates the keyword index
        self._keyword: Optional[Tuple[int, BM25Index, Dict[str, Document]]] = None
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self.store._collection

    def add_chunks(self, chunks: List[Document], batch_size: int = 128) -> int:
        if not chunks:
            return 0
        with self._lock:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                for c in batch:
                    c.metadata = _sanitize_metadata(c.metadata)
                self.store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
            self._version += 1
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.collection.delete(where={"doc_id": doc_id})
            self._version += 1

    def count(self) -> int:
        return self.collection.count()

    def count_for_document(self, doc_id: str) -> int:
        return len(self.collection.get(where={"doc_id": doc_id}, include=[])["ids"])

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        result = self.collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        chunks = [
            {"id": cid, "text": text, "metadata": meta}
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]
        return sorted(chunks, key=lambda c: c["metadata"].get("chunk_index", 0))

    def _keyword_index(self) -> Tuple[BM25Index, Dict[str, Document]]:
        """BM25 over every chunk, built from the store on first use and rebuilt after any change.

        Chunks are indexed with their document title and section heading in front, so a question
        that names a document ("the vendor SOP") or topic matches its chunks by keyword.
        """
        with self._lock:
            if self._keyword is None or self._keyword[0] != self._version:
                data = self.collection.get(include=["documents", "metadatas"])
                docs = {cid: Document(page_content=text, metadata=meta)
                        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])}
                index = BM25Index(list(docs), [contextual_text(d.page_content, d.metadata.get("title", ""),
                                                               d.metadata.get("section", "")) for d in docs.values()])
                self._keyword = (self._version, index, docs)
            return self._keyword[1], self._keyword[2]

    def keyword_search(self, query: str, k: int, categories: Optional[List[str]] = None,
                       doc_ids: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
        index, docs = self._keyword_index()
        allowed = None
        if categories or doc_ids:
            allowed = [cid for cid, d in docs.items()
                       if (not categories or d.metadata.get("category") in categories)
                       and (not doc_ids or d.metadata.get("doc_id") in doc_ids)]
        return [(docs[cid], score) for cid, score in index.search(query, k, allowed)]

    def cosine_scores(self, query: str, chunk_ids: List[str]) -> Dict[str, float]:
        """Cosine similarity of the query to specific chunks (for keyword-only hits)."""
        if not chunk_ids:
            return {}
        data = self.collection.get(ids=list(chunk_ids), include=["embeddings"])
        q = np.array(self._embeddings.embed_query(query))
        return {cid: float(np.clip(np.dot(np.array(e), q), 0.0, 1.0)) for cid, e in zip(data["ids"], data["embeddings"])}

    def search(self, query: str, k: int, where: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Return (chunk, relevance) pairs; relevance = 1 - cosine distance, in [0, 1] for normalised vectors."""
        if self.count() == 0:
            return []
        k = min(k, self.count())
        results = self.store.similarity_search_with_score(query, k=k, filter=where)
        return [(doc, max(0.0, min(1.0, 1.0 - float(distance)))) for doc, distance in results]

"""Persistent ChromaDB vector store (via langchain-chroma)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.collection.delete(where={"doc_id": doc_id})

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

    def search(self, query: str, k: int, where: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Return (chunk, relevance) pairs; relevance = 1 - cosine distance, in [0, 1] for normalised vectors."""
        if self.count() == 0:
            return []
        k = min(k, self.count())
        results = self.store.similarity_search_with_score(query, k=k, filter=where)
        return [(doc, max(0.0, min(1.0, 1.0 - float(distance)))) for doc, distance in results]

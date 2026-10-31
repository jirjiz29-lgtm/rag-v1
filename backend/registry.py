"""JSON-backed registry of uploaded documents (status, hash, chunk counts, dates)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._docs: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._docs = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._docs = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, indent=2)
        os.replace(tmp, self.path)

    def get(self, doc_id: str) -> Optional[dict]:
        with self._lock:
            doc = self._docs.get(doc_id)
            return dict(doc) if doc else None

    def find_by_hash(self, sha256: str) -> Optional[dict]:
        with self._lock:
            for doc in self._docs.values():
                if doc.get("sha256") == sha256:
                    return dict(doc)
        return None

    def upsert(self, doc_id: str, **fields) -> dict:
        with self._lock:
            doc = self._docs.setdefault(doc_id, {"doc_id": doc_id})
            doc.update(fields)
            self._save()
            return dict(doc)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            self._save()

    def all(self) -> List[dict]:
        with self._lock:
            return sorted((dict(d) for d in self._docs.values()), key=lambda d: d.get("uploaded_at", ""), reverse=True)

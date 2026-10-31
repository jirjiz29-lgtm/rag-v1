"""Knowledge base management: ingest, deduplicate, update, delete and re-index documents."""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config import SUPPORTED_EXTENSIONS, Settings
from backend.document_loader import DocumentLoadError, load_document
from backend.registry import DocumentRegistry, utc_now
from backend.text_splitter import chunk_documents
from backend.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised for uploads rejected before processing (bad type, too large, empty)."""


def safe_filename(name: str) -> str:
    """Strip directories and unsafe characters so uploads can't escape the knowledge base folder."""
    name = Path(name or "").name.strip()
    name = re.sub(r"[^A-Za-z0-9._ ()-]", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name or name.startswith("."):
        raise ValidationError("Invalid file name")
    return name[:150]


def make_doc_id(filename: str) -> str:
    return hashlib.sha1(filename.lower().encode("utf-8")).hexdigest()[:16]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class KnowledgeBaseManager:
    def __init__(self, settings: Settings, vector_db: VectorDatabase, registry: DocumentRegistry):
        self.settings = settings
        self.vector_db = vector_db
        self.registry = registry
        self.kb_dir = settings.knowledge_base_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ validation
    def validate_upload(self, filename: str, data: bytes) -> str:
        name = safe_filename(filename)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file type '{ext or 'none'}'. Upload PDF, DOCX, TXT or MD files."
            )
        if not data:
            raise ValidationError(f"{name} is empty.")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise ValidationError(f"{name} exceeds the {self.settings.max_upload_mb} MB upload limit.")
        if ext == ".pdf" and not data.startswith(b"%PDF"):
            raise ValidationError(f"{name} is not a valid PDF file.")
        if ext == ".docx" and not data.startswith(b"PK"):
            raise ValidationError(f"{name} is not a valid DOCX file.")
        return name

    # ------------------------------------------------------------------ ingestion
    def ingest_upload(self, filename: str, data: bytes, category: str = "General") -> Dict:
        """Validate, store and index an uploaded file. Returns a result dict with a `status` key."""
        name = self.validate_upload(filename, data)
        digest = sha256_bytes(data)
        with self._lock:
            duplicate = self._find_indexed_duplicate(digest)
            if duplicate:
                return {
                    "status": "duplicate",
                    "filename": name,
                    "doc_id": duplicate["doc_id"],
                    "message": f"Identical content is already indexed as '{duplicate['filename']}'. Skipped.",
                }
            target = self.kb_dir / name
            fd, tmp = tempfile.mkstemp(dir=self.kb_dir, suffix=".upload")
            with open(fd, "wb") as f:
                f.write(data)
            shutil.move(tmp, target)
            return self._index_file(target, digest, category)

    def ingest_path(self, path: Path, category: str = "General") -> Dict:
        """Index a file that already lives inside the knowledge base folder."""
        path = Path(path)
        data = path.read_bytes()
        self.validate_upload(path.name, data)
        digest = sha256_bytes(data)
        with self._lock:
            duplicate = self._find_indexed_duplicate(digest)
            if duplicate:
                return {"status": "duplicate", "filename": path.name, "doc_id": duplicate["doc_id"],
                        "message": f"Already indexed as '{duplicate['filename']}'."}
            if path.parent.resolve() != self.kb_dir.resolve():
                target = self.kb_dir / safe_filename(path.name)
                shutil.copy2(path, target)
                path = target
            return self._index_file(path, digest, category)

    def _find_indexed_duplicate(self, digest: str) -> Optional[dict]:
        dup = self.registry.find_by_hash(digest)
        if dup and dup.get("status") == "indexed" and self.vector_db.count_for_document(dup["doc_id"]) > 0:
            return dup
        return None

    def _index_file(self, path: Path, digest: str, category: str) -> Dict:
        doc_id = make_doc_id(path.name)
        existing = self.registry.get(doc_id)
        is_update = bool(existing and existing.get("status") in {"indexed", "reindexing"})
        now = utc_now()
        self.registry.upsert(
            doc_id,
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            category=category,
            size_bytes=path.stat().st_size,
            sha256=digest,
            status="processing",
            error=None,
            uploaded_at=(existing or {}).get("uploaded_at", now),
            updated_at=now,
        )
        try:
            docs = load_document(path, {"category": category})
            chunks = chunk_documents(docs, doc_id, self.settings.chunk_size, self.settings.chunk_overlap)
            if not chunks:
                raise DocumentLoadError("Document produced no indexable chunks.")
            for c in chunks:
                c.metadata["uploaded_at"] = (existing or {}).get("uploaded_at", now)
            # Replace any previous version of this document atomically from the caller's view.
            self.vector_db.delete_document(doc_id)
            self.vector_db.add_chunks(chunks)
        except DocumentLoadError as exc:
            if is_update:
                # Keep serving the previous good version; just record the failure.
                self.registry.upsert(doc_id, status="indexed", error=f"Last update failed: {exc}")
            else:
                self.registry.upsert(doc_id, status="failed", error=str(exc), chunks=0)
            logger.info("Indexing failed for %s: %s", path.name, exc)
            return {"status": "failed", "filename": path.name, "doc_id": doc_id, "message": str(exc)}

        pages = {c.metadata.get("page") for c in chunks if c.metadata.get("page")}
        sections = {c.metadata.get("section") for c in chunks}
        record = self.registry.upsert(
            doc_id,
            status="indexed",
            chunks=len(chunks),
            pages=max(pages) if pages else None,
            sections=len(sections),
            version=(existing or {}).get("version", 0) + 1,
            indexed_at=utc_now(),
        )
        status = "updated" if is_update else "indexed"
        logger.info("%s %s (%d chunks)", status.capitalize(), path.name, len(chunks))
        return {
            "status": status,
            "filename": path.name,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "message": f"{'Updated' if is_update else 'Indexed'} {path.name} into {len(chunks)} chunks.",
            "document": record,
        }

    # ------------------------------------------------------------------ management
    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            record = self.registry.get(doc_id)
            if not record:
                return False
            self.vector_db.delete_document(doc_id)
            file_path = self.kb_dir / record["filename"]
            if file_path.exists():
                file_path.unlink()
            self.registry.remove(doc_id)
            return True

    def reindex_document(self, doc_id: str) -> Dict:
        with self._lock:
            record = self.registry.get(doc_id)
            if not record:
                raise KeyError(doc_id)
            path = self.kb_dir / record["filename"]
            if not path.exists():
                self.registry.upsert(doc_id, status="failed", error="Source file is missing; please re-upload.")
                return {"status": "failed", "doc_id": doc_id, "filename": record["filename"],
                        "message": "Source file is missing; please re-upload."}
            digest = sha256_bytes(path.read_bytes())
            # Force a re-index even if the content hash is unchanged (e.g. chunk settings changed).
            self.registry.upsert(doc_id, status="reindexing")
            result = self._index_file(path, digest, record.get("category", "General"))
            if result["status"] == "updated":
                result["status"] = "reindexed"
            return result

    def reindex_all(self) -> List[Dict]:
        return [self.reindex_document(d["doc_id"]) for d in self.registry.all()]

    def sync_folder(self) -> List[Dict]:
        """Index new/changed files dropped into the knowledge_base folder; prune entries whose file is gone."""
        results = []
        with self._lock:
            on_disk = {p.name: p for p in self.kb_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS}
            for record in self.registry.all():
                if record.get("status") in {"indexed", "processing", "reindexing"} and record["filename"] not in on_disk:
                    self.vector_db.delete_document(record["doc_id"])
                    self.registry.remove(record["doc_id"])
                    results.append({"status": "removed", "filename": record["filename"], "doc_id": record["doc_id"]})
            for name, path in sorted(on_disk.items()):
                record = self.registry.get(make_doc_id(name))
                try:
                    data = path.read_bytes()
                    if record and record.get("sha256") == sha256_bytes(data) and record.get("status") == "indexed":
                        continue
                    category = (record or {}).get("category", "General")
                    results.append(self.ingest_path(path, category))
                except ValidationError as exc:
                    results.append({"status": "failed", "filename": name, "message": str(exc)})
        return results

    def list_documents(self) -> List[dict]:
        return self.registry.all()

    def get_document(self, doc_id: str) -> Optional[dict]:
        return self.registry.get(doc_id)

    def stats(self) -> Dict:
        docs = self.registry.all()
        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for d in docs:
            by_status[d.get("status", "unknown")] = by_status.get(d.get("status", "unknown"), 0) + 1
            if d.get("status") == "indexed":
                by_category[d.get("category", "General")] = by_category.get(d.get("category", "General"), 0) + 1
        return {
            "total_documents": len(docs),
            "indexed_documents": by_status.get("indexed", 0),
            "failed_documents": by_status.get("failed", 0),
            "total_chunks": self.vector_db.count(),
            "by_status": by_status,
            "by_category": by_category,
            "last_updated": max((d.get("updated_at", "") for d in docs), default=None),
        }

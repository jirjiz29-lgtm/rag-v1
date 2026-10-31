"""FastAPI backend for the Company Knowledge Assistant.

Run with:  uvicorn main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import DOCUMENT_CATEGORIES, SUPPORTED_EXTENSIONS, Settings, get_settings
from backend.knowledge_base import ValidationError
from backend.rag_pipeline import LLMClient, LLMError
from backend.services import Services, build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("knowledge_api")


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: List[ChatTurn] = Field(default_factory=list)
    categories: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    relevance_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def create_app(settings: Optional[Settings] = None, llm: Optional[LLMClient] = None, sync_on_startup: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = build_services(settings, llm=llm)
        app.state.services = services
        if sync_on_startup:
            results = services.knowledge_base.sync_folder()
            if results:
                logger.info("Startup sync: %s", {r["filename"]: r["status"] for r in results})
        yield

    app = FastAPI(title="Company Knowledge Assistant API", version="1.0.0", lifespan=lifespan)

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error. Check the server logs."})

    @app.get("/health")
    def health(request: Request) -> Dict:
        s = svc(request)
        return {
            "status": "ok",
            "llm_configured": s.pipeline.llm is not None,
            "api_key_warning": settings.api_key_warning,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "documents": len(s.knowledge_base.list_documents()),
            "chunks": s.vector_db.count(),
        }

    @app.get("/settings")
    def get_public_settings() -> Dict:
        # Never include secrets here.
        return {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "effort": settings.claude_effort,
            "fallbacks_enabled": settings.claude_enable_fallbacks,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "relevance_threshold": settings.relevance_threshold,
            "max_upload_mb": settings.max_upload_mb,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "categories": DOCUMENT_CATEGORIES,
            "llm_configured": settings.llm_configured,
            "api_key_warning": settings.api_key_warning,
        }

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request) -> Dict:
        try:
            result = svc(request).pipeline.answer(
                body.question,
                history=[t.model_dump() for t in body.history],
                categories=body.categories,
                doc_ids=body.doc_ids,
                top_k=body.top_k,
                relevance_threshold=body.relevance_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return result.to_dict()

    @app.get("/documents")
    def list_documents(request: Request) -> List[Dict]:
        return svc(request).knowledge_base.list_documents()

    @app.get("/stats")
    def stats(request: Request) -> Dict:
        return svc(request).knowledge_base.stats()

    @app.post("/documents/upload")
    async def upload_documents(
        request: Request,
        files: List[UploadFile] = File(...),
        category: str = Form("General"),
    ) -> Dict:
        if category not in DOCUMENT_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Unknown category '{category}'.")
        kb = svc(request).knowledge_base
        max_bytes = settings.max_upload_mb * 1024 * 1024
        results = []
        for upload in files:
            data = await upload.read(max_bytes + 1)
            try:
                results.append(kb.ingest_upload(upload.filename or "", data, category))
            except ValidationError as exc:
                results.append({"status": "rejected", "filename": upload.filename, "message": str(exc)})
        if results and all(r["status"] == "rejected" for r in results):
            raise HTTPException(status_code=400, detail=results)
        return {"results": results, "stats": kb.stats()}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str, request: Request) -> Dict:
        doc = svc(request).knowledge_base.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        return doc

    @app.get("/documents/{doc_id}/chunks")
    def get_chunks(doc_id: str, request: Request) -> Dict:
        s = svc(request)
        if not s.knowledge_base.get_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"doc_id": doc_id, "chunks": s.vector_db.get_document_chunks(doc_id)}

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request) -> Dict:
        if not svc(request).knowledge_base.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"status": "deleted", "doc_id": doc_id}

    @app.post("/documents/{doc_id}/reindex")
    def reindex_document(doc_id: str, request: Request) -> Dict:
        try:
            return svc(request).knowledge_base.reindex_document(doc_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Document not found.")

    @app.post("/documents/reindex")
    def reindex_all(request: Request) -> Dict:
        return {"results": svc(request).knowledge_base.reindex_all()}

    @app.post("/documents/sync")
    def sync_folder(request: Request) -> Dict:
        return {"results": svc(request).knowledge_base.sync_folder()}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("main:app", host=s.api_host, port=s.api_port)

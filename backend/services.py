"""Wires settings, embeddings, vector store, knowledge base and RAG pipeline together."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import Settings
from backend.embeddings import get_embeddings
from backend.knowledge_base import KnowledgeBaseManager
from backend.rag_pipeline import ClaudeClient, LLMClient, RAGPipeline
from backend.registry import DocumentRegistry
from backend.retriever import Retriever
from backend.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    vector_db: VectorDatabase
    knowledge_base: KnowledgeBaseManager
    retriever: Retriever
    pipeline: RAGPipeline


def build_services(settings: Settings, llm: Optional[LLMClient] = None) -> Services:
    embeddings = get_embeddings(settings.embedding_model)
    vector_db = VectorDatabase(settings.vectorstore_dir, settings.collection_name, embeddings)
    registry = DocumentRegistry(settings.registry_path)
    kb = KnowledgeBaseManager(settings, vector_db, registry)
    retriever = Retriever(vector_db, settings.top_k, settings.relevance_threshold)
    if llm is None and settings.llm_configured:
        if settings.llm_provider == "gemini":
            from backend.gemini_client import GeminiClient

            llm = GeminiClient(settings)
        else:
            llm = ClaudeClient(settings)
    if llm is None:
        logger.warning("%s not set - retrieval works, but questions cannot be answered.", settings.llm_key_name)
    return Services(settings, vector_db, kb, retriever, RAGPipeline(retriever, llm, settings))

"""Split loaded documents into overlapping, retrieval-sized chunks."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
        add_start_index=True,
    )


def chunk_documents(docs: List[Document], doc_id: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Chunk documents and attach stable ids + positional metadata to each chunk.

    Sections are chunked independently so a chunk never mixes two headings.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = total
        chunk.metadata["chunk_id"] = f"{doc_id}:{i}"
        chunk.id = chunk.metadata["chunk_id"]
    return chunks

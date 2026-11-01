"""Split loaded documents into overlapping, retrieval-sized chunks."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


@lru_cache(maxsize=2)
def embedding_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """The embedding model's own tokenizer, so chunk sizes are measured in the tokens the model reads."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )


def build_token_splitter(chunk_tokens: int, overlap_tokens: int,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> RecursiveCharacterTextSplitter:
    """Split on paragraph/line/sentence boundaries, measuring length in embedding-model tokens."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedding_tokenizer(model_name), chunk_size=chunk_tokens, chunk_overlap=overlap_tokens,
        separators=_SEPARATORS, add_start_index=True,
    )


def chunk_documents(docs: List[Document], doc_id: str, chunk_size: int, chunk_overlap: int,
                    model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Document]:
    """Chunk documents and attach stable ids + positional metadata to each chunk.

    Sections are chunked independently so a chunk never mixes two headings.
    """
    splitter = build_token_splitter(chunk_size, chunk_overlap, model_name)
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

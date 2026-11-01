"""Cross-encoder reranking of retrieved candidates.

The embedding model encodes the question and each chunk separately, so it can only compare two
summaries. A cross-encoder reads the question and the chunk together and scores how well that chunk
answers that question. It is too slow to run over the whole index, so it re-scores a small candidate
pool from the first-stage (hybrid) search.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import List, Sequence

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Relevance probability (0-1) of each text for the query."""
        if not texts:
            return []
        logits = _load(self.model_name).predict([(query, t) for t in texts], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

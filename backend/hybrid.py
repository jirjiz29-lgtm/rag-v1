"""Keyword (BM25) retrieval and rank fusion, used alongside semantic search.

Dense embeddings match meaning ("pause a trek" ~ "change an activity") but can miss exact terms such as
field names (`gc_pax`), codes or rare words. BM25 scores exact term overlap. Reciprocal-rank fusion
combines the two rankings without having to calibrate their very different score scales.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or our should "
    "so that the their them there these they this to was we what when where which who why will with you your "
    "about any all also after before each per may must not no than then into been being".split()
)


def contextual_text(text: str, title: str = "", section: str = "") -> str:
    """Text used for keyword search: the chunk prefixed with the document title and section heading.

    A chunk such as "Collect and verify, as appropriate: ..." never says "vendor"; with
    "Vendor Management SOP > 2. Vendor onboarding" in front, keyword search matches vendor questions.
    (Evaluation showed the same prefix *hurts* semantic search on short chunks, so embeddings use the
    plain chunk.) Citations always show the chunk's own text.
    """
    where = " > ".join(p for p in (title, section) if p)
    return f"{where}\n{text}" if where else text


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens (keeps snake_case identifiers whole), stopwords removed, light plural stemming."""
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed list of documents (k1=1.5, b=0.75)."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(t)) for t in texts]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self._tfs)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int, allowed: Iterable[str] = None) -> List[Tuple[str, float]]:
        terms = [t for t in tokenize(query) if t in self._idf]
        if not terms or not self.ids:
            return []
        allowed_set = set(allowed) if allowed is not None else None
        scored = []
        for i, tf in enumerate(self._tfs):
            if allowed_set is not None and self.ids[i] not in allowed_set:
                continue
            norm = self.k1 * (1 - self.b + self.b * self._lens[i] / (self._avg_len or 1))
            score = sum(self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if score > 0:
                scored.append((self.ids[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] = None) -> List[Tuple[str, float]]:
    """Reciprocal-rank fusion: score(id) = sum_r w_r / (k + rank_r(id)). Higher is better."""
    weights = weights or [1.0] * len(rankings)
    scores: Dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

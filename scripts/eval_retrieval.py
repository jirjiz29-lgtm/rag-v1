"""Retrieval evaluation harness for the knowledge base.

Measures how well a retrieval configuration finds the evidence for each question in
tests/data/retrieval_eval.json, and how much irrelevant context it would send to the LLM.

Two modes:
  * experiments (default): builds in-memory indexes for many configurations (chunking strategy,
    chunk size/overlap, contextual headers, dense/BM25/hybrid search, reranking, thresholds,
    multi-part query handling) and prints a comparison table.
  * --production: runs the app's real Retriever end to end (temporary Chroma index built through
    the normal ingestion path), so before/after numbers come from the code that actually ships.

Usage:
    python scripts/eval_retrieval.py                       # experiment grid
    python scripts/eval_retrieval.py --production          # current app retriever
    python scripts/eval_retrieval.py --json out.json       # also save raw results
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

EVAL_FILE = ROOT / "tests" / "data" / "retrieval_eval.json"
KB_DIR = ROOT / "knowledge_base"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- eval set
@dataclass
class EvalQuestion:
    id: str
    type: str
    question: str
    evidence: List[List[str]]  # each item: list of alternative phrases (any one satisfies it)

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_eval_set(path: Path = EVAL_FILE) -> List[EvalQuestion]:
    data = json.loads(path.read_text())
    questions = []
    for q in data["questions"]:
        items = []
        for item in q["evidence"]:
            alts = item if isinstance(item[0], list) else [item]
            items.append([_norm(text) for _doc, text in alts])
        questions.append(EvalQuestion(q["id"], q["type"], q["question"], items))
    return questions


# --------------------------------------------------------------------------- metrics
@dataclass
class QuestionResult:
    question: EvalQuestion
    texts: List[str]  # retrieved chunk texts in rank order (what would be sent to the LLM)
    tokens: int
    seconds: float

    def _found(self, item: List[str]) -> Optional[int]:
        for rank, text in enumerate(self.texts, start=1):
            t = _norm(text)
            if any(alt in t for alt in item):
                return rank
        return None

    @property
    def recall(self) -> float:
        items = self.question.evidence
        return sum(self._found(i) is not None for i in items) / len(items)

    @property
    def full_coverage(self) -> bool:
        return all(self._found(i) is not None for i in self.question.evidence)

    @property
    def reciprocal_rank(self) -> float:
        ranks = [r for r in (self._found(i) for i in self.question.evidence) if r]
        return 1.0 / min(ranks) if ranks else 0.0

    @property
    def relevant_chunks(self) -> int:
        alts = [a for item in self.question.evidence for a in item]
        return sum(any(a in _norm(t) for a in alts) for t in self.texts)


@dataclass
class Summary:
    name: str
    recall: float
    full_coverage: float
    multi_coverage: float
    paraphrase_recall: float
    mrr: float
    avg_chunks: float
    avg_tokens: float
    context_precision: float
    answerable_rejected: int
    unanswerable_rejected: int
    unanswerable_total: int
    ms_per_query: float

    def row(self) -> str:
        return (f"| {self.name} | {self.recall:.3f} | {self.full_coverage:.3f} | {self.multi_coverage:.3f} | "
                f"{self.paraphrase_recall:.3f} | {self.mrr:.3f} | {self.avg_chunks:.1f} | {self.avg_tokens:.0f} | "
                f"{self.context_precision:.3f} | {self.answerable_rejected} | "
                f"{self.unanswerable_rejected}/{self.unanswerable_total} | {self.ms_per_query:.0f} |")


TABLE_HEADER = ("| Config | Recall | Full coverage | Multi-doc coverage | Paraphrase recall | MRR | Chunks sent | "
                "Tokens sent | Context precision | Answerable rejected | Unanswerable rejected | ms/query |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|")


def summarize(name: str, results: List[QuestionResult]) -> Summary:
    ans = [r for r in results if r.question.answerable]
    una = [r for r in results if not r.question.answerable]
    multi = [r for r in ans if r.question.type == "multi"]
    para = [r for r in ans if r.question.type == "paraphrase"]
    sent = [r for r in ans if r.texts]
    return Summary(
        name=name,
        recall=statistics.mean(r.recall for r in ans),
        full_coverage=statistics.mean(r.full_coverage for r in ans),
        multi_coverage=statistics.mean(r.full_coverage for r in multi) if multi else 0.0,
        paraphrase_recall=statistics.mean(r.recall for r in para) if para else 0.0,
        mrr=statistics.mean(r.reciprocal_rank for r in ans),
        avg_chunks=statistics.mean(len(r.texts) for r in ans),
        avg_tokens=statistics.mean(r.tokens for r in ans),
        context_precision=(sum(r.relevant_chunks for r in sent) / sum(len(r.texts) for r in sent)) if sent else 0.0,
        answerable_rejected=sum(1 for r in ans if not r.texts),
        unanswerable_rejected=sum(1 for r in una if not r.texts),
        unanswerable_total=len(una),
        ms_per_query=1000 * statistics.mean(r.seconds for r in results),
    )


# --------------------------------------------------------------------------- corpus + chunking
def _tokenizer():
    from backend.text_splitter import embedding_tokenizer

    return embedding_tokenizer()


def count_tokens(text: str) -> int:
    return len(_tokenizer().tokenize(text))


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str


def load_sections(kb_dir: Path = KB_DIR):
    from config import SUPPORTED_EXTENSIONS
    from backend.document_loader import load_document

    return {p.name: load_document(p) for p in sorted(kb_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS}


def chunk_sections_chars(sections, size: int = 1000, overlap: int = 200) -> List[Chunk]:
    """The original strategy: split by heading, then by characters."""
    from backend.text_splitter import build_splitter

    out = []
    for doc, docs in sections.items():
        for i, c in enumerate(build_splitter(size, overlap).split_documents(docs)):
            out.append(Chunk(f"{doc}:{i}", doc, c.metadata.get("title", doc), c.metadata.get("section", ""), c.page_content))
    return out


def chunk_sections_tokens(sections, size: int, overlap: int) -> List[Chunk]:
    """Split by heading, then cap each piece at `size` embedding-model tokens."""
    from backend.text_splitter import build_token_splitter

    splitter = build_token_splitter(size, overlap)
    out = []
    for doc, docs in sections.items():
        for i, c in enumerate(splitter.split_documents(docs)):
            out.append(Chunk(f"{doc}:{i}", doc, c.metadata.get("title", doc), c.metadata.get("section", ""), c.page_content))
    return out


def chunk_windows_tokens(sections, size: int, overlap: int) -> List[Chunk]:
    """Fixed token windows over the whole document, ignoring headings (the classic 'chunk size N' setup)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(), chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    out = []
    for doc, docs in sections.items():
        full = "\n\n".join(d.page_content for d in docs)
        for i, text in enumerate(splitter.split_text(full)):
            out.append(Chunk(f"{doc}:{i}", doc, docs[0].metadata.get("title", doc), "", text))
    return out


# --------------------------------------------------------------------------- in-memory retrieval
_SPLIT_CLAUSE = re.compile(
    r"(?:\?\s+|[,;]\s*and\s+|\s+and\s+(?=(?:how|what|who|when|where|which|why|is|are|can|should|does|do)\b))",
    re.IGNORECASE,
)


def split_subqueries(question: str) -> List[str]:
    """Split a multi-part question into its parts ("How is X done, and what does Y say about Z?").

    Each part is searched separately so a question spanning two documents gets evidence from both,
    instead of the stronger half crowding out the other. Single questions come back unchanged.
    """
    parts = [p.strip(" ?,.;") for p in _SPLIT_CLAUSE.split(question or "")]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts if len(parts) > 1 else [question]


@dataclass
class Config:
    name: str
    chunker: Callable = None
    header: bool = False             # prefix "title > section" for semantic search
    header_bm25: Optional[bool] = None   # ... for keyword search (defaults to `header`)
    header_rerank: Optional[bool] = None  # ... for the cross-encoder (defaults to `header`)
    question_floor: Optional[float] = None  # reject the question if no chunk reaches this cosine
    method: str = "dense"            # dense | bm25 | hybrid
    pool: int = 20                   # first-stage candidates (hybrid / rerank)
    rerank: bool = False
    top_k: int = 6
    dense_threshold: Optional[float] = 0.25   # applied to dense cosine (only when not reranking)
    rerank_threshold: Optional[float] = None  # applied to cross-encoder probability
    relative_cutoff: Optional[float] = None   # drop chunks scoring < cutoff * best score (reranked)
    per_doc_cap: Optional[int] = None
    decompose: bool = False


def embed_text(chunk: Chunk, header: bool) -> str:
    from backend.hybrid import contextual_text

    return contextual_text(chunk.text, chunk.title, chunk.section) if header else chunk.text


class MemoryIndex:
    def __init__(self, chunks: List[Chunk], header: bool, header_bm25: bool):
        from backend.embeddings import get_embeddings
        from backend.hybrid import BM25Index

        self.chunks = chunks
        self.emb = get_embeddings()
        self.matrix = np.array(self.emb.embed_documents([embed_text(c, header) for c in chunks]))
        self.bm25 = BM25Index([c.id for c in chunks], [embed_text(c, header_bm25) for c in chunks])
        self.by_id = {c.id: c for c in chunks}

    def dense(self, query: str, k: int):
        q = np.array(self.emb.embed_query(query))
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i].id, float(sims[i])) for i in order]


def run_config(cfg: Config, index: MemoryIndex, questions: Sequence[EvalQuestion]) -> List[QuestionResult]:
    from backend.hybrid import rrf_fuse
    from backend.reranker import CrossEncoderReranker
    from backend.retriever import select_chunks

    reranker = CrossEncoderReranker() if cfg.rerank else None
    results = []
    for q in questions:
        start = time.perf_counter()
        queries = split_subqueries(q.question) if cfg.decompose else [q.question]
        dense_scores: Dict[str, float] = {}
        per_query_rankings = []
        for sub in queries:
            dense = index.dense(sub, cfg.pool if cfg.method != "dense" or cfg.rerank else cfg.top_k)
            for cid, s in dense:
                dense_scores[cid] = max(s, dense_scores.get(cid, 0.0))
            if cfg.method == "dense":
                ranking = [cid for cid, _ in dense]
            elif cfg.method == "bm25":
                ranking = [cid for cid, _ in index.bm25.search(sub, cfg.pool)]
            else:
                ranking = [cid for cid, _ in rrf_fuse([[c for c, _ in dense], [c for c, _ in index.bm25.search(sub, cfg.pool)]])]
            per_query_rankings.append(ranking[: cfg.pool])

        if cfg.question_floor is not None and max(dense_scores.values(), default=0.0) < cfg.question_floor:
            picked = []
        elif not cfg.rerank:
            fused = per_query_rankings[0] if len(per_query_rankings) == 1 else [c for c, _ in rrf_fuse(per_query_rankings)]
            picked = []
            for cid in fused:
                if cfg.method == "dense" and cfg.dense_threshold is not None and dense_scores.get(cid, 0) < cfg.dense_threshold:
                    continue
                picked.append(cid)
            picked = picked[: cfg.top_k]
        else:
            candidates = list(dict.fromkeys(c for ranking in per_query_rankings for c in ranking))
            hr = cfg.header if cfg.header_rerank is None else cfg.header_rerank
            texts = [embed_text(index.by_id[c], hr) for c in candidates]
            # Each candidate's relevance = best score against the full question or any sub-question.
            scores = {c: 0.0 for c in candidates}
            best_for_sub = []
            for sub in dict.fromkeys([q.question, *queries]):
                sub_scores = reranker.score(sub, texts)
                for c, s in zip(candidates, sub_scores):
                    scores[c] = max(scores[c], s)
                if sub != q.question:
                    best_for_sub.append(max(zip(candidates, sub_scores), key=lambda x: x[1]))
            picked = select_chunks(
                [(c, scores[c], index.by_id[c].doc) for c in candidates],
                top_k=cfg.top_k, threshold=cfg.rerank_threshold, relative_cutoff=cfg.relative_cutoff,
                per_doc_cap=cfg.per_doc_cap,
                must_include=[c for c, s in best_for_sub if cfg.rerank_threshold is None or s >= cfg.rerank_threshold],
            )
        texts = [index.by_id[c].text for c in picked]
        results.append(QuestionResult(q, texts, sum(count_tokens(t) for t in texts), time.perf_counter() - start))
    return results


# --------------------------------------------------------------------------- production retriever
def run_production(questions: Sequence[EvalQuestion], kb_dir: Path = KB_DIR, **overrides) -> List[QuestionResult]:
    """Index the knowledge base through the app's real ingestion path and query its real Retriever."""
    from backend.services import build_services
    from config import get_settings

    tmp = Path(tempfile.mkdtemp(prefix="retrieval-eval-"))
    try:
        settings = get_settings().with_overrides(knowledge_base_dir=kb_dir, vectorstore_dir=tmp / "vs",
                                                 collection_name="retrieval_eval", **overrides)
        services = build_services(settings, llm=_NoLLM())
        services.knowledge_base.sync_folder()
        results = []
        for q in questions:
            start = time.perf_counter()
            chunks = services.retriever.retrieve(q.question)
            elapsed = time.perf_counter() - start
            texts = [c.text for c in chunks]
            results.append(QuestionResult(q, texts, sum(count_tokens(t) for t in texts), elapsed))
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _NoLLM:
    model = "none"

    def generate(self, *a, **k):  # pragma: no cover - retrieval-only evaluation
        raise RuntimeError("not used")


# --------------------------------------------------------------------------- experiment grid
def experiment_grid(sections) -> List[Config]:
    from functools import partial

    base = Config("baseline: section chunks 1000 chars, dense, top-6, cos≥0.25",
                  chunker=partial(chunk_sections_chars, size=1000, overlap=200))
    grid = [base]
    # 1-2. Chunk size / overlap in tokens: fixed windows ignoring headings vs heading-aware chunks.
    for size in (400, 600, 800):
        for ov in (0, int(size * 0.1), int(size * 0.2)):
            grid.append(replace(base, name=f"windows {size} tok / overlap {ov}",
                                chunker=partial(chunk_windows_tokens, size=size, overlap=ov)))
    for size, ov in ((128, 16), (200, 30), (256, 32)):
        grid.append(replace(base, name=f"section chunks {size} tok / overlap {ov}",
                            chunker=partial(chunk_sections_tokens, size=size, overlap=ov)))
    sec = partial(chunk_sections_tokens, size=200, overlap=30)
    # 3. Better semantic retrieval: embed title + section heading with each chunk.
    grid.append(replace(base, name="section 200 tok + contextual headers", chunker=sec, header=True))
    # 6. Keyword-only and hybrid search.
    grid.append(replace(base, name="  + BM25 only", chunker=sec, header=True, method="bm25", dense_threshold=None))
    grid.append(replace(base, name="  + hybrid (dense+BM25, RRF)", chunker=sec, header=True, method="hybrid", dense_threshold=None))
    # 7. Cross-encoder reranking of the hybrid pool, without and with thresholds (5, 9).
    grid.append(replace(base, name="  + rerank, top-6, no threshold", chunker=sec, header=True, method="hybrid",
                        rerank=True, dense_threshold=None))
    for thr in (0.05, 0.1, 0.2, 0.3):
        grid.append(replace(base, name=f"  + rerank, top-6, p≥{thr}", chunker=sec, header=True, method="hybrid",
                            rerank=True, dense_threshold=None, rerank_threshold=thr))
    grid.append(replace(base, name="  + rerank p≥0.1, relative cutoff 0.1", chunker=sec, header=True, method="hybrid",
                        rerank=True, dense_threshold=None, rerank_threshold=0.1, relative_cutoff=0.1))
    # 8. Multi-document questions: decompose multi-part questions, cap chunks per document.
    grid.append(replace(base, name="  + rerank p≥0.1 + sub-queries", chunker=sec, header=True, method="hybrid",
                        rerank=True, dense_threshold=None, rerank_threshold=0.1, decompose=True))
    grid.append(replace(base, name="  + rerank p≥0.1 + sub-queries + ≤3 per doc", chunker=sec, header=True,
                        method="hybrid", rerank=True, dense_threshold=None, rerank_threshold=0.1, decompose=True,
                        per_doc_cap=3))
    # Round 2 (after round 1 showed headers hurt dense search and absolute reranker thresholds reject
    # correct paraphrase matches): headers only where they help, relative cut-offs, multi-doc handling.
    r2 = replace(base, chunker=sec, method="hybrid", dense_threshold=None)
    grid += [
        replace(r2, name="R2 hybrid, no headers"),
        replace(r2, name="R2 hybrid, headers for BM25 only", header_bm25=True),
        replace(r2, name="R2 + rerank (plain text)", header_bm25=True, rerank=True),
        replace(r2, name="R2 + rerank (text with headers)", header_bm25=True, rerank=True, header_rerank=True),
        replace(r2, name="R2 + rerank + sub-queries", header_bm25=True, rerank=True, decompose=True),
        replace(r2, name="R2 + rerank + sub-queries + ≤2 per doc", header_bm25=True, rerank=True, decompose=True, per_doc_cap=2),
    ]
    for ratio in (0.001, 0.01, 0.05):
        grid.append(replace(r2, name=f"R2 + rerank + sub-queries, keep p ≥ {ratio}×best", header_bm25=True,
                            rerank=True, decompose=True, relative_cutoff=ratio))
    for k in (4, 5, 8):
        grid.append(replace(r2, name=f"R2 full, top-{k}, keep p ≥ 0.01×best", header_bm25=True, rerank=True,
                            decompose=True, relative_cutoff=0.01, top_k=k))
    for floor in (0.2, 0.25):
        grid.append(replace(r2, name=f"R2 full, top-6, 0.01×best, question floor cos {floor}", header_bm25=True,
                            rerank=True, decompose=True, relative_cutoff=0.01, question_floor=floor))
    return grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true", help="evaluate the app's real retriever")
    parser.add_argument("--only", help="run only configs whose name contains this text")
    parser.add_argument("--json", type=Path, help="write per-config summaries to this file")
    args = parser.parse_args()

    questions = load_eval_set()
    summaries = []
    if args.production:
        summaries.append(summarize("production retriever", run_production(questions)))
    else:
        sections = load_sections()
        indexes: Dict = {}
        for cfg in experiment_grid(sections):
            if args.only and args.only not in cfg.name:
                continue
            hb = cfg.header if cfg.header_bm25 is None else cfg.header_bm25
            key = (cfg.chunker.func.__name__, tuple(sorted(cfg.chunker.keywords.items())), cfg.header, hb)
            if key not in indexes:
                indexes[key] = MemoryIndex(cfg.chunker(sections), cfg.header, hb)
            s = summarize(cfg.name, run_config(cfg, indexes[key], questions))
            summaries.append(s)
            print(s.row(), flush=True)
    print("\n" + TABLE_HEADER)
    for s in summaries:
        print(s.row())
    if args.json:
        args.json.write_text(json.dumps([s.__dict__ for s in summaries], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

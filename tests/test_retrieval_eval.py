"""Retrieval quality: before/after evaluation on the real knowledge base, plus unit tests of each stage.

The evaluation set (tests/data/retrieval_eval.json) has 47 answerable questions with verbatim evidence
phrases (single-document, paraphrased, keyword and multi-document) and 10 the documents can't answer.
"""
import json
import re
import sys
from functools import partial
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import (  # noqa: E402
    EVAL_FILE, KB_DIR, Config, MemoryIndex, chunk_sections_chars, load_eval_set, load_sections, run_config,
    run_production, summarize,
)

from backend.hybrid import BM25Index, rrf_fuse, tokenize  # noqa: E402
from backend.retriever import Retriever, select_chunks  # noqa: E402


@pytest.fixture(scope="module")
def questions():
    return load_eval_set()


@pytest.fixture(scope="module")
def before(questions):
    """The original retriever: heading chunks of 1000 characters, semantic search only, top-6, cosine >= 0.25."""
    cfg = Config("before", chunker=partial(chunk_sections_chars, size=1000, overlap=200))
    index = MemoryIndex(cfg.chunker(load_sections()), header=False, header_bm25=False)
    return summarize("before", run_config(cfg, index, questions))


@pytest.fixture(scope="module")
def after(questions):
    """The app's real retriever with its default settings (hybrid + rerank), indexed through normal ingestion."""
    return summarize("after", run_production(questions, chunk_size=200, chunk_overlap=30, top_k=6,
                                             relevance_threshold=0.2, hybrid_search=True, candidate_pool=20,
                                             reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                             rerank_keep_ratio=0.001))


# ---------------------------------------------------------------------- evaluation set
def test_eval_set_evidence_exists_verbatim():
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    docs = {p.name: norm(p.read_text()) for p in KB_DIR.glob("*.md")}
    data = json.loads(EVAL_FILE.read_text())
    for q in data["questions"]:
        for item in q["evidence"]:
            for doc, text in (item if isinstance(item[0], list) else [item]):
                assert norm(text) in docs[doc], (q["id"], doc, text)
    kinds = {q["type"] for q in data["questions"]}
    assert {"single", "paraphrase", "keyword", "multi", "unanswerable"} <= kinds


# ---------------------------------------------------------------------- before vs after
def test_after_beats_before_on_every_accuracy_metric(before, after):
    assert after.recall > before.recall
    assert after.full_coverage > before.full_coverage
    assert after.multi_coverage > before.multi_coverage
    assert after.paraphrase_recall >= before.paraphrase_recall
    assert after.mrr > before.mrr


def test_after_sends_less_and_more_relevant_context(before, after):
    assert after.avg_tokens < before.avg_tokens
    assert after.avg_chunks < before.avg_chunks
    assert after.context_precision > before.context_precision


def test_after_never_withholds_context_for_answerable_questions(before, after):
    # Grounding comes first: a question the documents answer must reach the LLM with its evidence.
    assert after.answerable_rejected == 0
    assert after.answerable_rejected <= before.answerable_rejected


def test_after_meets_quality_bar(after):
    assert after.recall >= 0.95
    assert after.full_coverage >= 0.90
    assert after.multi_coverage >= 0.80
    assert after.paraphrase_recall >= 0.95
    assert after.unanswerable_rejected >= 2  # clearly off-topic questions skip the LLM entirely


# ---------------------------------------------------------------------- stages
def test_bm25_keeps_identifiers_and_ranks_exact_terms():
    assert "gc_pax" in tokenize("What is the gc_pax field?")
    index = BM25Index(["a", "b"], ["Lead fields: gc_name, gc_pax, gc_budget", "Group size and travellers"])
    assert index.search("gc_pax", 2)[0][0] == "a"
    assert index.search("gc_pax", 2, allowed=["b"]) == []


def test_rrf_rewards_items_ranked_well_by_both():
    fused = [cid for cid, _ in rrf_fuse([["x", "y", "z"], ["y", "z", "x"]])]
    assert fused[0] == "y"


def test_select_chunks_applies_relative_cutoff_and_top_k():
    scored = [("a", 0.9, "d1"), ("b", 0.5, "d2"), ("c", 0.0005, "d3"), ("d", 0.3, "d1")]
    assert select_chunks(scored, top_k=2) == ["a", "b"]
    assert select_chunks(scored, top_k=6, relative_cutoff=0.001) == ["a", "b", "d"]


class _BrokenReranker:
    def score(self, query, texts):
        raise RuntimeError("model unavailable")


class _ReverseReranker:
    def score(self, query, texts):
        return [i / len(texts) for i in range(len(texts))]  # prefers the last candidates


def test_reranker_reorders_and_failure_falls_back_to_hybrid_order(loaded_services):
    vdb = loaded_services.vector_db
    q = "How many quotes do I need for a purchase over $10,000?"
    hybrid = Retriever(vdb, top_k=6, relevance_threshold=0.2)
    reranked = Retriever(vdb, top_k=6, relevance_threshold=0.2, reranker=_ReverseReranker(), keep_ratio=0)
    broken = Retriever(vdb, top_k=6, relevance_threshold=0.2, reranker=_BrokenReranker())
    base_ids = [c.chunk_id for c in hybrid.retrieve(q)]
    assert [c.chunk_id for c in reranked.retrieve(q)] != base_ids
    assert all(c.rerank_score is not None for c in reranked.retrieve(q))
    assert [c.chunk_id for c in broken.retrieve(q)] == base_ids


def test_keyword_only_hits_get_a_real_cosine_score(loaded_services):
    retriever = Retriever(loaded_services.vector_db, top_k=20, relevance_threshold=0.0, candidate_pool=20)
    chunks = retriever.retrieve("P1 critical ticket first response")
    assert chunks and all(0.0 < c.score <= 1.0 for c in chunks)


def test_question_floor_skips_off_topic_questions(loaded_services):
    retriever = Retriever(loaded_services.vector_db, top_k=6, relevance_threshold=0.2)
    assert retriever.retrieve("Who won the 2011 cricket world cup?") == []
    assert retriever.retrieve("How much is the home office stipend?")

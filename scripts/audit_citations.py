"""Ask real questions through the configured LLM and verify every citation against the vector store.

For each cited source this checks, independently of the pipeline, that the chunk exists in the
index and that the document name, page, section and excerpt shown to the user are exactly the
stored metadata, and that highlighted sentences are verbatim text from that chunk.

Usage:
    python scripts/audit_citations.py              # live knowledge base (knowledge_base/)
    python scripts/audit_citations.py --samples    # temporary index of sample_documents/ (PDF/DOCX/TXT)
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag_pipeline import LLMError  # noqa: E402
from backend.services import build_services  # noqa: E402
from config import get_settings  # noqa: E402

LIVE_QUESTIONS = [
    "What should I do if a customer complains during a trip? Give me the steps.",
    "How should customer personal data be protected across sales, support and operations?",
    "What checks are needed before confirming a new vendor, and how should vendor issues during a trip be reported?",
    "What rules apply to advertising claims and discounts, and who must approve package prices?",
]
SAMPLE_QUESTIONS = [
    "How many days of paid time off do full-time employees get, and how do I request it?",
    "What is the meal allowance for business travel?",
    "Compare the remote work rules in the employee handbook and the remote work policy.",
    "How many vendor quotes are needed for a large purchase, and who approves expense claims?",
]


def answer_with_retry(services, question, attempts: int = 3):
    """The LLM can be briefly overloaded; that says nothing about citations, so retry before giving up."""
    for attempt in range(1, attempts + 1):
        try:
            return services.pipeline.answer(question)
        except LLMError as exc:
            if attempt == attempts:
                print(f"   - skipped: {exc}")
                return None
            time.sleep(10 * attempt)


def audit(services, questions) -> tuple:
    """Returns (problems, citations_checked, skipped). Every audit question is answerable from the documents,
    so an unanswered one counts as a problem: an audit that checks nothing must not report success."""
    problems = checked = skipped = 0
    for q in questions:
        print(f"\nQ: {q}")
        result = answer_with_retry(services, q)
        if result is None:
            skipped += 1
            continue
        print(f"   model={result.model}  found={result.found}  sources={len(result.sources)}  "
              f"documents={len({s['document'] for s in result.sources})}")
        if not result.found or not result.sources:
            problems += 1
            print("   ✗ no cited answer (the documents cover this question)")
        markers = set(re.findall(r"\[(S\d+)\]", result.answer))
        cited = {s["source_id"] for s in result.sources}
        if markers - cited:
            problems += 1
            print(f"   ✗ answer cites ids with no source: {sorted(markers - cited)}")
        for src in result.sources:
            stored = next((c for c in services.vector_db.get_document_chunks(src["doc_id"]) if c["id"] == src["chunk_id"]), None)
            issues = []
            if stored is None:
                issues.append("chunk not in index")
            else:
                meta = stored["metadata"]
                if src["document"] != meta["source"]:
                    issues.append("document name differs")
                if src["excerpt"] != stored["text"]:
                    issues.append("excerpt differs")
                if src["page"] is not None and (meta.get("file_type") != "pdf" or src["page"] != meta.get("page")):
                    issues.append(f"page {src['page']} not in metadata")
                if (src["section"] or "") not in ("", meta.get("section")):
                    issues.append("section differs")
                if any(h not in stored["text"] for h in src.get("highlights", [])):
                    issues.append("highlight not verbatim")
            problems += bool(issues)
            checked += 1
            status = "✗ " + ", ".join(issues) if issues else "✓"
            print(f"   [{src['number']}] {src['citation']}  {status}")
            for h in src.get("highlights", [])[:2]:
                print(f"         ▸ {h[:110]}")
        for c in result.conflicts:
            print(f"   ⚠ conflict: {c['description'][:120]} ({', '.join(c['documents'])})")
    return problems, checked, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true", help="audit a temporary index of sample_documents/")
    args = parser.parse_args()
    settings = get_settings()
    if args.samples:
        tmp = Path(tempfile.mkdtemp(prefix="citation-audit-"))
        settings = settings.with_overrides(knowledge_base_dir=tmp / "kb", vectorstore_dir=tmp / "vs",
                                           collection_name="citation_audit")
        services = build_services(settings)
        for f in sorted((ROOT / "sample_documents").iterdir()):
            services.knowledge_base.ingest_upload(f.name, f.read_bytes())
        questions = SAMPLE_QUESTIONS
    else:
        services = build_services(settings)
        services.knowledge_base.sync_folder()  # index knowledge_base/ exactly as the app does at startup
        questions = LIVE_QUESTIONS
    problems, checked, skipped = audit(services, questions)
    if problems:
        print(f"\n{problems} problem(s) found.")
    elif skipped:
        print(f"\n{checked} citations verified; {skipped} question(s) skipped because the LLM was unavailable.")
    else:
        print(f"\nAll {checked} citations verified against the index.")
    return 1 if problems or skipped else 0


if __name__ == "__main__":
    sys.exit(main())

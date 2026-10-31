"""Citation system: every citation field must come from stored chunk metadata, never from the model."""
import pytest

from backend.citations import highlight_sentences, scrub_page_mentions, valid_page, valid_section
from backend.document_loader import load_document


def stored_chunk(services, src):
    """Look the cited chunk up in the vector store, independently of the pipeline."""
    chunks = services.vector_db.get_document_chunks(src["doc_id"])
    return next(c for c in chunks if c["id"] == src["chunk_id"])


def cite_each_source(sources, question):
    """Fake model: one sentence per source, built from that source's own first line."""
    lines = [f"- {s['text'].splitlines()[-1][:120]} [{s['id']}]" for s in sources[:3]]
    return {"answer": "\n".join(lines), "found_in_documents": True,
            "cited_sources": [s["id"] for s in sources[:3]], "conflicts": []}


# Citations match the index exactly ----------------------------------------------------
@pytest.mark.parametrize("question", [
    "How many days of paid time off do full-time employees get?",          # PDF
    "How much is the home office stipend?",                                 # DOCX
    "What is the first response time for a P1 critical ticket?",           # TXT
    "How many quotes do I need for a purchase over $10,000?",              # DOCX with table
])
def test_every_citation_field_matches_stored_metadata(pipeline, loaded_services, question):
    loaded_services.fake.responder = cite_each_source
    result = pipeline.answer(question)
    assert result.found and result.sources
    for src in result.sources:
        stored = stored_chunk(loaded_services, src)
        meta = stored["metadata"]
        assert src["document"] == meta["source"]
        assert src["excerpt"] == stored["text"]
        assert src["page"] == (meta.get("page") if meta["file_type"] == "pdf" else None)
        assert src["section"] == (meta.get("section") or None)
        assert src["chunk_index"] == meta["chunk_index"]
        # The display label is assembled from those same fields.
        assert src["citation"].startswith(src["document"])
        if src["page"]:
            assert f"p. {src['page']}" in src["citation"]
        if src["section"]:
            assert src["citation"].endswith(src["section"])


def test_pdf_citation_has_exact_page_and_section(pipeline):
    result = pipeline.answer("How many days of paid time off do full-time employees get?")
    src = result.sources[0]
    assert (src["document"], src["page"], src["total_pages"], src["section"]) == \
        ("employee_handbook.pdf", 1, 3, "3. Paid Time Off")
    assert src["citation"] == "employee_handbook.pdf · p. 1 of 3 · 3. Paid Time Off"
    assert src["page_note"] == ""


def test_non_paged_formats_never_get_a_page_number(pipeline):
    result = pipeline.answer("How much is the home office stipend?")
    src = result.sources[0]
    assert src["document"] == "remote_work_policy.docx"
    assert src["page"] is None and src["total_pages"] is None
    assert "p." not in src["citation"]
    assert src["page_note"] == "DOCX files have no fixed page numbers"


def test_sources_are_numbered_in_order_across_documents(pipeline, loaded_services):
    loaded_services.fake.responder = cite_each_source
    result = pipeline.answer("Compare the remote work rules in the employee handbook and the remote work policy")
    assert [s["number"] for s in result.sources] == list(range(1, len(result.sources) + 1))
    assert len({s["document"] for s in result.sources}) >= 2


# The model cannot inject references ---------------------------------------------------
def test_model_invented_page_and_source_references_are_removed(pipeline, loaded_services):
    def responder(sources, question):
        return {
            "answer": f"Employees get 20 days of PTO (page 7) [{sources[0]['id']}] (see p. 1). Extra rule [S42].",
            "found_in_documents": True,
            "cited_sources": [sources[0]["id"], "S42", "employee_handbook.pdf page 9"],
            "conflicts": [],
        }

    loaded_services.fake.responder = responder
    result = pipeline.answer("How many days of paid time off do full-time employees get?")
    assert "(page 7)" not in result.answer            # no retrieved chunk is on page 7
    assert "[S42]" not in result.answer               # never retrieved
    assert [s["source_id"] for s in result.sources] == ["S1"]
    assert all(s["page"] in (None, 1, 2, 3) for s in result.sources)


def test_scrub_keeps_real_pages_and_removes_fake_ones():
    text = "Rule A (page 2). Rule B (p. 9). Rule C (pp. 2-3). Rule D (pages 2-8)."
    assert scrub_page_mentions(text, {2, 3}) == "Rule A (page 2). Rule B. Rule C (pp. 2-3). Rule D."


@pytest.mark.parametrize("meta,expected", [
    ({"file_type": "pdf", "page": 2, "total_pages": 3}, 2),
    ({"file_type": "pdf", "page": 5, "total_pages": 3}, None),   # beyond the document
    ({"file_type": "pdf", "page": 0}, None),
    ({"file_type": "pdf", "page": "x"}, None),
    ({"file_type": "docx", "page": 2}, None),                     # DOCX has no fixed pages
    ({"file_type": "md"}, None),
])
def test_valid_page(meta, expected):
    assert valid_page(meta) == expected


def test_placeholder_or_empty_sections_are_not_shown_as_headings():
    assert valid_section({"section": "General"}) is None
    assert valid_section({"section": "  "}) is None
    assert valid_section({"section": "4. Remote Work"}) == "4. Remote Work"


def test_text_before_first_heading_gets_no_invented_section(tmp_path):
    f = tmp_path / "memo.txt"
    f.write_text("This memo applies to every office and every contractor we work with.\n\nREFUNDS\n"
                 "Refunds are paid within 7 business days after the finance team approves the request.")
    sections = [d.metadata["section"] for d in load_document(f)]
    assert sections == ["", "REFUNDS"]


# Highlighted excerpts are verbatim -----------------------------------------------------
def test_highlights_are_verbatim_sentences_from_the_cited_chunk(pipeline, loaded_services):
    loaded_services.fake.responder = cite_each_source
    for question in ("How many quotes do I need for a purchase over $10,000?",
                     "What is the escalation process for a P1 ticket?"):
        result = pipeline.answer(question)
        assert any(s["highlights"] for s in result.sources), question
        for src in result.sources:
            for sentence in src["highlights"]:
                assert sentence in src["excerpt"]


def test_highlight_selects_the_supporting_sentence():
    chunk = ("REFUNDS\nSupport agents may approve refunds up to $200. "
             "Larger refunds require Support Manager approval. Tickets close after 5 business days.")
    picked = highlight_sentences(chunk, ["Refunds above $200 need approval from the Support Manager [S1]."])
    assert picked == ["Support agents may approve refunds up to $200.", "Larger refunds require Support Manager approval."]
    assert highlight_sentences(chunk, ["Something completely unrelated about parking."]) == []


def test_marker_after_full_stop_still_links_to_its_sentence():
    from backend.citations import claims_for
    answer = "Page the on-call engineer after 30 minutes. [S1] Notify the account manager. [S2]"
    assert claims_for(answer, "S1") == ["Page the on-call engineer after 30 minutes. [S1]"]
    assert claims_for(answer, "S2") == ["Notify the account manager. [S2]"]


def test_heading_lines_are_never_highlighted():
    chunk = "## 4. During the trip\nDuring the trip\nMaintain a log for:\n- Record supplier failures and customer complaints."
    picked = highlight_sentences(chunk, ["Record supplier failures and complaints during the trip [S1]."],
                                 heading="4. During the trip")
    assert picked == ["- Record supplier failures and customer complaints."]

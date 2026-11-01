"""Ingestion and knowledge base management: validation, duplicates, updates, deletion, re-indexing."""
import pytest

from backend.document_loader import DocumentLoadError, load_document
from backend.knowledge_base import ValidationError, safe_filename


def test_loaders_extract_text_and_metadata(sample_dir):
    pdf = load_document(sample_dir / "employee_handbook.pdf")
    assert {d.metadata["page"] for d in pdf} == {1, 2, 3}
    assert any(d.metadata["section"] == "4. Remote Work" for d in pdf)

    docx = load_document(sample_dir / "vendor_management_procedure.docx")
    assert any("Quotes required" in d.page_content for d in docx)  # table content kept
    assert any(d.metadata["section"] == "1. Vendor Selection" for d in docx)

    txt = load_document(sample_dir / "expense_reimbursement_sop.txt")
    assert any(d.metadata["section"] == "SUBMISSION STEPS" for d in txt)


def test_unsupported_and_empty_files_rejected(services, tmp_path):
    kb = services.knowledge_base
    with pytest.raises(ValidationError):
        kb.ingest_upload("malware.exe", b"MZ...", "General")
    with pytest.raises(ValidationError):
        kb.ingest_upload("notes.txt", b"", "General")
    with pytest.raises(ValidationError):
        kb.ingest_upload("fake.pdf", b"this is not a pdf", "General")
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\n  ")
    with pytest.raises(DocumentLoadError):
        load_document(blank)


def test_whitespace_only_upload_fails_without_indexing(services):
    result = services.knowledge_base.ingest_upload("blank.txt", b"   \n \n", "General")
    assert result["status"] == "failed"
    assert services.vector_db.count() == 0


def test_path_traversal_filename_is_sanitised():
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert "/" not in safe_filename("a/b\\c.pdf")


# 5. Duplicate uploads ------------------------------------------------------------------
def test_duplicate_upload_is_skipped(services, sample_dir):
    kb = services.knowledge_base
    data = (sample_dir / "expense_reimbursement_sop.txt").read_bytes()
    first = kb.ingest_upload("expense_reimbursement_sop.txt", data, "Operations")
    count = services.vector_db.count()
    second = kb.ingest_upload("expense_reimbursement_sop.txt", data, "Operations")
    renamed = kb.ingest_upload("copy_of_expenses.txt", data, "Operations")
    assert first["status"] == "indexed"
    assert second["status"] == "duplicate"
    assert renamed["status"] == "duplicate"
    assert services.vector_db.count() == count
    assert len(kb.list_documents()) == 1


def test_changed_document_with_same_name_replaces_old_version(services):
    kb = services.knowledge_base
    v1 = b"TRAVEL POLICY\nThe hotel limit is $150 per night for all business trips in the region."
    v2 = b"TRAVEL POLICY\nThe hotel limit is $220 per night for all business trips in the region."
    kb.ingest_upload("travel.txt", v1, "Company Policies")
    result = kb.ingest_upload("travel.txt", v2, "Company Policies")
    assert result["status"] == "updated"
    texts = " ".join(c["text"] for c in services.vector_db.get_document_chunks(result["doc_id"]))
    assert "$220" in texts and "$150" not in texts
    assert kb.get_document(result["doc_id"])["version"] == 2


# 6. Deleted documents disappear from retrieval -----------------------------------------
def test_deleted_document_is_not_retrieved(services, sample_dir):
    kb = services.knowledge_base
    for name in ("customer_support_guidelines.txt", "remote_work_policy.docx"):
        kb.ingest_upload(name, (sample_dir / name).read_bytes(), "General")
    question = "What is the first response time for a P1 ticket?"
    assert services.retriever.retrieve(question)[0].document == "customer_support_guidelines.txt"

    doc_id = next(d["doc_id"] for d in kb.list_documents() if d["filename"] == "customer_support_guidelines.txt")
    assert kb.delete_document(doc_id)
    assert services.vector_db.count_for_document(doc_id) == 0
    assert all(c.document != "customer_support_guidelines.txt" for c in services.retriever.retrieve(question, relevance_threshold=0))
    assert not (kb.kb_dir / "customer_support_guidelines.txt").exists()
    assert kb.delete_document(doc_id) is False


def test_reindex_uses_new_chunk_settings(services, sample_dir):
    kb = services.knowledge_base
    res = kb.ingest_upload("employee_handbook.pdf", (sample_dir / "employee_handbook.pdf").read_bytes(), "Employee Handbook")
    before = services.vector_db.count_for_document(res["doc_id"])
    kb.settings = kb.settings.with_overrides(chunk_size=150, chunk_overlap=20)
    out = kb.reindex_document(res["doc_id"])
    assert out["status"] == "reindexed"
    assert services.vector_db.count_for_document(res["doc_id"]) > before
    assert services.vector_db.count() == services.vector_db.count_for_document(res["doc_id"])


def test_sync_folder_indexes_dropped_files_and_prunes_missing(services, sample_dir):
    kb = services.knowledge_base
    (kb.kb_dir / "onboarding_guide.pdf").write_bytes((sample_dir / "onboarding_guide.pdf").read_bytes())
    results = kb.sync_folder()
    assert [r["status"] for r in results] == ["indexed"]
    assert kb.sync_folder() == []  # unchanged files are skipped
    (kb.kb_dir / "onboarding_guide.pdf").unlink()
    assert [r["status"] for r in kb.sync_folder()] == ["removed"]
    assert services.vector_db.count() == 0


def test_vector_store_persists_across_restarts(tmp_path, sample_dir, fake_llm):
    from backend.services import build_services
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    first = build_services(settings, llm=fake_llm)
    first.knowledge_base.ingest_upload("remote_work_policy.docx", (sample_dir / "remote_work_policy.docx").read_bytes())
    count = first.vector_db.count()
    second = build_services(settings, llm=fake_llm)
    assert second.vector_db.count() == count > 0
    assert len(second.knowledge_base.list_documents()) == 1


def test_markdown_sections_and_code_fences(services, tmp_path):
    md = tmp_path / "refund_policy.md"
    md.write_text(
        "# Refund Policy\n\nOwner: Finance.\n\n"
        "## 4. Refunds\nFull refunds are available up to 30 days before departure.\n\n"
        "### Partial refunds\nCancellations 15-29 days before departure receive 50% back.\n\n"
        "```\n# not a heading inside a code fence\n```\n"
    )
    docs = load_document(md)
    sections = [d.metadata["section"] for d in docs]
    assert "4. Refunds" in sections
    assert "4. Refunds > Partial refunds" in sections
    assert not any("not a heading" in s for s in sections)
    assert all(d.metadata["file_type"] == "md" for d in docs)

    result = services.knowledge_base.ingest_upload("refund_policy.md", md.read_bytes(), "Company Policies")
    assert result["status"] == "indexed"
    top = services.retriever.retrieve("What share of the price is refunded for a cancellation 20 days before departure?")[0]
    assert top.document == "refund_policy.md" and top.section == "4. Refunds > Partial refunds"

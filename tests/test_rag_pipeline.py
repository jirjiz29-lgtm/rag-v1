"""RAG behaviour: retrieval accuracy, citations, fallback, multi-document search, conflicts."""
from backend.prompts import FALLBACK_MESSAGE


def docs_of(chunks):
    return {c.document for c in chunks}


# 1. Questions are answered from the right documents -----------------------------------
def test_retrieves_correct_document_for_question(loaded_services):
    cases = {
        "How many days of paid time off do full-time employees get?": "employee_handbook.pdf",
        "What is the first response time for a P1 critical ticket?": "customer_support_guidelines.txt",
        "How many quotes do I need for a purchase over $10,000?": "vendor_management_procedure.docx",
        "How much is the home office stipend?": "remote_work_policy.docx",
        "What should I do in my first week as a new hire?": "onboarding_guide.pdf",
    }
    for question, expected in cases.items():
        chunks = loaded_services.retriever.retrieve(question)
        assert chunks, question
        assert chunks[0].document == expected, (question, [(c.document, round(c.score, 2)) for c in chunks])


def test_answer_is_grounded_in_top_chunk(pipeline, loaded_services):
    result = pipeline.answer("How many days of paid time off do full-time employees get?")
    assert result.found
    assert "20 days" in result.answer
    assert result.sources[0]["document"] == "employee_handbook.pdf"
    # The prompt sent to the model contained the retrieved evidence.
    sent = loaded_services.fake.calls[-1]["messages"][-1]["content"]
    assert "20 days of paid time off" in sent


# 2. Answers carry valid citations ------------------------------------------------------
def test_sources_include_document_page_section_and_excerpt(pipeline):
    result = pipeline.answer("How many days of paid time off do full-time employees get?")
    src = result.sources[0]
    assert src["document"] == "employee_handbook.pdf"
    assert src["page"] == 1
    assert src["section"] == "3. Paid Time Off"
    assert "20 days" in src["excerpt"]
    assert 0 < src["score"] <= 1


def test_fabricated_citations_are_removed(pipeline, loaded_services):
    def responder(sources, question):
        return {
            "answer": f"Employees get 20 days of PTO [{sources[0]['id']}][S99].",
            "found_in_documents": True,
            "cited_sources": [sources[0]["id"], "S99", "S42"],
            "conflicts": [{"description": "made up", "sources": ["S77"]}],
        }

    loaded_services.fake.responder = responder
    result = pipeline.answer("How many PTO days do employees get?")
    assert "[S99]" not in result.answer
    assert [s["source_id"] for s in result.sources] == ["S1"]
    assert result.conflicts == []  # conflict pointed only at a non-existent source


def test_answer_without_valid_citations_falls_back(pipeline, loaded_services):
    loaded_services.fake.responder = lambda s, q: {
        "answer": "Employees get 25 days.", "found_in_documents": True, "cited_sources": [], "conflicts": []
    }
    result = pipeline.answer("How many PTO days do employees get?")
    assert not result.found
    assert result.answer == FALLBACK_MESSAGE
    assert result.sources == []


# 3. Unsupported questions trigger the fallback -----------------------------------------
def test_unrelated_question_returns_fallback_without_calling_llm(pipeline, loaded_services):
    result = pipeline.answer("What is the recipe for a traditional Neapolitan pizza dough?")
    assert not result.found
    assert result.answer == FALLBACK_MESSAGE
    assert loaded_services.fake.calls == []


def test_llm_reporting_not_found_returns_exact_fallback(pipeline, loaded_services):
    loaded_services.fake.responder = lambda s, q: {
        "answer": "The documents do not say.", "found_in_documents": False, "cited_sources": [], "conflicts": []
    }
    result = pipeline.answer("What is the parental leave policy for employees?")
    assert result.answer == FALLBACK_MESSAGE
    assert not result.found


# 4. Multiple documents are searched together -------------------------------------------
def test_multi_document_retrieval(loaded_services):
    chunks = loaded_services.retriever.retrieve(
        "How many days per week can employees work remotely, and what approval is needed?"
    )
    assert {"employee_handbook.pdf", "remote_work_policy.docx"} <= docs_of(chunks)


def test_multi_document_answer_cites_both(pipeline, loaded_services):
    def responder(sources, question):
        ids = [s["id"] for s in sources if s["document"] in {"employee_handbook.pdf", "remote_work_policy.docx"}]
        return {"answer": "Comparison " + "".join(f"[{i}]" for i in ids), "found_in_documents": True,
                "cited_sources": ids, "conflicts": []}

    loaded_services.fake.responder = responder
    result = pipeline.answer("Compare the remote work rules in the handbook and the remote work policy")
    assert {s["document"] for s in result.sources} == {"employee_handbook.pdf", "remote_work_policy.docx"}


def test_metadata_filter_restricts_documents(loaded_services):
    doc_id = next(d["doc_id"] for d in loaded_services.knowledge_base.list_documents()
                  if d["filename"] == "remote_work_policy.docx")
    chunks = loaded_services.retriever.retrieve("remote work days per week", doc_ids=[doc_id])
    assert chunks and docs_of(chunks) == {"remote_work_policy.docx"}


# 7. Conflicting information is surfaced transparently ----------------------------------
def test_conflicting_sources_are_both_retrieved(loaded_services):
    chunks = loaded_services.retriever.retrieve("What is the daily meal allowance when travelling for business?")
    texts = " ".join(c.text for c in chunks)
    assert "$50 per day" in texts and "$75 per day" in texts


def test_conflicts_are_passed_through_with_documents(pipeline, loaded_services):
    def responder(sources, question):
        a = next(s["id"] for s in sources if "$50" in s["text"])
        b = next(s["id"] for s in sources if "$75" in s["text"])
        return {
            "answer": f"The handbook says $50/day [{a}] but the expense SOP says $75/day [{b}].",
            "found_in_documents": True,
            "cited_sources": [a, b],
            "conflicts": [{"description": "Meal allowance differs ($50 vs $75 per day)", "sources": [a, b]}],
        }

    loaded_services.fake.responder = responder
    result = pipeline.answer("What is the daily meal allowance when travelling for business?")
    assert len(result.conflicts) == 1
    assert set(result.conflicts[0]["documents"]) == {"employee_handbook.pdf", "expense_reimbursement_sop.txt"}


def test_system_prompt_enforces_grounding_rules(pipeline, loaded_services):
    pipeline.answer("How many PTO days do employees get?")
    system = loaded_services.fake.calls[-1]["system"]
    assert FALLBACK_MESSAGE in system
    assert "Never invent" in system and "disagree" in system


def test_follow_up_question_uses_history_for_retrieval(pipeline):
    history = [
        {"role": "user", "content": "What is the escalation process for P1 support tickets?"},
        {"role": "assistant", "content": "Page the on-call engineer after 30 minutes [S1]."},
    ]
    result = pipeline.answer("And after 4 hours?", history=history)
    assert any(r["document"] == "customer_support_guidelines.txt" for r in result.retrieved)

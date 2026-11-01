"""Suggested questions: generated from the indexed documents and verified against retrieval."""
from fastapi.testclient import TestClient

from backend.rag_pipeline import LLMError
from main import create_app
from tests.conftest import make_settings


def proposals(_sources, _question):
    return {"questions": [
        {"question": "How many days of paid time off do full-time employees get?", "document": "employee_handbook.pdf"},
        {"question": "What is the escalation process for a P1 ticket?", "document": "customer_support_guidelines.txt"},
        {"question": "How many quotes are needed for purchases over $10,000?", "document": "vendor_management_procedure.docx"},
        {"question": "What is the company's parking policy for visitors?", "document": "employee_handbook.pdf"},  # not answered
        {"question": "How do I submit an expense claim?", "document": "missing_file.pdf"},  # unknown document
        {"question": "Hi?", "document": "employee_handbook.pdf"},  # too short
    ]}


class _Recorder:
    """Wraps the FakeLLM so tests can see whether the suggestion prompt was sent."""

    def __init__(self, fake, responder):
        self.fake, self.responder, self.calls = fake, responder, 0
        self.model = "fake"

    def generate(self, system, messages, schema):
        self.calls += 1
        return self.responder(None, None)


def test_suggestions_are_verified_against_retrieval(loaded_services):
    svc = loaded_services.suggestions
    svc.llm = _Recorder(loaded_services.fake, proposals)
    svc.cache_path.unlink(missing_ok=True)
    result = svc.get(12)
    docs = {q["document"] for q in result["questions"]}
    texts = [q["question"] for q in result["questions"]]
    assert result["source"] == "llm"
    assert "How many days of paid time off do full-time employees get?" in texts
    assert "missing_file.pdf" not in docs
    assert not any("parking" in t for t in texts)
    assert not any(len(t) < 15 for t in texts)
    assert all(q["category"] for q in result["questions"])


def test_suggestions_are_cached_until_documents_change(loaded_services):
    svc = loaded_services.suggestions
    svc.llm = _Recorder(loaded_services.fake, proposals)
    svc.cache_path.unlink(missing_ok=True)
    first = svc.get(6)
    second = svc.get(6)
    assert first == second and svc.llm.calls == 1


def test_suggestions_fall_back_to_headings(loaded_services):
    svc = loaded_services.suggestions

    class Broken:
        def generate(self, *a):
            raise LLMError("unavailable")

    for llm in (None, Broken()):
        svc.llm = llm
        svc.cache_path.unlink(missing_ok=True)
        result = svc.get(12)
        assert result["source"] == "headings"
        assert len(result["questions"]) >= 3
        assert all(q["question"].startswith("What does the ") for q in result["questions"])


def test_suggestions_and_stats_endpoints(tmp_path, fake_llm, sample_dir):
    app = create_app(make_settings(tmp_path), llm=fake_llm, sync_on_startup=False)
    with TestClient(app) as client:
        assert client.get("/suggestions").json() == {"questions": [], "source": "empty"}
        files = [("files", (n, (sample_dir / n).read_bytes(), "application/octet-stream"))
                 for n in ("employee_handbook.pdf", "remote_work_policy.docx")]
        client.post("/documents/upload", files=files, data={"category": "Company Policies"})
        body = client.get("/suggestions?limit=3").json()
        assert body["source"] == "headings"  # the FakeLLM's answer isn't a suggestion list
        assert 1 <= len(body["questions"]) <= 3
        stats = client.get("/stats").json()
        assert stats["by_file_type"] == {"pdf": 1, "docx": 1}
        assert [d["filename"] for d in stats["chunks_by_document"]][0] in {"employee_handbook.pdf", "remote_work_policy.docx"}
        assert stats["avg_chunks_per_document"] > 0 and stats["total_size_bytes"] > 0

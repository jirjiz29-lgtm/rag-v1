"""End-to-end tests of the FastAPI backend (FakeLLM stands in for Claude)."""
import pytest
from fastapi.testclient import TestClient

from backend.prompts import FALLBACK_MESSAGE
from main import create_app
from tests.conftest import make_settings


@pytest.fixture
def client(tmp_path, fake_llm):
    app = create_app(make_settings(tmp_path), llm=fake_llm, sync_on_startup=False)
    with TestClient(app) as c:
        yield c


def upload(client, sample_dir, *names, category="General"):
    files = [("files", (n, (sample_dir / n).read_bytes(), "application/octet-stream")) for n in names]
    return client.post("/documents/upload", files=files, data={"category": category})


def test_health_and_settings_hide_secrets(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/settings").json()
    assert not any("key" in k and k != "api_key_warning" for k in body)
    assert body["supported_extensions"] == [".docx", ".md", ".pdf", ".txt"]


def test_upload_list_chat_delete_flow(client, sample_dir):
    res = upload(client, sample_dir, "employee_handbook.pdf", "customer_support_guidelines.txt", category="Company Policies")
    assert res.status_code == 200
    assert [r["status"] for r in res.json()["results"]] == ["indexed", "indexed"]

    docs = client.get("/documents").json()
    assert len(docs) == 2 and all(d["uploaded_at"] for d in docs)
    stats = client.get("/stats").json()
    assert stats["total_documents"] == 2 and stats["total_chunks"] > 0

    answer = client.post("/chat", json={"question": "How many PTO days do employees get?"}).json()
    assert answer["found"] and answer["sources"][0]["document"] == "employee_handbook.pdf"

    handbook = next(d for d in docs if d["filename"] == "employee_handbook.pdf")
    chunks = client.get(f"/documents/{handbook['doc_id']}/chunks").json()["chunks"]
    assert len(chunks) == handbook["chunks"]

    assert client.post(f"/documents/{handbook['doc_id']}/reindex").json()["status"] == "reindexed"
    assert client.delete(f"/documents/{handbook['doc_id']}").status_code == 200
    assert client.get(f"/documents/{handbook['doc_id']}").status_code == 404
    after = client.post("/chat", json={"question": "How many PTO days do employees get?"}).json()
    assert all(s["document"] != "employee_handbook.pdf" for s in after["retrieved"])


def test_duplicate_upload_via_api(client, sample_dir):
    upload(client, sample_dir, "remote_work_policy.docx")
    res = upload(client, sample_dir, "remote_work_policy.docx")
    assert res.json()["results"][0]["status"] == "duplicate"


def test_unsupported_file_rejected(client):
    res = client.post("/documents/upload", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert res.status_code == 400
    assert "Unsupported file type" in str(res.json()["detail"])


def test_mixed_batch_reports_each_file(client, sample_dir):
    files = [
        ("files", ("ok.txt", (sample_dir / "expense_reimbursement_sop.txt").read_bytes(), "text/plain")),
        ("files", ("bad.csv", b"a,b", "text/csv")),
    ]
    res = client.post("/documents/upload", files=files)
    assert [r["status"] for r in res.json()["results"]] == ["indexed", "rejected"]


def test_unknown_category_rejected(client, sample_dir):
    assert upload(client, sample_dir, "remote_work_policy.docx", category="Nope").status_code == 400


def test_chat_on_empty_kb_returns_fallback(client):
    body = client.post("/chat", json={"question": "What is our parental leave policy?"}).json()
    assert body["answer"] == FALLBACK_MESSAGE and body["sources"] == []


def test_chat_validation(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "hi", "top_k": 500}).status_code == 422


def test_missing_api_key_gives_clear_error(tmp_path, sample_dir):
    app = create_app(make_settings(tmp_path), llm=None, sync_on_startup=False)
    with TestClient(app) as c:
        upload(c, sample_dir, "employee_handbook.pdf")
        res = c.post("/chat", json={"question": "How many PTO days do employees get?"})
        assert res.status_code == 503
        assert "GEMINI_API_KEY" in res.json()["detail"]  # names the active provider's key


def test_secret_value_never_returned(tmp_path):
    from fastapi.testclient import TestClient
    from main import create_app
    from tests.conftest import make_settings

    secret = "sk-ant-test-SECRET-value-123"
    app = create_app(make_settings(tmp_path, anthropic_api_key=secret), sync_on_startup=False)
    with TestClient(app) as c:
        for path in ("/health", "/settings"):
            assert secret not in c.get(path).text

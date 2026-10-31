"""Embedded mode: the dashboard's BackendClient driving the API in-process (as on Streamlit Community Cloud)."""
import pytest
from fastapi.testclient import TestClient

from frontend.api_client import APIError, BackendClient
from main import create_app
from tests.conftest import make_settings


@pytest.fixture
def embedded(tmp_path, fake_llm):
    with TestClient(create_app(make_settings(tmp_path), llm=fake_llm, sync_on_startup=False),
                    raise_server_exceptions=False) as session:
        yield BackendClient("http://embedded", session=session)


def test_embedded_client_upload_chat_and_delete(embedded, sample_dir):
    assert embedded.health()["status"] == "ok"
    files = [(n, (sample_dir / n).read_bytes(), None) for n in ("employee_handbook.pdf", "remote_work_policy.docx")]
    results = embedded.upload(files, "Company Policies")["results"]
    assert [r["status"] for r in results] == ["indexed", "indexed"]

    answer = embedded.chat("How many days of paid time off do full-time employees get?", [])
    assert answer["found"] and answer["sources"][0]["citation"].startswith("employee_handbook.pdf · p. 1")

    doc_id = next(d["doc_id"] for d in embedded.documents() if d["filename"] == "remote_work_policy.docx")
    assert embedded.chunks(doc_id)["chunks"]
    embedded.delete(doc_id)
    assert {d["filename"] for d in embedded.documents()} == {"employee_handbook.pdf"}


def test_embedded_client_surfaces_api_errors(embedded):
    with pytest.raises(APIError, match="not found"):
        embedded.delete("does-not-exist")

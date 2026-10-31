"""Fixed knowledge base: documents come only from the knowledge_base folder and cannot be changed via the API."""
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import DOCUMENT_CATEGORIES, SUPPORTED_EXTENSIONS
from main import create_app
from tests.conftest import make_settings

REPO_KB = Path(__file__).resolve().parent.parent / "knowledge_base"


def fixed_app(tmp_path, fake_llm):
    """An app started like production: read-only, indexing the repo's knowledge_base copy at startup."""
    kb = tmp_path / "kb"
    shutil.copytree(REPO_KB, kb)
    settings = make_settings(tmp_path, kb_read_only=True)
    return TestClient(create_app(settings, llm=fake_llm, sync_on_startup=True)), kb


def test_repo_knowledge_base_is_complete_and_categorised():
    docs = sorted(p.name for p in REPO_KB.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    categories = json.loads((REPO_KB / "categories.json").read_text())
    assert len(docs) == 10
    assert sorted(categories) == docs
    assert set(categories.values()) <= set(DOCUMENT_CATEGORIES)


def test_startup_indexes_every_fixed_document_with_its_category(tmp_path, fake_llm):
    client, _ = fixed_app(tmp_path, fake_llm)
    with client:
        docs = {d["filename"]: d for d in client.get("/documents").json()}
        assert len(docs) == 10 and all(d["status"] == "indexed" for d in docs.values())
        assert docs["05_Customer_Support_SOP.md"]["category"] == "Customer Support"
        assert docs["08_Vendor_Management_SOP.md"]["category"] == "Vendor Management"
        assert client.get("/settings").json()["kb_read_only"] is True

        answer = client.post("/chat", json={"question": "What should be collected when onboarding a new vendor?"}).json()
        assert answer["found"] and answer["sources"][0]["document"] == "08_Vendor_Management_SOP.md"


def test_api_refuses_every_change_to_the_fixed_knowledge_base(tmp_path, fake_llm):
    client, _ = fixed_app(tmp_path, fake_llm)
    with client:
        doc_id = client.get("/documents").json()[0]["doc_id"]
        attempts = [
            client.post("/documents/upload", files=[("files", ("new.txt", b"New policy text " * 10, "text/plain"))]),
            client.delete(f"/documents/{doc_id}"),
            client.post(f"/documents/{doc_id}/reindex"),
            client.post("/documents/reindex"),
            client.post("/documents/sync"),
        ]
        assert [r.status_code for r in attempts] == [403] * 5
        assert "read-only" in attempts[0].json()["detail"]
        assert len(client.get("/documents").json()) == 10
        # Reading is still allowed.
        assert client.get(f"/documents/{doc_id}/chunks").json()["chunks"]


def test_changing_the_folder_updates_the_index_on_restart(tmp_path, fake_llm):
    client, kb = fixed_app(tmp_path, fake_llm)
    with client:
        pass
    (kb / "10_Daily_Operations_Manual.md").unlink()
    profile = kb / "01_Company_Profile.md"
    profile.write_text(profile.read_text() + "\n## 9. Head office\nThe head office is open Monday to Saturday.\n")

    with TestClient(create_app(make_settings(tmp_path, kb_read_only=True), llm=fake_llm)) as restarted:
        docs = {d["filename"]: d for d in restarted.get("/documents").json()}
        assert "10_Daily_Operations_Manual.md" not in docs
        assert docs["01_Company_Profile.md"]["version"] == 2
        assert docs["01_Company_Profile.md"]["category"] == "General"

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import Settings  # noqa: E402
from backend.services import build_services  # noqa: E402
from scripts.generate_sample_docs import generate  # noqa: E402

SOURCE_TAG = re.compile(r'<source id="(S\d+)" document="([^"]+)"[^>]*>\n(.*?)\n</source>', re.S)


class FakeLLM:
    """Deterministic stand-in for the LLM (Gemini or Claude), so the pipeline can be tested offline.

    By default it answers from the first source and cites it. Tests can install a custom
    `responder(sources, question) -> dict` to simulate other model behaviour.
    """

    model = "fake-llm"

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.responder: Optional[Callable[[List[Dict[str, str]], str], Dict[str, Any]]] = None

    def generate(self, system, messages, schema):
        content = messages[-1]["content"]
        sources = [{"id": m[0], "document": m[1], "text": m[2]} for m in SOURCE_TAG.findall(content)]
        question = content.split("<question>")[-1].split("</question>")[0].strip()
        self.calls.append({"system": system, "messages": messages, "sources": sources, "question": question})
        if self.responder:
            return self.responder(sources, question)
        first = sources[0]
        return {
            "answer": f"According to {first['document']}: {first['text'][:200]} [{first['id']}]",
            "found_in_documents": True,
            "cited_sources": [first["id"]],
            "conflicts": [],
        }


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("samples")
    generate(out)
    return out


def make_settings(tmp: Path, **overrides) -> Settings:
    base = dict(
        anthropic_api_key="",
        gemini_api_key="",
        knowledge_base_dir=tmp / "kb",
        vectorstore_dir=tmp / "vs",
        collection_name="test_" + tmp.name.replace("-", "_")[:40],
        chunk_size=200,  # tokens, as in production
        chunk_overlap=30,
        top_k=6,
        relevance_threshold=0.20,
        kb_read_only=False,  # most tests add and remove documents
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(tmp_path, fake_llm):
    """Fresh, empty knowledge base for tests that mutate state."""
    return build_services(make_settings(tmp_path), llm=fake_llm)


@pytest.fixture(scope="session")
def loaded_services(tmp_path_factory, sample_dir):
    """Knowledge base with all sample documents indexed (shared, read-only)."""
    tmp = tmp_path_factory.mktemp("loaded")
    llm = FakeLLM()
    svc = build_services(make_settings(tmp), llm=llm)
    for f in sorted(sample_dir.iterdir()):
        result = svc.knowledge_base.ingest_upload(f.name, f.read_bytes(), "General")
        assert result["status"] == "indexed", result
    svc.fake = llm
    return svc


@pytest.fixture
def pipeline(loaded_services):
    loaded_services.fake.responder = None
    loaded_services.fake.calls.clear()
    return loaded_services.pipeline

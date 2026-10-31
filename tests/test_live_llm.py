"""Live tests against the configured LLM (LLM_PROVIDER: gemini or anthropic).

Skipped unless that provider's API key is set. Run:  pytest tests/test_live_llm.py -v
"""
import pytest

from backend.prompts import FALLBACK_MESSAGE
from backend.services import build_services
from config import get_settings
from tests.conftest import make_settings

pytestmark = pytest.mark.skipif(
    not get_settings().llm_configured,
    reason=f"{get_settings().llm_key_name} not set - skipping live LLM tests",
)


@pytest.fixture(scope="module")
def live(tmp_path_factory, sample_dir):
    env = get_settings()
    settings = make_settings(
        tmp_path_factory.mktemp("live"),
        llm_provider=env.llm_provider,
        gemini_api_key=env.gemini_api_key,
        gemini_model=env.gemini_model,
        gemini_fallback_models=env.gemini_fallback_models,
        anthropic_api_key=env.anthropic_api_key,
        claude_model=env.claude_model,
        claude_effort=env.claude_effort,
    )
    svc = build_services(settings)
    for f in sorted(sample_dir.iterdir()):
        svc.knowledge_base.ingest_upload(f.name, f.read_bytes())
    return svc.pipeline


def test_live_answer_with_citation(live):
    r = live.answer("How many days of paid time off do full-time employees get per year?")
    assert r.found and "20" in r.answer
    assert any(s["document"] == "employee_handbook.pdf" for s in r.sources)


def test_live_step_by_step(live):
    r = live.answer("Give me step-by-step instructions to submit an expense claim.")
    assert r.found and "30 days" in r.answer
    assert any(s["document"] == "expense_reimbursement_sop.txt" for s in r.sources)


def test_live_unsupported_question_falls_back(live):
    r = live.answer("What is the company's parental leave policy?")
    assert r.answer == FALLBACK_MESSAGE


def test_live_conflict_detected(live):
    r = live.answer("What is the daily meal allowance when travelling on business?")
    assert r.found
    assert r.conflicts, r.answer
    assert "50" in r.answer and "75" in r.answer

"""Central configuration for the Company Knowledge Assistant.

All values come from environment variables (loaded from a local `.env` file via
python-dotenv) with safe defaults. Secrets such as the Anthropic API key are only
ever read here, on the backend - the Streamlit frontend never sees them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

DOCUMENT_CATEGORIES = [
    "Company Policies",
    "Employee Handbook",
    "Standard Operating Procedures",
    "Customer Support",
    "Sales",
    "Marketing",
    "Operations",
    "Vendor Management",
    "Onboarding",
    "Training",
    "General",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else BASE_DIR / path


@dataclass(frozen=True)
class Settings:
    # Which LLM answers questions: "gemini" or "anthropic"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").strip().lower())
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 16000))
    llm_timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 60.0))

    # Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""), repr=False)
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    # Tried in order when the main model is overloaded or times out (comma-separated, may be empty).
    gemini_fallback_models: tuple = field(default_factory=lambda: tuple(
        m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite").split(",") if m.strip()
    ))

    # Claude
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""), repr=False)
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5-5"))
    claude_effort: str = field(default_factory=lambda: os.getenv("CLAUDE_EFFORT", "medium"))
    claude_max_tokens: int = field(default_factory=lambda: _env_int("CLAUDE_MAX_TOKENS", 16000))
    claude_timeout_seconds: float = field(default_factory=lambda: _env_float("CLAUDE_TIMEOUT_SECONDS", 120.0))
    # Server-side refusal fallback (Claude API only). Disable on Bedrock/Vertex/Foundry.
    claude_enable_fallbacks: bool = field(default_factory=lambda: _env_bool("CLAUDE_ENABLE_FALLBACKS", True))

    # Embeddings / chunking / retrieval
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    # Minimum cosine relevance (0-1) for a chunk to count as evidence.
    relevance_threshold: float = field(default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.25))
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 6))

    # Storage
    knowledge_base_dir: Path = field(default_factory=lambda: _env_path("KNOWLEDGE_BASE_DIR", BASE_DIR / "knowledge_base"))
    vectorstore_dir: Path = field(default_factory=lambda: _env_path("VECTORSTORE_DIR", BASE_DIR / "vectorstore"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "company_knowledge"))

    # Upload validation
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 25))

    # API server / frontend
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    backend_url: str = field(default_factory=lambda: os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))

    @property
    def registry_path(self) -> Path:
        return self.vectorstore_dir / "document_registry.json"

    @property
    def llm_api_key(self) -> str:
        return self.gemini_api_key if self.llm_provider == "gemini" else self.anthropic_api_key

    @property
    def llm_key_name(self) -> str:
        return "GEMINI_API_KEY" if self.llm_provider == "gemini" else "ANTHROPIC_API_KEY"

    @property
    def llm_model(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.claude_model

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def api_key_warning(self) -> str:
        """Human-readable hint when the active provider's key is missing or clearly the wrong kind."""
        if not self.llm_api_key:
            return f"{self.llm_key_name} is not set."
        if self.llm_provider == "anthropic" and not self.anthropic_api_key.startswith("sk-ant-"):
            return "ANTHROPIC_API_KEY does not look like an Anthropic key (expected it to start with 'sk-ant-')."
        return ""

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.llm_provider not in {"gemini", "anthropic"}:
            raise ValueError("LLM_PROVIDER must be 'gemini' or 'anthropic'")
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE must be at least 100 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1")
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("RELEVANCE_THRESHOLD must be between 0 and 1")
        if self.claude_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("CLAUDE_EFFORT must be one of low, medium, high, xhigh, max")


def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings

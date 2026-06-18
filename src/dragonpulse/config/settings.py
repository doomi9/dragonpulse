"""Application settings for DragonPulse.

All configuration is loaded from environment variables (optionally via a local
``.env`` file) using ``pydantic-settings``. Settings are intentionally
local-first: nothing is sent anywhere unless the user explicitly opts in
(e.g. by enabling the LLM integration).

Usage
-----
>>> from dragonpulse.config import get_settings
>>> settings = get_settings()
>>> settings.active_api_key  # resolves basic/system based on API_KEY_TIER
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repo root: .../dragonpulse  (this file is src/dragonpulse/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Allowed values for the RAG embedding backend selector.
EmbeddingBackend = Literal["auto", "hashing", "ollama", "sentence_transformers"]


class KeyTier(str, Enum):
    """Which SAM.gov API key to use."""

    BASIC = "basic"
    SYSTEM = "system"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from env / ``.env``.

    Every field is prefixed with ``DRAGONPULSE_`` in the environment.
    """

    model_config = SettingsConfigDict(
        env_prefix="DRAGONPULSE_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- SAM.gov keys -------------------------------------------------------
    sam_api_key_basic: Optional[str] = Field(
        default=None, description="Basic personal SAM.gov key (10 req/day tier)."
    )
    sam_api_key_system: Optional[str] = Field(
        default=None, description="Higher-limit system-account key (added later)."
    )
    api_key_tier: KeyTier = Field(
        default=KeyTier.BASIC, description="Which key to use: basic or system."
    )

    # --- Caching ------------------------------------------------------------
    cache_ttl_seconds: int = Field(
        default=43_200, ge=0, description="TTL for cached API responses (seconds)."
    )
    cache_disabled: bool = Field(
        default=False, description="If true, always bypass the disk cache."
    )
    daily_request_budget: int = Field(
        default=9,
        ge=0,
        description="Soft daily live-request guardrail to protect the 10/day key.",
    )

    # --- Search defaults ----------------------------------------------------
    # NoDecode: skip pydantic-settings' JSON parsing so a plain comma-separated
    # env string is accepted; the validator below splits it into a list.
    default_naics: Annotated[List[str], NoDecode] = Field(
        default_factory=list,
        description="Comma-separated NAICS codes pre-selected in the sidebar.",
    )

    # --- Paths --------------------------------------------------------------
    data_dir: Path = Field(default=DEFAULT_DATA_DIR, description="Root data directory.")

    # --- Logging ------------------------------------------------------------
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # --- Optional LLM (opt-in) ---------------------------------------------
    llm_enabled: bool = Field(default=False, description="Master switch for LLM calls.")
    llm_base_url: Optional[str] = Field(
        default=None, description="OpenAI-compatible base URL (blank = OpenAI cloud)."
    )
    llm_api_key: Optional[str] = Field(default=None)
    # Recommended primary model is a strong local model (Llama 3.3 70B, Q3_K_M
    # quant) served via Ollama. Pull it first:
    #   ollama pull llama3.3:70b-instruct-q3_K_M
    llm_model: str = Field(default="llama3.3:70b-instruct-q3_K_M")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    # --- RAG knowledge base -------------------------------------------------
    rag_embedding_backend: EmbeddingBackend = Field(
        default="auto",
        description="Embedding backend: auto | hashing | ollama | sentence_transformers.",
    )
    rag_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Model name for the ollama/sentence_transformers backends.",
    )
    rag_chunk_chars: int = Field(
        default=3600,
        ge=200,
        le=12000,
        description=(
            "Target characters per chunk (~900 tokens at 3600). Larger, more "
            "coherent chunks give a strong model better context."
        ),
    )
    rag_chunk_overlap: int = Field(
        default=400, ge=0, le=2000, description="Character overlap between chunks."
    )
    rag_top_k: int = Field(default=5, ge=1, le=50, description="Default retrieved chunks.")
    kb_summarize: bool = Field(
        default=True,
        description=(
            "Generate a short per-document summary at ingestion (LLM when "
            "available, otherwise a heuristic) to enrich retrieval metadata."
        ),
    )
    kb_max_upload_mb: int = Field(
        default=1000,
        ge=1,
        le=2000,
        description="Maximum Knowledge Base upload size per file, in megabytes.",
    )
    kb_ocr_enabled: bool = Field(
        default=True,
        description="Auto-OCR scanned/image-only PDFs on upload when text extraction finds none.",
    )
    kb_ocr_dpi: int = Field(
        default=200,
        ge=72,
        le=600,
        description="Render DPI for OCR. Higher is more accurate but slower.",
    )

    # ----------------------------------------------------------------------- #
    # Validators
    # ----------------------------------------------------------------------- #
    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, value: object) -> object:
        if isinstance(value, str) and value:
            return Path(value).expanduser()
        return value

    @field_validator("default_naics", mode="before")
    @classmethod
    def _split_naics(cls, value: object) -> object:
        """Accept a comma-separated env string or a real list."""
        if isinstance(value, str):
            return [code.strip() for code in value.split(",") if code.strip()]
        return value

    # ----------------------------------------------------------------------- #
    # Derived properties
    # ----------------------------------------------------------------------- #
    @property
    def active_api_key(self) -> Optional[str]:
        """Return the API key for the currently selected tier (or None)."""
        if self.api_key_tier is KeyTier.SYSTEM:
            return self.sam_api_key_system or self.sam_api_key_basic
        return self.sam_api_key_basic

    @property
    def has_api_key(self) -> bool:
        return bool(self.active_api_key)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def attachments_dir(self) -> Path:
        return self.data_dir / "attachments"

    @property
    def rag_dir(self) -> Path:
        return self.data_dir / "rag"

    @property
    def drafts_dir(self) -> Path:
        return self.data_dir / "drafts"

    @property
    def llm_active(self) -> bool:
        """LLM is used when enabled AND reachable.

        A local OpenAI-compatible server (Ollama/LM Studio/vLLM) is identified by
        ``llm_base_url`` and needs no real key; cloud providers need ``llm_api_key``.
        """
        if not self.llm_enabled:
            return False
        return bool(self.llm_api_key or self.llm_base_url)

    @property
    def llm_is_local(self) -> bool:
        """True when pointing at a local OpenAI-compatible server."""
        return bool(self.llm_base_url)

    @property
    def resolved_llm_api_key(self) -> str:
        """Key to hand the OpenAI SDK (local servers accept any placeholder)."""
        if self.llm_api_key:
            return self.llm_api_key
        return "local-no-key-required"

    @property
    def kb_max_upload_bytes(self) -> int:
        """Maximum Knowledge Base upload size per file, in bytes."""
        return self.kb_max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """Create local data directories if they do not yet exist."""
        for path in (
            self.data_dir,
            self.cache_dir,
            self.attachments_dir,
            self.rag_dir,
            self.drafts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def masked_api_key(self) -> str:
        """Safe-to-log representation of the active key."""
        key = self.active_api_key
        if not key:
            return "<none>"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}…{key[-4:]}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    Cached so the ``.env`` file and environment are read once per process.
    Call ``get_settings.cache_clear()`` in tests to force a reload.
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings

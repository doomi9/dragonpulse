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
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repo root: .../dragonpulse  (this file is src/dragonpulse/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


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
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

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

    def ensure_dirs(self) -> None:
        """Create local data directories if they do not yet exist."""
        for path in (self.data_dir, self.cache_dir, self.attachments_dir):
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

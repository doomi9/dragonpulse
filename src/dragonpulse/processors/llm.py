"""Optional, opt-in LLM wrapper.

DragonPulse is local-first: the LLM is **off by default**. When disabled (or
when the ``openai`` package / credentials are missing), callers receive an
:class:`LLMUnavailable` signal and are expected to fall back to deterministic
templates. This keeps every feature usable with zero external dependencies
while allowing power users to opt in.

The wrapper speaks the OpenAI Chat Completions protocol, which is also spoken
by local servers (Ollama, LM Studio, vLLM) via a custom ``base_url`` — so
"using an LLM" can still mean "fully local".

Grounding contract
------------------
All prompts in DragonPulse pass an explicit ``context`` block and instruct the
model to only use facts from that context and to cite them. The system prompt
here enforces that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings

logger = get_logger(__name__)

_GROUNDING_SYSTEM_PROMPT = (
    "You are DragonPulse, an assistant for U.S. government contractors. "
    "You must ground every statement strictly in the CONTEXT provided by the "
    "user message. Do not invent facts, names, dates, dollar amounts, or "
    "requirements that are not present in the CONTEXT. When you reference a "
    "fact, cite its source label in brackets, e.g. [opportunity metadata] or "
    "[attachment: SOW.pdf]. If the CONTEXT lacks the information needed, say so "
    "explicitly rather than guessing. Keep a professional, concise tone."
)


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call is requested but no LLM is available/enabled."""


@dataclass
class LLMResult:
    """A structured LLM response with provenance for transparency."""

    text: str
    model: str
    used_llm: bool  # False when produced by a deterministic fallback
    sources: List[str]  # source labels the output is grounded in


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = None  # lazily constructed

    @property
    def available(self) -> bool:
        """True only when enabled, credentialed, and the SDK is importable."""
        if not self.settings.llm_active:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            logger.warning("LLM enabled but 'openai' package not installed.")
            return False
        return True

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.available:
            raise LLMUnavailable("LLM is not enabled or not configured.")
        from openai import OpenAI

        kwargs = {"api_key": self.settings.resolved_llm_api_key}
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        *,
        instruction: str,
        context: str,
        sources: List[str],
        max_tokens: int = 700,
    ) -> LLMResult:
        """Run a grounded completion.

        Parameters
        ----------
        instruction:
            What to produce (e.g. "Draft a short outreach email...").
        context:
            The grounding facts. The model is told to use only these.
        sources:
            Human-readable source labels recorded on the result.
        """
        client = self._ensure_client()
        user_message = (
            f"CONTEXT (cite from these only):\n{context}\n\n"
            f"TASK:\n{instruction}"
        )
        logger.info("LLM completion via model=%s (base_url set=%s)",
                    self.settings.llm_model, bool(self.settings.llm_base_url))
        try:
            resp = client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize all SDK errors
            raise LLMUnavailable(f"LLM call failed: {exc}") from exc

        text = (resp.choices[0].message.content or "").strip()
        return LLMResult(
            text=text,
            model=self.settings.llm_model,
            used_llm=True,
            sources=list(sources),
        )

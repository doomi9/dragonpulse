"""Business-logic processors for DragonPulse.

These turn raw opportunity data into actionable, *grounded* outputs:
- :mod:`checklist`   — "What needs to be done" action plans.
- :mod:`outreach`    — "Who to reach out to" personalized email drafts.
- :mod:`attachments` — download + text extraction for resource links.
- :mod:`llm`         — optional, opt-in LLM wrapper with graceful fallback.
"""

from dragonpulse.processors.attachments import (
    AttachmentExtract,
    download_attachment,
    extract_text,
)
from dragonpulse.processors.checklist import ChecklistItem, build_checklist
from dragonpulse.processors.embeddings import get_embedding_backend
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.llm import LLMClient, LLMResult, LLMUnavailable
from dragonpulse.processors.outreach import OutreachDraft, generate_outreach_email
from dragonpulse.processors.pricing import PricingAnalysis, analyze_awards
from dragonpulse.processors.text_extract import (
    UnsupportedDocument,
    chunk_text,
    extract_and_chunk,
    extract_text_from_bytes,
)

__all__ = [
    "LLMClient",
    "LLMResult",
    "LLMUnavailable",
    "ChecklistItem",
    "build_checklist",
    "OutreachDraft",
    "generate_outreach_email",
    "AttachmentExtract",
    "download_attachment",
    "extract_text",
    "PricingAnalysis",
    "analyze_awards",
    "KnowledgeBase",
    "get_embedding_backend",
    "UnsupportedDocument",
    "chunk_text",
    "extract_and_chunk",
    "extract_text_from_bytes",
]

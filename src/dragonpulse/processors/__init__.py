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
from dragonpulse.processors.draft_store import DraftStore
from dragonpulse.processors.embeddings import (
    BackendStatus,
    describe_backend,
    get_embedding_backend,
)
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.llm import LLMClient, LLMResult, LLMUnavailable
from dragonpulse.processors.outreach import OutreachDraft, generate_outreach_email
from dragonpulse.processors.pricing import PricingAnalysis, analyze_awards
from dragonpulse.processors.proposal import (
    SECTION_SPECS,
    ProposalGenerator,
    SectionSpec,
    clean_proposal_content,
    compliance_matrix_to_xlsx_bytes,
    draft_to_docx_bytes,
)
from dragonpulse.processors.recommender import (
    DEFAULT_PICK_CATEGORIES,
    Recommendation,
    RecommendationResult,
    broad_capability_terms,
    generate_queries,
    recommend,
)
from dragonpulse.processors.sam_scrape import (
    SamScrapeError,
    ScrapedOpportunity,
    fetch_opportunity_from_link,
    parse_sam_link,
    search_opportunities_via_frontend,
)
from dragonpulse.processors.text_extract import (
    ChunkPiece,
    ScannedPDFError,
    UnsupportedDocument,
    chunk_text,
    extract_and_chunk,
    extract_text_from_bytes,
    extract_text_with_ocr,
    ocr_available,
    ocr_pdf_bytes,
    semantic_chunks,
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
    "recommend",
    "generate_queries",
    "broad_capability_terms",
    "Recommendation",
    "RecommendationResult",
    "DEFAULT_PICK_CATEGORIES",
    "SamScrapeError",
    "ScrapedOpportunity",
    "fetch_opportunity_from_link",
    "parse_sam_link",
    "search_opportunities_via_frontend",
    "KnowledgeBase",
    "get_embedding_backend",
    "describe_backend",
    "BackendStatus",
    "UnsupportedDocument",
    "ScannedPDFError",
    "ChunkPiece",
    "chunk_text",
    "semantic_chunks",
    "extract_and_chunk",
    "extract_text_from_bytes",
    "extract_text_with_ocr",
    "ocr_available",
    "ocr_pdf_bytes",
    "ProposalGenerator",
    "SectionSpec",
    "SECTION_SPECS",
    "clean_proposal_content",
    "draft_to_docx_bytes",
    "compliance_matrix_to_xlsx_bytes",
    "DraftStore",
]

"""Models for the local RAG knowledge base.

The knowledge base stores *documents* (your past proposals, performance write-ups,
capability statements, etc.) split into *chunks*. Each chunk is embedded into a
vector and persisted on disk. Retrieval returns :class:`RetrievedChunk` objects
that carry enough provenance (document name, chunk ordinal, score) to **cite the
source** of every grounded answer — a core DragonPulse principle.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Suggested document categories (folder-like). Users can also type their own.
DEFAULT_CATEGORIES = [
    "Past Performance",
    "Capabilities",
    "Technical",
    "Management",
    "Pricing",
    "Certifications",
    "Other",
]

# Coarse document *types* inferred at ingestion (distinct from the user-assigned
# category). Used as retrieval metadata and to enrich each chunk's embedding.
DOCUMENT_TYPES = [
    "Proposal",
    "Past Performance",
    "Capability Statement",
    "Technical Document",
    "Statement of Work",
    "Resume",
    "Pricing",
    "Certification",
    "Other",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Document(BaseModel):
    """Metadata for one ingested source document (text is stored per-chunk)."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    source_type: str = "upload"  # upload | attachment | manual
    category: str = "Uncategorized"  # folder-like grouping (user-assignable)
    doc_type: str = "Other"  # inferred kind of document (see DOCUMENT_TYPES)
    summary: str = ""  # short (1-2 sentence) gist; LLM-generated when available
    content_sha: Optional[str] = None
    char_count: int = 0
    chunk_count: int = 0
    added_at: str = Field(default_factory=_now_iso)
    indexed_at: str = Field(default_factory=_now_iso)  # updated on (re)embed
    tags: List[str] = Field(default_factory=list)

    @staticmethod
    def sha_of(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Chunk(BaseModel):
    """A single embedded text chunk belonging to a :class:`Document`."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    doc_id: str
    doc_name: str
    ordinal: int  # position of this chunk within its document (0-based)
    text: str
    # Retrieval metadata carried on each chunk so it can be used for filtering
    # and to build a context-enriched embedding (without polluting the cited text).
    category: str = ""
    doc_type: str = ""
    section: Optional[str] = None  # nearest heading this chunk falls under

    def citation(self) -> str:
        """Human-readable source label, e.g. ``"Past Proposal.pdf #3"``."""
        base = f"{self.doc_name} #{self.ordinal + 1}"
        if self.section:
            return f"{base} · {self.section}"
        return base


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search, with its score."""

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]

    @property
    def citation(self) -> str:
        return self.chunk.citation()


class KBStats(BaseModel):
    """Summary statistics for the knowledge base (for the UI)."""

    documents: int = 0
    chunks: int = 0
    backend: str = "unknown"
    dimension: int = 0
    total_chars: int = 0

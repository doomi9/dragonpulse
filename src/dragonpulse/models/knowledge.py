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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Document(BaseModel):
    """Metadata for one ingested source document (text is stored per-chunk)."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    source_type: str = "upload"  # upload | attachment | manual
    content_sha: Optional[str] = None
    char_count: int = 0
    chunk_count: int = 0
    added_at: str = Field(default_factory=_now_iso)
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

    def citation(self) -> str:
        """Human-readable source label, e.g. ``"Past Proposal.pdf #3"``."""
        return f"{self.doc_name} #{self.ordinal + 1}"


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

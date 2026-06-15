"""Tests for the RAG knowledge base: chunking, embeddings, and the store."""

from __future__ import annotations

import numpy as np
import pytest

from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.processors.embeddings import (
    HashingEmbedding,
    describe_backend,
    get_embedding_backend,
)
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.text_extract import (
    UnsupportedDocument,
    chunk_text,
    extract_text_from_bytes,
)


class _FakeOllamaBackend:
    """Minimal stand-in for OllamaEmbedding (no network) for unit tests."""

    def __init__(self, dimension: int = 64, model: str = "nomic-embed-text") -> None:
        self.name = "ollama"
        self.model = model
        self.dimension = dimension
        self._inner = HashingEmbedding(dimension=dimension)

    def embed(self, texts):
        return self._inner.embed(texts)

    def signature(self) -> str:
        return f"ollama:{self.model}"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunking_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + ("word " * 40) for i in range(20))
    chunks = chunk_text(text, chunk_chars=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 80 for c in chunks)


def test_chunking_hard_splits_giant_paragraph():
    text = "x" * 5000
    chunks = chunk_text(text, chunk_chars=1000, overlap=100)
    assert len(chunks) >= 5


def test_chunking_empty_text_returns_empty():
    assert chunk_text("   \n\n  ") == []


def test_chunk_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_chars=100, overlap=100)


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def test_extract_txt():
    text = extract_text_from_bytes(b"hello world", "notes.txt")
    assert text == "hello world"


def test_extract_unsupported_type():
    with pytest.raises(UnsupportedDocument):
        extract_text_from_bytes(b"data", "image.png")


def test_extract_empty_raises():
    with pytest.raises(UnsupportedDocument):
        extract_text_from_bytes(b"   ", "blank.txt")


# --------------------------------------------------------------------------- #
# Hashing embedding
# --------------------------------------------------------------------------- #
def test_hashing_embedding_is_deterministic_and_normalized():
    emb = HashingEmbedding(dimension=256)
    a = emb.embed(["power line construction for the army"])
    b = emb.embed(["power line construction for the army"])
    assert np.allclose(a, b)  # deterministic across calls
    assert a.shape == (1, 256)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)  # L2-normalized


def test_hashing_similarity_orders_by_relevance():
    emb = HashingEmbedding(dimension=512)
    docs = emb.embed(
        [
            "electrical power line construction and utility poles",  # relevant
            "custom software development in python",  # irrelevant
        ]
    )
    q = emb.embed(["power line construction"])[0]
    scores = docs @ q
    assert scores[0] > scores[1]


def test_empty_embed_returns_empty_matrix():
    emb = HashingEmbedding(dimension=64)
    out = emb.embed([])
    assert out.shape == (0, 64)


# --------------------------------------------------------------------------- #
# KnowledgeBase
# --------------------------------------------------------------------------- #
def _kb(tmp_path) -> KnowledgeBase:
    settings = Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        rag_chunk_chars=300,
        rag_chunk_overlap=50,
        rag_embedding_backend="hashing",
    )
    return KnowledgeBase(settings=settings, backend=HashingEmbedding(dimension=256))


def test_kb_add_search_and_cite(tmp_path):
    kb = _kb(tmp_path)
    kb.add_document(
        "Power Proposal.txt",
        "We constructed high-voltage power lines and communication towers for the Army. "
        "Our crews delivered the utility infrastructure on schedule.",
    )
    kb.add_document(
        "Software Proposal.txt",
        "We built a custom Python web application with a React front end and REST APIs.",
    )
    stats = kb.stats()
    assert stats.documents == 2
    assert stats.chunks >= 2

    hits = kb.search("power line and utility construction", k=1)
    assert hits
    assert "Power Proposal.txt" in hits[0].citation
    assert hits[0].score > 0


def test_kb_persists_across_instances(tmp_path):
    kb = _kb(tmp_path)
    kb.add_document("Doc.txt", "engineering services for federal facilities " * 10)
    # New instance over the same dir should load the persisted store.
    kb2 = KnowledgeBase(
        settings=kb.settings, backend=HashingEmbedding(dimension=256)
    )
    assert kb2.stats().documents == 1
    assert kb2.search("engineering services", k=1)


def test_kb_delete_document(tmp_path):
    kb = _kb(tmp_path)
    doc = kb.add_document("Doc.txt", "logistics support services for the navy " * 8)
    assert kb.stats().chunks > 0
    assert kb.delete_document(doc.doc_id) is True
    assert kb.stats().documents == 0
    assert kb.stats().chunks == 0
    assert kb.delete_document("nonexistent") is False


def test_kb_skips_duplicates(tmp_path):
    kb = _kb(tmp_path)
    text = "identical content for dedup test " * 12
    d1 = kb.add_document("A.txt", text)
    d2 = kb.add_document("B.txt", text)  # same content -> skipped
    assert d1.doc_id == d2.doc_id
    assert kb.stats().documents == 1


def test_kb_category_and_metadata(tmp_path):
    kb = _kb(tmp_path)
    doc = kb.add_document(
        "PastPerf.txt",
        "power transmission work for the army corps " * 8,
        category="Past Performance",
        tags=["army", "power"],
    )
    assert doc.category == "Past Performance"
    assert doc.tags == ["army", "power"]
    assert doc.indexed_at  # set on index
    assert "Past Performance" in kb.categories()


def test_kb_update_document_category_and_tags(tmp_path):
    kb = _kb(tmp_path)
    doc = kb.add_document("Doc.txt", "engineering services " * 10)
    assert doc.category == "Uncategorized"
    assert kb.update_document(doc.doc_id, category="Technical", tags=["idiq"]) is True
    refreshed = next(d for d in kb.list_documents() if d.doc_id == doc.doc_id)
    assert refreshed.category == "Technical"
    assert refreshed.tags == ["idiq"]
    assert kb.update_document("missing", category="X") is False


def test_kb_reindex_updates_indexed_at(tmp_path):
    kb = _kb(tmp_path)
    doc = kb.add_document("Doc.txt", "substation upgrades " * 10)
    original = doc.indexed_at
    import time as _t

    _t.sleep(1.1)  # timestamps are second-resolution
    kb.reindex()
    refreshed = next(d for d in kb.list_documents() if d.doc_id == doc.doc_id)
    assert refreshed.indexed_at >= original


def test_kb_reindexes_on_backend_change(tmp_path):
    kb = _kb(tmp_path)
    kb.add_document("Doc.txt", "grid modernization and substation upgrades " * 10)
    # Reopen with a different-dimension backend -> signature mismatch -> reindex.
    kb2 = KnowledgeBase(
        settings=kb.settings, backend=HashingEmbedding(dimension=128)
    )
    assert kb2.stats().dimension == 128
    assert kb2._vectors.shape[1] == 128
    assert kb2.search("substation upgrades", k=1)


def test_get_embedding_backend_defaults_to_hashing(tmp_path):
    settings = Settings(data_dir=tmp_path, rag_embedding_backend="hashing")
    backend = get_embedding_backend(settings)
    assert backend.name == "hashing"


# --------------------------------------------------------------------------- #
# Semantic upgrade: reindex() and backend status
# --------------------------------------------------------------------------- #
def test_reindex_switches_backend_and_preserves_docs(tmp_path):
    kb = _kb(tmp_path)
    kb.add_document(
        "Power.txt",
        "high voltage power transmission lines and substations for the army " * 6,
    )
    kb.add_document(
        "Software.txt", "custom python web application and rest apis " * 6
    )
    assert kb.stats().documents == 2
    assert kb.backend.name == "hashing"

    # Simulate switching to semantic (Ollama-like) embeddings in-session.
    sig = kb.reindex(_FakeOllamaBackend(dimension=64))
    assert sig == "ollama:nomic-embed-text"
    assert kb.backend.name == "ollama"
    assert kb.stats().dimension == 64
    assert kb._vectors.shape == (kb.stats().chunks, 64)

    # Documents/chunks are preserved and search still finds the right doc.
    assert kb.stats().documents == 2
    hits = kb.search("power line construction for the army", k=1)
    assert hits and "Power.txt" in hits[0].citation


def test_reindex_persists_new_backend(tmp_path):
    kb = _kb(tmp_path)
    kb.add_document("Doc.txt", "grid modernization and substation upgrades " * 8)
    kb.reindex(_FakeOllamaBackend(dimension=64))

    # Reopen with the same (ollama-like) backend -> signature matches, vectors load.
    kb2 = KnowledgeBase(
        settings=kb.settings, backend=_FakeOllamaBackend(dimension=64)
    )
    assert kb2.stats().documents == 1
    assert kb2.stats().dimension == 64
    assert kb2.search("substation upgrades", k=1)


def test_describe_backend_semantic_ollama(tmp_path):
    settings = Settings(data_dir=tmp_path, rag_embedding_backend="auto")
    status = describe_backend(_FakeOllamaBackend(), settings)
    assert status.is_semantic is True
    assert status.fell_back is False
    assert "Ollama" in status.headline


def test_describe_backend_lexical_default(tmp_path):
    # No base URL, lexical chosen on purpose -> plain info, not a fallback.
    settings = Settings(data_dir=tmp_path, rag_embedding_backend="hashing")
    status = describe_backend(HashingEmbedding(64), settings)
    assert status.is_semantic is False
    assert status.fell_back is False
    assert "lexical" in status.headline.lower()


def test_describe_backend_flags_fallback(tmp_path):
    # Wanted Ollama (auto + base URL) but ended up on hashing -> fell_back warning.
    settings = Settings(
        data_dir=tmp_path,
        rag_embedding_backend="auto",
        llm_base_url="http://localhost:11434/v1",
    )
    status = describe_backend(HashingEmbedding(64), settings)
    assert status.is_semantic is False
    assert status.fell_back is True


def test_auto_backend_prefers_ollama_when_base_url_set(tmp_path, monkeypatch):
    """With auto + a base URL, the factory attempts Ollama (then falls back)."""
    settings = Settings(
        data_dir=tmp_path,
        rag_embedding_backend="auto",
        llm_base_url="http://localhost:11434/v1",
    )

    # Force the Ollama probe to succeed by stubbing the class.
    import dragonpulse.processors.embeddings as emb

    monkeypatch.setattr(emb, "OllamaEmbedding", lambda base_url, model: _FakeOllamaBackend())
    backend = get_embedding_backend(settings)
    assert backend.name == "ollama"

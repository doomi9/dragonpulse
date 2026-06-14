"""Tests for the RAG knowledge base: chunking, embeddings, and the store."""

from __future__ import annotations

import numpy as np
import pytest

from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.processors.embeddings import HashingEmbedding, get_embedding_backend
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.text_extract import (
    UnsupportedDocument,
    chunk_text,
    extract_text_from_bytes,
)


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

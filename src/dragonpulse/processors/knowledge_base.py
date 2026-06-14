"""Local, on-disk RAG knowledge base.

A deliberately simple and auditable vector store:

- ``vectors.npy``   — float32 matrix of L2-normalized chunk embeddings.
- ``chunks.json``   — parallel list of chunk metadata + text (same row order).
- ``documents.json``— per-document metadata.
- ``index.json``    — store-level metadata, incl. the embedding signature.

Because the matrix is L2-normalized, cosine similarity is a single dot product,
which is fast and exact for the scale of a contractor's proposal library
(hundreds–thousands of chunks). No cloud, no external vector DB.

If the configured embedding backend changes (different signature), the store
transparently **reindexes** from the stored chunk texts on next load.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings
from dragonpulse.models.knowledge import Chunk, Document, KBStats, RetrievedChunk
from dragonpulse.processors.embeddings import EmbeddingBackend, get_embedding_backend
from dragonpulse.processors.text_extract import chunk_text

logger = get_logger(__name__)


class KnowledgeBase:
    """A persistent, local vector store over document chunks."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.backend = backend or get_embedding_backend(self.settings)
        self.dir = self.settings.rag_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._documents: List[Document] = []
        self._chunks: List[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, self.backend.dimension), dtype=np.float32)

        self._load()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def _vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    @property
    def _chunks_path(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def _documents_path(self) -> Path:
        return self.dir / "documents.json"

    @property
    def _index_path(self) -> Path:
        return self.dir / "index.json"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self._chunks_path.exists():
            return
        try:
            self._documents = [
                Document.model_validate(d)
                for d in json.loads(self._documents_path.read_text("utf-8"))
            ]
            self._chunks = [
                Chunk.model_validate(c)
                for c in json.loads(self._chunks_path.read_text("utf-8"))
            ]
            index_meta = json.loads(self._index_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Knowledge base index unreadable (%s); starting empty.", exc)
            self._documents, self._chunks = [], []
            return

        stored_sig = index_meta.get("signature")
        if stored_sig != self.backend.signature():
            logger.info(
                "Embedding backend changed (%s -> %s); reindexing knowledge base.",
                stored_sig,
                self.backend.signature(),
            )
            self._reindex_from_chunks()
            return

        if self._vectors_path.exists():
            self._vectors = np.load(self._vectors_path)
        else:
            self._reindex_from_chunks()

    def _save(self) -> None:
        np.save(self._vectors_path, self._vectors)
        self._chunks_path.write_text(
            json.dumps([c.model_dump() for c in self._chunks], indent=2), "utf-8"
        )
        self._documents_path.write_text(
            json.dumps([d.model_dump() for d in self._documents], indent=2), "utf-8"
        )
        self._index_path.write_text(
            json.dumps(
                {
                    "signature": self.backend.signature(),
                    "dimension": self.backend.dimension,
                    "documents": len(self._documents),
                    "chunks": len(self._chunks),
                },
                indent=2,
            ),
            "utf-8",
        )

    def _reindex_from_chunks(self) -> None:
        """Recompute all vectors from stored chunk texts (e.g. backend change)."""
        if not self._chunks:
            self._vectors = np.zeros((0, self.backend.dimension), dtype=np.float32)
        else:
            self._vectors = self.backend.embed([c.text for c in self._chunks])
        self._save()

    def reindex(self, backend: Optional[EmbeddingBackend] = None) -> str:
        """Re-embed every stored chunk, optionally switching the embedding backend.

        This is how DragonPulse upgrades an existing library from lexical to
        semantic embeddings (or vice-versa) without re-uploading documents: the
        original chunk *text* is retained, so only the vectors are rebuilt.

        Parameters
        ----------
        backend:
            New backend to adopt. If ``None``, the current backend is re-run
            (useful to refresh after a previously-unavailable server comes up).

        Returns
        -------
        str
            The signature of the backend now in use (e.g. ``"ollama:nomic-embed-text"``).
        """
        with self._lock:
            if backend is not None:
                self.backend = backend
            self._reindex_from_chunks()
        logger.info(
            "Reindexed %d chunks across %d documents using %s.",
            len(self._chunks),
            len(self._documents),
            self.backend.signature(),
        )
        return self.backend.signature()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_document(
        self,
        name: str,
        text: str,
        *,
        source_type: str = "upload",
        tags: Optional[List[str]] = None,
        skip_if_duplicate: bool = True,
    ) -> Document:
        """Chunk, embed, and persist a document. Returns its :class:`Document`.

        If ``skip_if_duplicate`` and a document with identical content already
        exists, the existing record is returned without re-indexing.
        """
        sha = Document.sha_of(text)
        if skip_if_duplicate:
            for existing in self._documents:
                if existing.content_sha == sha:
                    logger.info("Skipping duplicate document '%s'.", name)
                    return existing

        pieces = chunk_text(
            text,
            chunk_chars=self.settings.rag_chunk_chars,
            overlap=self.settings.rag_chunk_overlap,
        )
        if not pieces:
            raise ValueError(f"Document '{name}' produced no chunks.")

        doc = Document(
            name=name,
            source_type=source_type,
            content_sha=sha,
            char_count=len(text),
            chunk_count=len(pieces),
            tags=tags or [],
        )
        new_chunks = [
            Chunk(doc_id=doc.doc_id, doc_name=name, ordinal=i, text=piece)
            for i, piece in enumerate(pieces)
        ]
        new_vectors = self.backend.embed([c.text for c in new_chunks])

        with self._lock:
            self._documents.append(doc)
            self._chunks.extend(new_chunks)
            self._vectors = (
                new_vectors
                if self._vectors.size == 0
                else np.vstack([self._vectors, new_vectors])
            )
            self._save()
        logger.info("Indexed '%s': %d chunks (%d chars).", name, len(pieces), len(text))
        return doc

    def search(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Return the top-``k`` chunks most similar to ``query`` (cited)."""
        k = k or self.settings.rag_top_k
        if not query.strip() or self._vectors.shape[0] == 0:
            return []
        q = self.backend.embed([query])[0]  # already L2-normalized
        scores = self._vectors @ q  # cosine similarity (vectors are normalized)
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks/vectors. Returns True if found."""
        with self._lock:
            if not any(d.doc_id == doc_id for d in self._documents):
                return False
            keep_mask = np.array(
                [c.doc_id != doc_id for c in self._chunks], dtype=bool
            )
            self._chunks = [c for c, keep in zip(self._chunks, keep_mask) if keep]
            self._documents = [d for d in self._documents if d.doc_id != doc_id]
            self._vectors = (
                self._vectors[keep_mask]
                if self._vectors.size
                else self._vectors
            )
            self._save()
        logger.info("Deleted document %s from knowledge base.", doc_id)
        return True

    def clear(self) -> None:
        """Delete the entire knowledge base."""
        with self._lock:
            self._documents, self._chunks = [], []
            self._vectors = np.zeros((0, self.backend.dimension), dtype=np.float32)
            self._save()

    def list_documents(self) -> List[Document]:
        return list(self._documents)

    def stats(self) -> KBStats:
        return KBStats(
            documents=len(self._documents),
            chunks=len(self._chunks),
            backend=self.backend.signature(),
            dimension=self.backend.dimension,
            total_chars=sum(d.char_count for d in self._documents),
        )

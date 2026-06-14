"""Pluggable, local-first text embedding backends.

DragonPulse must work offline with zero heavy dependencies, but also let power
users opt into higher-quality semantic embeddings. Three backends are provided:

1. :class:`HashingEmbedding` (default) — a pure-NumPy feature-hashing bag-of-words
   embedding. No model download, fully deterministic and offline. Gives solid
   *lexical* retrieval (great for matching solicitation language to past
   proposals) and is completely auditable.
2. :class:`OllamaEmbedding` — calls a local Ollama server's ``/api/embeddings``
   (e.g. ``nomic-embed-text``). True semantic embeddings, still 100% local.
3. :class:`SentenceTransformerEmbedding` — uses ``sentence-transformers`` if it
   is installed. Semantic, local, but a heavy dependency (torch).

:func:`get_embedding_backend` selects one based on settings, falling back to the
hashing backend if a richer backend is unavailable, so the app never breaks.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Optional, Protocol

import numpy as np

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# A small English stopword set keeps lexical vectors focused on content terms.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "are", "be", "as", "at", "by", "this", "that", "it", "from", "will", "shall",
    "we", "our", "their", "they", "i", "you", "he", "she", "but", "not", "can",
}


class EmbeddingBackend(Protocol):
    """Protocol every embedding backend implements."""

    name: str
    dimension: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an L2-normalized ``(len(texts), dimension)`` float32 matrix."""
        ...

    def signature(self) -> str:
        """Stable identifier (name + model + dim) used to detect index mismatch."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1) Hashing backend (default, zero-dependency)
# --------------------------------------------------------------------------- #
class HashingEmbedding:
    """Deterministic feature-hashing bag-of-words embedding (pure NumPy).

    Uses a stable hash (md5) so vectors are identical across processes — unlike
    Python's builtin ``hash``, which is salted per run. Term frequencies are
    sublinearly scaled and the vector is L2-normalized, so a dot product equals
    cosine similarity.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self.name = "hashing"
        self.dimension = dimension

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [
            t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
            if len(t) > 1 and t not in _STOPWORDS
        ]

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        counts: dict = {}
        for tok in self._tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        return vec

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts])
        return _l2_normalize(matrix)

    def signature(self) -> str:
        return f"hashing:{self.dimension}"


# --------------------------------------------------------------------------- #
# 2) Ollama backend (local semantic embeddings)
# --------------------------------------------------------------------------- #
class OllamaEmbedding:
    """Embeddings from a local Ollama server (``/api/embeddings``)."""

    def __init__(self, base_url: str, model: str) -> None:
        self.name = "ollama"
        self.model = model
        # Accept an OpenAI-style ".../v1" base_url or a bare host.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self.endpoint = f"{root}/api/embeddings"
        self._dimension = 0  # discovered on first call

    @property
    def dimension(self) -> int:
        if self._dimension == 0:
            self.embed(["dimension probe"])
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        import requests

        vectors: List[List[float]] = []
        for text in texts:
            resp = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.size:
            self._dimension = matrix.shape[1]
        return _l2_normalize(matrix)

    def signature(self) -> str:
        return f"ollama:{self.model}"


# --------------------------------------------------------------------------- #
# 3) sentence-transformers backend (optional heavy dependency)
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedding:
    """Embeddings via the ``sentence-transformers`` package (if installed)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = "sentence_transformers"
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )
        return matrix

    def signature(self) -> str:
        return f"sentence_transformers:{self.model_name}"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_embedding_backend(settings: Optional[Settings] = None) -> EmbeddingBackend:
    """Build the configured backend, falling back to hashing on any failure.

    - ``hashing``: always available, zero deps.
    - ``ollama``: requires ``llm_base_url`` (the local server) to be set.
    - ``sentence_transformers``: requires the package to be importable.
    - ``auto``: prefer Ollama if a local server is configured, else hashing.
    """
    settings = settings or get_settings()
    choice = settings.rag_embedding_backend

    if choice == "auto":
        choice = "ollama" if settings.llm_base_url else "hashing"

    if choice == "ollama":
        if not settings.llm_base_url:
            logger.warning("Ollama embeddings need DRAGONPULSE_LLM_BASE_URL; using hashing.")
            return HashingEmbedding()
        try:
            backend = OllamaEmbedding(settings.llm_base_url, settings.rag_embedding_model)
            _ = backend.dimension  # probe so failures surface here, not later
            logger.info("Using Ollama embeddings (%s)", settings.rag_embedding_model)
            return backend
        except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
            logger.warning("Ollama embeddings unavailable (%s); using hashing.", exc)
            return HashingEmbedding()

    if choice == "sentence_transformers":
        try:
            backend = SentenceTransformerEmbedding(settings.rag_embedding_model)
            logger.info("Using sentence-transformers (%s)", settings.rag_embedding_model)
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("sentence-transformers unavailable (%s); using hashing.", exc)
            return HashingEmbedding()

    return HashingEmbedding()

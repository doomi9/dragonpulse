"""Disk-based response cache.

Each cache entry is a single JSON file named by a deterministic hash of the
request (endpoint + sorted params, with secrets excluded). The file stores the
payload plus metadata (timestamp, TTL, human-readable key) so we can inspect
the cache by hand and reason about freshness.

This is deliberately simple and dependency-free so it is easy to audit — an
important property for a local-first government-contracting tool.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dragonpulse.config.logging_config import get_logger

logger = get_logger(__name__)

# Param names that must never participate in the cache key or be written to disk.
_SENSITIVE_PARAMS = {"api_key", "apikey", "key", "token"}


@dataclass
class CacheEntry:
    """A single cached response and its metadata."""

    key: str
    created_at: float
    ttl_seconds: int
    payload: Any
    meta: Dict[str, Any]

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    def is_fresh(self, now: Optional[float] = None) -> bool:
        """Return True if the entry has not exceeded its TTL.

        A ``ttl_seconds`` of 0 means "never fresh" (always re-fetch); a negative
        TTL is treated as "never expires" for offline/replay scenarios.
        """
        now = time.time() if now is None else now
        if self.ttl_seconds < 0:
            return True
        if self.ttl_seconds == 0:
            return False
        return now < self.expires_at

    def to_json(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "created_at": self.created_at,
            "created_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.created_at)
            ),
            "ttl_seconds": self.ttl_seconds,
            "meta": self.meta,
            "payload": self.payload,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(
            key=data["key"],
            created_at=float(data["created_at"]),
            ttl_seconds=int(data["ttl_seconds"]),
            payload=data["payload"],
            meta=data.get("meta", {}),
        )


def _scrub(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop sensitive params (e.g. API keys) before hashing/persisting."""
    return {
        k: v
        for k, v in params.items()
        if k.lower() not in _SENSITIVE_PARAMS and v is not None
    }


def make_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """Build a stable, secret-free cache key for an endpoint + params.

    The key is a SHA-256 hash of the endpoint and the JSON-serialized, sorted,
    scrubbed params. Sorting makes the key order-independent.
    """
    scrubbed = _scrub(params)
    canonical = json.dumps(
        {"endpoint": endpoint, "params": scrubbed},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest


class DiskCache:
    """A minimal, auditable JSON file cache with TTL semantics."""

    def __init__(
        self,
        cache_dir: Path,
        default_ttl_seconds: int = 43_200,
        disabled: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.default_ttl_seconds = default_ttl_seconds
        self.disabled = disabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get(
        self, endpoint: str, params: Dict[str, Any], ttl_seconds: Optional[int] = None
    ) -> Optional[CacheEntry]:
        """Return a fresh cache entry for the request, or ``None``.

        ``None`` is returned when caching is disabled, the file is missing,
        unreadable, or stale.
        """
        if self.disabled:
            return None
        key = make_cache_key(endpoint, params)
        path = self._path_for(key)
        if not path.exists():
            logger.debug("Cache miss (absent): %s", key[:12])
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry.from_json(data)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Corrupt cache file %s removed: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return None

        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        entry.ttl_seconds = ttl  # honor caller/current TTL on read
        if not entry.is_fresh():
            logger.debug("Cache stale (age=%.0fs > ttl=%ds): %s", entry.age_seconds, ttl, key[:12])
            return None
        logger.debug("Cache hit (age=%.0fs): %s", entry.age_seconds, key[:12])
        return entry

    def set(
        self,
        endpoint: str,
        params: Dict[str, Any],
        payload: Any,
        ttl_seconds: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """Persist a payload for the request and return the written entry."""
        key = make_cache_key(endpoint, params)
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        entry = CacheEntry(
            key=key,
            created_at=time.time(),
            ttl_seconds=ttl,
            payload=payload,
            meta={
                "endpoint": endpoint,
                "params": _scrub(params),
                **(meta or {}),
            },
        )
        path = self._path_for(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry.to_json(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX
        logger.debug("Cache write: %s (ttl=%ds)", key[:12], ttl)
        return entry

    def invalidate(self, endpoint: str, params: Dict[str, Any]) -> bool:
        """Remove a specific cached entry. Returns True if a file was deleted."""
        path = self._path_for(make_cache_key(endpoint, params))
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Delete all cache files. Returns the number removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cleared %d cache files from %s", count, self.cache_dir)
        return count

    def stats(self) -> Dict[str, Any]:
        """Return basic cache statistics for display in the UI."""
        files = list(self.cache_dir.glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in files)
        fresh = 0
        for f in files:
            try:
                entry = CacheEntry.from_json(json.loads(f.read_text(encoding="utf-8")))
                if entry.is_fresh():
                    fresh += 1
            except Exception:  # noqa: BLE001 - stats must never crash the UI
                continue
        return {
            "files": len(files),
            "fresh": fresh,
            "stale": len(files) - fresh,
            "bytes": total_bytes,
            "dir": str(self.cache_dir),
        }

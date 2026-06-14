"""Tests for the disk cache and request budget."""

from __future__ import annotations

from dragonpulse.cache.disk_cache import DiskCache, make_cache_key
from dragonpulse.cache.request_budget import RequestBudget, RequestBudgetExceeded


def test_cache_key_is_secret_free_and_order_independent():
    k1 = make_cache_key("/x", {"b": 2, "a": 1, "api_key": "SECRET"})
    k2 = make_cache_key("/x", {"a": 1, "b": 2, "api_key": "OTHER"})
    # API key excluded -> same key regardless of secret or param order.
    assert k1 == k2
    assert "SECRET" not in k1


def test_cache_set_get_roundtrip(tmp_path):
    cache = DiskCache(tmp_path, default_ttl_seconds=100)
    cache.set("/ep", {"q": "hi"}, {"hello": "world"})
    entry = cache.get("/ep", {"q": "hi"})
    assert entry is not None
    assert entry.payload == {"hello": "world"}


def test_cache_respects_ttl(tmp_path):
    cache = DiskCache(tmp_path, default_ttl_seconds=100)
    cache.set("/ep", {"q": "hi"}, {"v": 1})
    # TTL of 0 means never fresh -> miss.
    assert cache.get("/ep", {"q": "hi"}, ttl_seconds=0) is None
    # Negative TTL means never expires -> hit.
    assert cache.get("/ep", {"q": "hi"}, ttl_seconds=-1) is not None


def test_cache_disabled_returns_none(tmp_path):
    cache = DiskCache(tmp_path, disabled=True)
    cache.set("/ep", {}, {"v": 1})
    assert cache.get("/ep", {}) is None


def test_cache_does_not_persist_secrets(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("/ep", {"api_key": "TOPSECRET", "q": "x"}, {"v": 1})
    # No file in the cache dir should contain the secret.
    for f in tmp_path.glob("*.json"):
        assert "TOPSECRET" not in f.read_text()


def test_cache_stats_and_clear(tmp_path):
    cache = DiskCache(tmp_path, default_ttl_seconds=100)
    cache.set("/a", {}, {"v": 1})
    cache.set("/b", {}, {"v": 2})
    stats = cache.stats()
    assert stats["files"] == 2
    assert stats["fresh"] == 2
    assert cache.clear() == 2
    assert cache.stats()["files"] == 0


def test_request_budget_enforced(tmp_path):
    budget = RequestBudget(tmp_path, daily_budget=2)
    assert budget.remaining() == 2
    budget.record()
    budget.record()
    assert budget.remaining() == 0
    try:
        budget.check()
        assert False, "expected RequestBudgetExceeded"
    except RequestBudgetExceeded:
        pass


def test_request_budget_persists(tmp_path):
    b1 = RequestBudget(tmp_path, daily_budget=5)
    b1.record(3)
    b2 = RequestBudget(tmp_path, daily_budget=5)  # reload from disk
    assert b2.used_today() == 3

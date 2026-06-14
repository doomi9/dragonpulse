"""Tests for the SAM API client (mocked transport) + an opt-in live test."""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from dragonpulse.api.base import (
    SamAuthError,
    SamClient,
    SamRateLimitError,
)
from dragonpulse.api.opportunities import OpportunitiesClient
from dragonpulse.cache.disk_cache import DiskCache
from dragonpulse.cache.request_budget import RequestBudget
from dragonpulse.config.settings import KeyTier, Settings


class FakeResponse:
    def __init__(self, status_code: int, json_body: Any = None, text: str = "", headers=None):
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self.reason = "Reason"
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Records calls and returns a queued response."""

    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = 0
        self.last_params: Dict[str, Any] = {}
        self.headers: Dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        self.last_params = params or {}
        return self.response


def _settings(tmp_path) -> Settings:
    return Settings(
        sam_api_key_basic="TESTKEY",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        cache_ttl_seconds=1000,
        daily_request_budget=5,
    )


def _client(tmp_path, response: FakeResponse) -> SamClient:
    settings = _settings(tmp_path)
    return SamClient(
        settings=settings,
        cache=DiskCache(settings.cache_dir, settings.cache_ttl_seconds),
        budget=RequestBudget(settings.cache_dir, settings.daily_request_budget),
        session=FakeSession(response),
    )


def test_successful_request_is_cached(tmp_path, sample_opportunity_payload):
    client = _client(tmp_path, FakeResponse(200, sample_opportunity_payload))
    payload, from_cache = client.get_json("/opportunities/v2/search", {"q": "x"})
    assert from_cache is False
    assert payload["totalRecords"] == 1
    assert client.budget.used_today() == 1

    # Second call hits cache; no new network call, budget unchanged.
    payload2, from_cache2 = client.get_json("/opportunities/v2/search", {"q": "x"})
    assert from_cache2 is True
    assert client.session.calls == 1
    assert client.budget.used_today() == 1


def test_api_key_injected_but_not_cached(tmp_path, sample_opportunity_payload):
    client = _client(tmp_path, FakeResponse(200, sample_opportunity_payload))
    client.get_json("/opportunities/v2/search", {"q": "x"})
    assert client.session.last_params.get("api_key") == "TESTKEY"
    # Ensure the key is not written to any cache file.
    for f in client.settings.cache_dir.glob("*.json"):
        assert "TESTKEY" not in f.read_text()


def test_auth_error(tmp_path):
    client = _client(tmp_path, FakeResponse(403, {"error": "forbidden"}))
    with pytest.raises(SamAuthError):
        client.get_json("/opportunities/v2/search", {"q": "x"})


def test_rate_limit_error_429(tmp_path):
    client = _client(tmp_path, FakeResponse(429, {"message": "too many"}))
    with pytest.raises(SamRateLimitError):
        client.get_json("/opportunities/v2/search", {"q": "x"})


def test_rate_limit_detected_in_body(tmp_path):
    client = _client(
        tmp_path, FakeResponse(400, {"error": "You have exceeded your rate limit"})
    )
    with pytest.raises(SamRateLimitError):
        client.get_json("/opportunities/v2/search", {"q": "x"})


def test_missing_key_raises(tmp_path):
    settings = _settings(tmp_path)
    settings.sam_api_key_basic = None
    client = SamClient(
        settings=settings,
        cache=DiskCache(settings.cache_dir),
        budget=RequestBudget(settings.cache_dir, 5),
        session=FakeSession(FakeResponse(200, {})),
    )
    with pytest.raises(SamAuthError):
        client.get_json("/opportunities/v2/search", {"q": "x"})


def test_opportunities_client_parses(tmp_path, sample_opportunity_payload):
    from dragonpulse.models.filters import OpportunityFilters

    sam = _client(tmp_path, FakeResponse(200, sample_opportunity_payload))
    client = OpportunitiesClient(sam)
    result = client.search(OpportunityFilters(keyword="cyber"))
    assert result.count == 1
    assert result.opportunities[0].title == "Cybersecurity Support Services"


# --------------------------------------------------------------------------- #
# Opt-in live test (skipped by default)
# --------------------------------------------------------------------------- #
@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("DRAGONPULSE_RUN_LIVE") != "1",
    reason="Set DRAGONPULSE_RUN_LIVE=1 (and a real key in .env) to run live SAM.gov tests.",
)
def test_live_search_smoke():
    """Hits the real Opportunities v2 API. Consumes one request from your quota."""
    from datetime import date, timedelta

    from dragonpulse.models.filters import OpportunityFilters

    client = OpportunitiesClient()
    filters = OpportunityFilters(
        keyword="information technology",
        posted_from=date.today() - timedelta(days=14),
        posted_to=date.today(),
        limit=5,
    )
    result = client.search(filters)
    assert result.total_records >= 0
    assert result.count <= 5

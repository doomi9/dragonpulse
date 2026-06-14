"""Tests for the awards client and pricing analysis."""

from __future__ import annotations

from typing import Any

from dragonpulse.api.awards import AwardsClient
from dragonpulse.api.base import SamClient
from dragonpulse.api.opportunities import OpportunitiesClient
from dragonpulse.cache.disk_cache import DiskCache
from dragonpulse.cache.request_budget import RequestBudget
from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.models.award import Award, AwardSearchResult
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.processors.pricing import analyze_awards


class _Resp:
    def __init__(self, body: Any):
        self.status_code = 200
        self._body = body
        self.text = ""
        self.reason = "OK"
        self.headers = {}

    def json(self):
        return self._body


class _Session:
    def __init__(self, body: Any):
        self._body = body
        self.headers = {}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return _Resp(self._body)


def _awards_client(tmp_path, body) -> AwardsClient:
    settings = Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        daily_request_budget=10,
    )
    sam = SamClient(
        settings=settings,
        cache=DiskCache(settings.cache_dir),
        budget=RequestBudget(settings.cache_dir, 10),
        session=_Session(body),
    )
    return AwardsClient(opportunities_client=OpportunitiesClient(sam))


def test_awards_client_forces_award_notice_type(tmp_path, sample_award_payload):
    client = _awards_client(tmp_path, sample_award_payload)
    result = client.search_awards(OpportunityFilters(keyword="it"), max_records=10)
    assert result.total_records == 1
    award = result.awards[0]
    assert award.award_amount == 1_250_000.0
    assert award.awardee_name == "Acme Federal LLC"


def test_analyze_awards_statistics_and_table():
    awards = [
        Award(notice_id="1", awardee_name="A", award_amount=100.0, naics_code="541512"),
        Award(notice_id="2", awardee_name="B", award_amount=300.0, naics_code="541512"),
        Award(notice_id="3", awardee_name="C", award_amount=None),  # unpriced
    ]
    analysis = analyze_awards(AwardSearchResult(awards=awards, total_records=3))
    assert analysis.total_awards == 3
    assert analysis.priced_awards == 2
    assert analysis.stats["min"] == 100.0
    assert analysis.stats["max"] == 300.0
    assert analysis.has_pricing
    # Table sorts priced awards first (descending), unpriced last.
    assert analysis.awards_table.iloc[0]["Awardee"] == "B"
    assert analysis.awards_table.iloc[-1]["Awardee"] == "C"


def test_analyze_awards_handles_no_pricing():
    awards = [Award(notice_id="1", awardee_name="A", award_amount=None)]
    analysis = analyze_awards(AwardSearchResult(awards=awards, total_records=1))
    assert analysis.total_awards == 1
    assert analysis.priced_awards == 0
    assert not analysis.has_pricing

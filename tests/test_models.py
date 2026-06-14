"""Tests for Pydantic model parsing and helpers."""

from __future__ import annotations

from datetime import date

import pytest

from dragonpulse.models.award import Award, AwardSearchResult
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.models.opportunity import OpportunitySearchResult


def test_parse_search_result(sample_opportunity_payload):
    result = OpportunitySearchResult.model_validate(sample_opportunity_payload)
    assert result.total_records == 1
    assert result.count == 1
    opp = result.opportunities[0]
    assert opp.notice_id == "abc123def456"
    assert opp.title == "Cybersecurity Support Services"
    assert opp.naics_code == "541512"


def test_opportunity_derived_fields(sample_opportunity_payload):
    opp = OpportunitySearchResult.model_validate(sample_opportunity_payload).opportunities[0]
    assert opp.agency == "DEPT OF DEFENSE"
    assert opp.office == "ACC"
    assert opp.sam_url == "https://sam.gov/opp/abc123def456/view"
    assert opp.response_deadline is not None
    assert opp.response_deadline.year == 2026


def test_resource_links_normalized_from_strings(sample_opportunity_payload):
    opp = OpportunitySearchResult.model_validate(sample_opportunity_payload).opportunities[0]
    assert len(opp.resource_links) == 2
    assert all(rl.url.startswith("https://") for rl in opp.resource_links)


def test_points_of_contact_primary(sample_opportunity_payload):
    opp = OpportunitySearchResult.model_validate(sample_opportunity_payload).opportunities[0]
    assert len(opp.points_of_contact) == 2
    primary = opp.primary_contact
    assert primary is not None
    assert primary.full_name == "Jane Contracting Officer"
    assert primary.email == "jane.co@army.mil"


def test_table_row_contains_key_fields(sample_opportunity_payload):
    opp = OpportunitySearchResult.model_validate(sample_opportunity_payload).opportunities[0]
    row = opp.to_table_row()
    for key in ("Title", "Agency", "Response Deadline", "Notice ID", "Link"):
        assert key in row


def test_award_from_payload(sample_award_payload):
    result = OpportunitySearchResult.model_validate(sample_award_payload)
    award = Award.from_opportunity(result.opportunities[0])
    assert award.award_amount == 1_250_000.0
    assert award.awardee_name == "Acme Federal LLC"
    assert award.naics_code == "541512"


def test_award_summary_statistics():
    awards = [
        Award(notice_id="1", award_amount=100.0),
        Award(notice_id="2", award_amount=200.0),
        Award(notice_id="3", award_amount=300.0),
    ]
    summary = AwardSearchResult(awards=awards).summary()
    assert summary["count"] == 3
    assert summary["min"] == 100.0
    assert summary["max"] == 300.0
    assert summary["median"] == 200.0
    assert summary["mean"] == 200.0


def test_filters_to_query_params():
    filters = OpportunityFilters(
        keyword="cyber",
        naics_codes=["541512", "541330"],
        set_aside_codes=["SDVOSBC"],
        notice_type_codes=["k", "o"],
        posted_from=date(2026, 1, 1),
        posted_to=date(2026, 1, 31),
        limit=10,
    )
    params = filters.to_query_params()
    assert params["postedFrom"] == "01/01/2026"
    assert params["postedTo"] == "01/31/2026"
    assert params["title"] == "cyber"
    assert params["ncode"] == "541512,541330"
    assert params["typeOfSetAside"] == "SDVOSBC"
    assert params["ptype"] == "k,o"
    assert params["limit"] == 10


def test_filters_reject_backwards_range():
    with pytest.raises(ValueError):
        OpportunityFilters(posted_from=date(2026, 2, 1), posted_to=date(2026, 1, 1))


def test_filters_reject_over_one_year():
    with pytest.raises(ValueError):
        OpportunityFilters(posted_from=date(2024, 1, 1), posted_to=date(2026, 1, 1))

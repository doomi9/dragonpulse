"""Tests for loading an opportunity from a SAM.gov link (public page, no keyed API)."""

from __future__ import annotations

import pytest

from dragonpulse.processors.sam_scrape import (
    SamScrapeError,
    _html_to_text,
    fetch_opportunity_from_link,
    parse_sam_link,
)

OPP_ID = "aff7868ee48c4dd2a925b3b554bc15c4"

_OPP_PAYLOAD = {
    "data2": {
        "type": "r",
        "title": "Construction of redundant 69-kV electrical power system",
        "naics": [{"code": ["237130"], "type": "primary"}],
        "archive": {"date": "2026-07-30", "type": "autocustom"},
        "solicitation": {
            "deadlines": {"response": "2026-06-08T23:59:00-05:00"},
            "setAside": "SBA",
        },
        "organizationId": "500181913",
        "solicitationNumber": "W519TC-26-R-69KV",
        "pointOfContact": [
            {"type": "primary", "email": "a@army.mil", "fullName": "Anna Thissen"},
        ],
        "placeOfPerformance": {
            "zip": "24141",
            "city": {"code": "65392", "name": "Radford"},
            "state": {"code": "VA", "name": "Virginia"},
            "country": {"code": "USA", "name": "UNITED STATES"},
        },
    },
    "postedDate": "2026-06-01T18:31:11.125+00:00",
    "description": [
        {"body": "<p><strong>Description:</strong>&nbsp;Design and build a 69-kV system.</p>"}
    ],
}

_ORG_PAYLOAD = {
    "_embedded": [
        {"org": {"fullParentPathName": "DEPT OF DEFENSE.DEPT OF THE ARMY.ACC.W6QK ACC-RI"}}
    ]
}

_RES_PAYLOAD = {
    "_embedded": {
        "opportunityAttachmentList": [
            {
                "attachments": [
                    {"resourceId": "res-1", "name": "SOW.pdf", "type": "file"},
                ],
                "resourceLinks": ["https://example.com/extra"],
            }
        ]
    }
}


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Routes requests to canned payloads by URL substring; records call count."""

    def __init__(self, *, opp=_OPP_PAYLOAD, org=_ORG_PAYLOAD, res=_RES_PAYLOAD,
                 opp_status=200):
        self.opp = opp
        self.org = org
        self.res = res
        self.opp_status = opp_status
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        if "/federalorganizations/" in url:
            return _FakeResponse(self.org)
        if "/resources" in url:
            return _FakeResponse(self.res)
        if "/opportunities/" in url:
            return _FakeResponse(self.opp, status_code=self.opp_status)
        return _FakeResponse(None, status_code=404)


# --------------------------------------------------------------------------- #
# parse_sam_link
# --------------------------------------------------------------------------- #
def test_parse_workspace_url():
    assert parse_sam_link(
        f"https://sam.gov/workspace/contract/opp/{OPP_ID}/view"
    ) == OPP_ID


def test_parse_classic_url():
    assert parse_sam_link(f"https://sam.gov/opp/{OPP_ID}/view") == OPP_ID


def test_parse_url_with_query():
    assert parse_sam_link(f"https://sam.gov/opp/{OPP_ID}/view?foo=bar") == OPP_ID


def test_parse_bare_id():
    assert parse_sam_link(f"  {OPP_ID}  ") == OPP_ID


def test_parse_empty_raises():
    with pytest.raises(SamScrapeError, match="Paste a SAM.gov"):
        parse_sam_link("   ")


def test_parse_garbage_raises():
    with pytest.raises(SamScrapeError, match="doesn't look like"):
        parse_sam_link("https://sam.gov/search?foo=bar")


# --------------------------------------------------------------------------- #
# fetch_opportunity_from_link
# --------------------------------------------------------------------------- #
def test_fetch_parses_all_fields():
    session = _FakeSession()
    scraped = fetch_opportunity_from_link(
        f"https://sam.gov/workspace/contract/opp/{OPP_ID}/view", session=session
    )
    o = scraped.opportunity
    assert o.notice_id == OPP_ID
    assert o.title == "Construction of redundant 69-kV electrical power system"
    assert o.solicitation_number == "W519TC-26-R-69KV"
    assert o.naics_code == "237130"
    assert o.notice_type == "Sources Sought"
    assert o.set_aside_description == "Total Small Business Set-Aside"
    assert o.agency == "DEPT OF DEFENSE"
    assert o.office == "W6QK ACC-RI"
    assert o.response_deadline is not None
    assert o.place_of_performance.city == "Radford"
    assert o.place_of_performance.state == "Virginia"
    assert [p.display_name for p in o.points_of_contact] == ["Anna Thissen"]
    # Scraped records are flagged as no-keyed-API and link-sourced.
    assert o.manual_entry is True
    assert o.loaded_via == "sam_link"
    assert o.sam_url.endswith(f"{OPP_ID}/view")
    # Description is cleaned to plain text and surfaced for indexing.
    assert "Design and build a 69-kV system." in scraped.description
    assert "<p>" not in scraped.description


def test_fetch_parses_attachments():
    scraped = fetch_opportunity_from_link(
        f"https://sam.gov/opp/{OPP_ID}/view", session=_FakeSession()
    )
    names = scraped.attachments
    assert any("SOW.pdf" in n for n in names)
    # The external resource link is also captured.
    assert any("example.com/extra" in (r.url) for r in scraped.opportunity.resource_links)


def test_fetch_uses_only_public_frontend_endpoints():
    session = _FakeSession()
    fetch_opportunity_from_link(f"https://sam.gov/opp/{OPP_ID}/view", session=session)
    # Every call must hit the public sam.gov frontend, never api.sam.gov (the key).
    assert session.calls, "expected at least one frontend call"
    assert all(u.startswith("https://sam.gov/api/prod/") for u in session.calls)
    assert not any("api.sam.gov" in u for u in session.calls)


def test_fetch_404_raises_friendly_error():
    session = _FakeSession(opp_status=404)
    with pytest.raises(SamScrapeError, match="Couldn't find that opportunity"):
        fetch_opportunity_from_link(f"https://sam.gov/opp/{OPP_ID}/view", session=session)


def test_fetch_empty_data_raises():
    session = _FakeSession(opp={"data2": {}})
    with pytest.raises(SamScrapeError, match="didn't contain opportunity data"):
        fetch_opportunity_from_link(f"https://sam.gov/opp/{OPP_ID}/view", session=session)


def test_fetch_survives_org_and_attachment_failures():
    # Org/resources return unusable payloads -> still parses the core opportunity.
    session = _FakeSession(org={}, res={})
    scraped = fetch_opportunity_from_link(
        f"https://sam.gov/opp/{OPP_ID}/view", session=session
    )
    assert scraped.opportunity.naics_code == "237130"
    assert scraped.opportunity.agency is None  # org resolution degraded gracefully
    assert scraped.attachments == []


# --------------------------------------------------------------------------- #
# HTML cleaning
# --------------------------------------------------------------------------- #
def test_html_to_text_preserves_structure():
    out = _html_to_text(
        "<p>First para.</p><ul><li>Item A</li><li>Item B</li></ul>"
        "<p>Line&nbsp;break<br>here.</p>"
    )
    assert "First para." in out
    assert "• Item A" in out
    assert "• Item B" in out
    assert "Line break" in out
    assert "<" not in out

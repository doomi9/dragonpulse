"""Tests for the rules-based checklist and outreach templates."""

from __future__ import annotations

from dragonpulse.models.opportunity import OpportunitySearchResult
from dragonpulse.processors.checklist import build_checklist
from dragonpulse.processors.outreach import CompanyProfile, build_template_draft


def _opp(payload):
    return OpportunitySearchResult.model_validate(payload).opportunities[0]


def test_checklist_is_grounded_and_nonempty(sample_opportunity_payload):
    opp = _opp(sample_opportunity_payload)
    items = build_checklist(opp, use_llm=False)
    assert items, "checklist should not be empty"
    # Every item must cite a source (grounding requirement).
    assert all(item.source for item in items)


def test_checklist_includes_deadline_item(sample_opportunity_payload):
    opp = _opp(sample_opportunity_payload)
    items = build_checklist(opp, use_llm=False)
    assert any("deadline" in i.action.lower() or "deadline" in i.detail.lower() for i in items)


def test_checklist_includes_set_aside_check(sample_opportunity_payload):
    opp = _opp(sample_opportunity_payload)
    items = build_checklist(opp, use_llm=False)
    assert any("set-aside" in i.action.lower() for i in items)


def test_outreach_template_is_grounded(sample_opportunity_payload):
    opp = _opp(sample_opportunity_payload)
    poc = opp.primary_contact
    draft = build_template_draft(opp, poc, CompanyProfile(company_name="DragonPulse Inc"))
    assert not draft.used_llm
    assert "DragonPulse Inc" in draft.body
    assert opp.notice_id in draft.subject or (opp.solicitation_number or "") in draft.subject
    assert draft.sources, "draft must record its grounding sources"
    assert draft.to_email == poc.email

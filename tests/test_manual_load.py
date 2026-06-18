"""Tests for manually loading opportunities (no API call)."""

from __future__ import annotations

import pytest

from dragonpulse.models.opportunity import (
    Opportunity,
    parse_opportunity_reference,
)


# --------------------------------------------------------------------------- #
# parse_opportunity_reference
# --------------------------------------------------------------------------- #
def test_parse_full_sam_url():
    nid, link = parse_opportunity_reference(
        "https://sam.gov/opp/abc123def456/view"
    )
    assert nid == "abc123def456"
    assert link == "https://sam.gov/opp/abc123def456/view"


def test_parse_url_without_view_suffix():
    nid, link = parse_opportunity_reference("https://sam.gov/opp/XYZ789")
    assert nid == "XYZ789"
    assert link == "https://sam.gov/opp/XYZ789"


def test_parse_url_with_query_string():
    nid, link = parse_opportunity_reference(
        "https://sam.gov/opp/deadbeef0001/view?keywords=foo"
    )
    assert nid == "deadbeef0001"


def test_parse_bare_notice_id():
    nid, link = parse_opportunity_reference("  abc123DEF456  ")
    assert nid == "abc123DEF456"
    assert link is None


def test_parse_empty_raises():
    with pytest.raises(ValueError, match="Enter a SAM.gov link"):
        parse_opportunity_reference("   ")


def test_parse_url_without_opp_segment_raises():
    with pytest.raises(ValueError, match="Couldn't find a Notice ID"):
        parse_opportunity_reference("https://sam.gov/search?index=opp")


def test_parse_garbage_with_space_raises():
    with pytest.raises(ValueError, match="doesn't look like"):
        parse_opportunity_reference("not a notice id")


# --------------------------------------------------------------------------- #
# Opportunity.manual
# --------------------------------------------------------------------------- #
def test_manual_opportunity_minimal():
    opp = Opportunity.manual("abc123")
    assert opp.notice_id == "abc123"
    assert opp.manual_entry is True
    assert opp.resource_links == []
    assert "abc123" in opp.title
    # A usable SAM.gov link is always derivable.
    assert opp.sam_url.endswith("abc123/view")


def test_manual_opportunity_with_metadata():
    opp = Opportunity.manual(
        "n1",
        title="Dam Exciter Replacement",
        ui_link="https://sam.gov/opp/n1/view",
        solicitation_number="W912-26-R-0001",
        naics_code="221111",
        agency="ARMY CORPS OF ENGINEERS",
    )
    assert opp.title == "Dam Exciter Replacement"
    assert opp.solicitation_number == "W912-26-R-0001"
    assert opp.naics_code == "221111"
    assert opp.agency == "ARMY CORPS OF ENGINEERS"
    assert opp.sam_url == "https://sam.gov/opp/n1/view"


def test_manual_opportunity_days_until_deadline_is_none():
    # No deadline provided -> no crash, returns None.
    assert Opportunity.manual("n2").days_until_deadline() is None


def test_api_parsed_opportunity_is_not_manual(sample_opportunity_payload):
    from dragonpulse.models.opportunity import OpportunitySearchResult

    opp = OpportunitySearchResult.model_validate(
        sample_opportunity_payload
    ).opportunities[0]
    assert opp.manual_entry is False


# --------------------------------------------------------------------------- #
# End-to-end: manual opportunity -> draft + compliance, with ZERO API calls
# --------------------------------------------------------------------------- #
def test_manual_opportunity_drafts_without_api_calls(tmp_path):
    from dragonpulse.cache.request_budget import RequestBudget
    from dragonpulse.config.settings import KeyTier, Settings
    from dragonpulse.processors.embeddings import HashingEmbedding
    from dragonpulse.processors.knowledge_base import KnowledgeBase
    from dragonpulse.processors.proposal import ProposalGenerator

    settings = Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        rag_embedding_backend="hashing",
        rag_chunk_chars=300,
        rag_chunk_overlap=50,
        llm_enabled=False,
    )
    budget = RequestBudget(state_dir=settings.cache_dir, daily_budget=9)
    before = budget.used_today()

    kb = KnowledgeBase(settings=settings, backend=HashingEmbedding(256))
    kb.add_document(
        "Capabilities.txt",
        "Dragon Infrastructure performs exciter replacement and switchgear upgrades "
        "for hydroelectric dams operated by federal agencies.",
        category="Capabilities",
    )

    # Manually loaded opportunity (no API record).
    opp = Opportunity.manual(
        "manual123",
        title="Hydroelectric Dam Exciter Replacement",
        ui_link="https://sam.gov/opp/manual123/view",
    )

    gen = ProposalGenerator(opp, settings=settings, knowledge_base=kb)
    # Solicitation supplied manually (pasted text) — no download.
    n = gen.load_solicitation(
        [(
            "Pasted SOW",
            "Section L. The offeror shall provide exciter replacement services. "
            "Section M. Proposals evaluated on technical approach and past performance. "
            "The contractor shall perform switchgear upgrades at the dam.",
        )]
    )
    assert n > 0

    draft = gen.generate_draft(include_optional=False)
    assert draft.sections
    matrix = gen.extract_compliance_matrix(draft)
    assert matrix.items  # compliance matrix works for a manual opportunity

    # The crucial assertion: not a single SAM.gov live request was spent.
    assert RequestBudget(state_dir=settings.cache_dir, daily_budget=9).used_today() == before


def _docx_bytes(text: str) -> bytes:
    """Build a minimal .docx (simulating an uploaded solicitation file)."""
    import io

    import docx

    document = docx.Document()
    for para in text.split("\n"):
        document.add_paragraph(para)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_manual_upload_solicitation_drafts_without_api_calls(tmp_path):
    """Upload-first flow: extract an uploaded solicitation file, index, draft."""
    from dragonpulse.cache.request_budget import RequestBudget
    from dragonpulse.config.settings import KeyTier, Settings
    from dragonpulse.processors.embeddings import HashingEmbedding
    from dragonpulse.processors.knowledge_base import KnowledgeBase
    from dragonpulse.processors.proposal import ProposalGenerator
    from dragonpulse.processors.text_extract import extract_text_with_ocr

    settings = Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        rag_embedding_backend="hashing",
        rag_chunk_chars=300,
        rag_chunk_overlap=50,
        llm_enabled=False,
    )
    budget = RequestBudget(state_dir=settings.cache_dir, daily_budget=9)
    before = budget.used_today()

    kb = KnowledgeBase(settings=settings, backend=HashingEmbedding(256))
    kb.add_document(
        "Capabilities.txt",
        "Dragon Infrastructure performs exciter replacement and switchgear upgrades.",
        category="Capabilities",
    )

    # No reference pasted: the loader mints a local Notice ID for the upload.
    opp = Opportunity.manual("MANUAL-20260615-1200-abcd", title="SOW")
    assert opp.manual_entry is True

    # Simulate the uploaded solicitation file being extracted locally (no API).
    sow = (
        "Section L. The offeror shall submit a technical proposal. "
        "Section M. Proposals will be evaluated on technical approach. "
        "The contractor shall perform exciter replacement at the dam."
    )
    text = extract_text_with_ocr(_docx_bytes(sow), "SOW.docx")
    assert "exciter" in text.lower()

    gen = ProposalGenerator(opp, settings=settings, knowledge_base=kb)
    n = gen.load_solicitation([("SOW.docx", text)])
    assert n > 0

    draft = gen.generate_draft(include_optional=False)
    assert draft.sections
    matrix = gen.extract_compliance_matrix(draft)
    assert matrix.items

    # Zero SAM.gov live requests for the whole upload -> draft flow.
    assert RequestBudget(state_dir=settings.cache_dir, daily_budget=9).used_today() == before

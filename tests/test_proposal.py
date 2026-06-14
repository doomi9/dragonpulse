"""Tests for the grounded proposal generator (no-LLM scaffold path)."""

from __future__ import annotations

from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.models.opportunity import OpportunitySearchResult
from dragonpulse.models.proposal import (
    CitationEvidence,
    ProposalDraft,
    ProposalSection,
)
from dragonpulse.processors.embeddings import HashingEmbedding
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.proposal import (
    SECTION_SPECS,
    ProposalGenerator,
    SolicitationIndex,
    draft_to_docx_bytes,
)


def _settings(tmp_path) -> Settings:
    # LLM disabled -> deterministic scaffold path (no network needed).
    return Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        rag_embedding_backend="hashing",
        rag_chunk_chars=300,
        rag_chunk_overlap=50,
        llm_enabled=False,
    )


def _kb(settings) -> KnowledgeBase:
    kb = KnowledgeBase(settings=settings, backend=HashingEmbedding(256))
    kb.add_document(
        "Capabilities.txt",
        "Dragon Infrastructure provides high-voltage power line construction, "
        "substation upgrades, and electrical grid modernization for federal clients. "
        "Our crews hold relevant safety certifications and deliver on schedule.",
    )
    kb.add_document(
        "Past Performance.txt",
        "We completed a power transmission line project for the Army Corps of Engineers, "
        "including substation work, on time and on budget.",
    )
    return kb


def _opp(payload):
    return OpportunitySearchResult.model_validate(payload).opportunities[0]


def test_solicitation_index_build_and_search(tmp_path):
    backend = HashingEmbedding(256)
    idx = SolicitationIndex(backend)
    n = idx.add(
        [("SOW.txt", "The contractor shall construct power transmission lines and "
                     "perform substation maintenance for the installation. " * 10)],
        chunk_chars=300,
        overlap=50,
    )
    assert n > 0
    hits = idx.search("power line construction requirements", k=2)
    assert hits
    assert hits[0].origin == "solicitation"
    assert hits[0].label.startswith("Solicitation: SOW.txt")


def test_generate_draft_scaffold_is_grounded(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    kb = _kb(settings)
    gen = ProposalGenerator(_opp(sample_opportunity_payload), settings=settings, knowledge_base=kb)
    gen.load_solicitation(
        [("SOW.txt", "The contractor shall provide cybersecurity support services "
                     "including monitoring and incident response. " * 10)]
    )
    draft = gen.generate_draft(include_optional=False)

    # Core (non-optional) sections present.
    ids = {s.section_id for s in draft.sections}
    assert "executive_summary" in ids
    assert "technical_approach" in ids
    assert "past_performance" in ids
    assert "pricing_strategy" not in ids  # optional excluded

    # No LLM -> scaffold, but every section must carry grounding sources.
    for section in draft.sections:
        assert section.used_llm is False
        assert section.sources, f"{section.section_id} has no grounding sources"
        # Sources come from solicitation and/or the KB.
        origins = {s.origin for s in section.sources}
        assert origins <= {"solicitation", "knowledge_base"}


def test_generate_draft_includes_optional_pricing(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    draft = gen.generate_draft(include_optional=True)
    assert any(s.section_id == "pricing_strategy" for s in draft.sections)


def test_regeneration_records_feedback(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    spec = next(s for s in SECTION_SPECS if s.section_id == "technical_approach")
    first = gen.generate_section(spec)
    second = gen.generate_section(spec, feedback="focus on liquid cooling", prior=first)
    assert second.feedback_history == ["focus on liquid cooling"]


def test_draft_markdown_and_docx_export(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    draft = gen.generate_draft()
    md = draft.to_markdown()
    assert "# Proposal Draft" in md
    assert "## Executive Summary" in md

    docx_bytes = draft_to_docx_bytes(draft)
    assert docx_bytes[:2] == b"PK"  # .docx is a zip container
    assert len(docx_bytes) > 0


def test_proposal_section_source_split():
    section = ProposalSection(
        section_id="x",
        title="X",
        content="body",
        sources=[
            CitationEvidence(label="Solicitation: SOW #1", origin="solicitation", snippet="a"),
            CitationEvidence(label="KB: Cap #1", origin="knowledge_base", snippet="b"),
        ],
    )
    assert len(section.solicitation_sources) == 1
    assert len(section.kb_sources) == 1


def test_draft_replace_section():
    draft = ProposalDraft(notice_id="n1", opportunity_title="T")
    draft.sections.append(ProposalSection(section_id="a", title="A", content="old"))
    draft.replace_section(ProposalSection(section_id="a", title="A", content="new"))
    assert len(draft.sections) == 1
    assert draft.get_section("a").content == "new"

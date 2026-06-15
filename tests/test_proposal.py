"""Tests for the grounded proposal generator (no-LLM scaffold path)."""

from __future__ import annotations

from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.models.opportunity import OpportunitySearchResult
from dragonpulse.models.proposal import (
    COMPLIANCE_STATUSES,
    CitationEvidence,
    ComplianceItem,
    ComplianceMatrix,
    ProposalDraft,
    ProposalSection,
)
from dragonpulse.processors.embeddings import HashingEmbedding
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.proposal import (
    SECTION_SPECS,
    ProposalGenerator,
    SolicitationIndex,
    compliance_matrix_to_xlsx_bytes,
    draft_to_docx_bytes,
    kb_evidence_is_weak,
)

_SOW_WITH_REQUIREMENTS = (
    "Section L - Instructions to Offerors. The offeror shall submit a technical "
    "proposal not to exceed 20 pages. The contractor shall provide cybersecurity "
    "monitoring and incident response services. The offeror must be registered in "
    "SAM.gov prior to award. Section M - Evaluation Factors. Proposals will be "
    "evaluated on technical approach and past performance. The Government will "
    "evaluate the management plan for adequacy of staffing. The contractor must "
    "maintain a quality control plan throughout the period of performance. "
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


# --------------------------------------------------------------------------- #
# Compliance matrix
# --------------------------------------------------------------------------- #
def _gen_with_sow(tmp_path, payload, sow: str = _SOW_WITH_REQUIREMENTS):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(_opp(payload), settings=settings, knowledge_base=_kb(settings))
    gen.load_solicitation([("SOW.txt", sow * 4)])
    return gen


def test_compliance_matrix_rules_extraction(tmp_path, sample_opportunity_payload):
    gen = _gen_with_sow(tmp_path, sample_opportunity_payload)
    draft = gen.generate_draft()
    matrix = gen.extract_compliance_matrix(draft, max_items=10)

    assert isinstance(matrix, ComplianceMatrix)
    assert matrix.used_llm is False  # LLM disabled -> rules fallback
    assert matrix.items, "expected at least one extracted requirement"
    # Every item is grounded in a solicitation source and mapped to a section.
    for item in matrix.items:
        assert item.source_label.startswith("Solicitation:")
        assert item.section_id is not None
        assert item.section_title
        assert item.status in COMPLIANCE_STATUSES
        assert item.category in ("Section L", "Section M", "Evaluation", "SOW (shall)", "Other")
    # The "shall"/"must" sentences should have been captured.
    joined = " ".join(it.requirement.lower() for it in matrix.items)
    assert "shall" in joined or "must" in joined


def test_compliance_matrix_caps_items(tmp_path, sample_opportunity_payload):
    gen = _gen_with_sow(tmp_path, sample_opportunity_payload)
    draft = gen.generate_draft()
    matrix = gen.extract_compliance_matrix(draft, max_items=3)
    assert len(matrix.items) <= 3


def test_compliance_matrix_empty_without_solicitation(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    draft = gen.generate_draft()  # no solicitation loaded
    matrix = gen.extract_compliance_matrix(draft)
    assert matrix.items == []


def test_compliance_matrix_markdown_and_status_counts():
    matrix = ComplianceMatrix(
        notice_id="n1",
        items=[
            ComplianceItem(
                requirement="The offeror shall submit a technical proposal.",
                category="Section L",
                source_label="Solicitation: SOW.txt #1",
                section_title="Technical Approach",
                status="Addressed",
            ),
            ComplianceItem(
                requirement="Proposals will be evaluated on past performance.",
                category="Section M",
                source_label="Solicitation: SOW.txt #2",
                section_title="Relevant Past Performance",
                status="Partial",
            ),
        ],
    )
    counts = matrix.status_counts()
    assert counts["Addressed"] == 1
    assert counts["Partial"] == 1
    assert counts["Not Addressed"] == 0

    md = matrix.to_markdown()
    assert "## Compliance Matrix" in md
    assert "Technical Approach" in md
    assert "| Status |" in md


def test_compliance_matrix_xlsx_export():
    matrix = ComplianceMatrix(
        notice_id="n1",
        items=[
            ComplianceItem(
                requirement="The contractor shall maintain a quality control plan.",
                category="SOW (shall)",
                source_label="Solicitation: SOW.txt #3",
                section_title="Management & Staffing Plan",
                status="Addressed",
                notes="Covered in QC subsection.",
            )
        ],
    )
    xlsx = compliance_matrix_to_xlsx_bytes(matrix)
    assert xlsx[:2] == b"PK"  # .xlsx is a zip container
    assert len(xlsx) > 0


def test_draft_markdown_includes_compliance(tmp_path, sample_opportunity_payload):
    gen = _gen_with_sow(tmp_path, sample_opportunity_payload)
    draft = gen.generate_draft()
    draft.compliance = gen.extract_compliance_matrix(draft)
    md = draft.to_markdown()
    assert "## Compliance Matrix" in md
    # DOCX with an embedded matrix still renders.
    docx_bytes = draft_to_docx_bytes(draft)
    assert docx_bytes[:2] == b"PK"


# --------------------------------------------------------------------------- #
# Strengthen a section + past-performance honesty
# --------------------------------------------------------------------------- #
def test_strengthen_section_records_requirement_feedback(tmp_path, sample_opportunity_payload):
    gen = _gen_with_sow(tmp_path, sample_opportunity_payload)
    draft = gen.generate_draft()
    section = draft.get_section("technical_approach")
    strengthened = gen.strengthen_section_for_requirement(
        section,
        "The contractor shall provide 24/7 incident response.",
        source_label="Solicitation: SOW.txt #1",
    )
    assert strengthened.section_id == "technical_approach"
    assert strengthened.feedback_history  # the strengthen feedback was recorded
    assert "incident response" in strengthened.feedback_history[-1]


def test_remap_compliance_updates_sections(tmp_path, sample_opportunity_payload):
    gen = _gen_with_sow(tmp_path, sample_opportunity_payload)
    draft = gen.generate_draft()
    matrix = gen.extract_compliance_matrix(draft)
    assert matrix.items
    # Re-mapping should keep every item mapped to a real section.
    gen.remap_compliance(matrix, draft)
    section_ids = {s.section_id for s in draft.sections}
    for item in matrix.items:
        assert item.section_id in section_ids


def test_kb_evidence_is_weak_logic():
    strong = [
        CitationEvidence(label="KB: A #1", origin="knowledge_base", snippet="x", score=0.7),
        CitationEvidence(label="Sol: S #1", origin="solicitation", snippet="y", score=0.1),
    ]
    weak = [
        CitationEvidence(label="KB: A #1", origin="knowledge_base", snippet="x", score=0.2),
    ]
    none_kb = [
        CitationEvidence(label="Sol: S #1", origin="solicitation", snippet="y", score=0.9),
    ]
    assert kb_evidence_is_weak(strong) is False
    assert kb_evidence_is_weak(weak) is True
    assert kb_evidence_is_weak(none_kb) is True  # no KB evidence at all


def test_past_performance_spec_tuned():
    spec = next(s for s in SECTION_SPECS if s.section_id == "past_performance")
    assert spec.honest_transferable is True
    assert spec.kb_k >= 6

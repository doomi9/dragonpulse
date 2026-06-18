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
    _primary_proposal_document,
    _section_match_bonus,
    clean_proposal_content,
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

    # Core Tab structure present (traditional government proposal format).
    ids = {s.section_id for s in draft.sections}
    assert "title_page" in ids
    assert "table_of_contents" in ids
    assert "tab_1_executive_summary" in ids
    assert "tab_2_equipment_descriptive" in ids
    assert "tab_5_past_performance" in ids
    assert "tab_6_appendices" in ids

    # No LLM -> scaffold for Tabs, but every Tab must carry grounding sources.
    for section in draft.sections:
        assert section.used_llm is False
        if section.section_id in ("title_page", "table_of_contents"):
            continue  # deterministic front matter — no solicitation/KB retrieval
        assert section.sources, f"{section.section_id} has no grounding sources"
        origins = {s.origin for s in section.sources}
        assert origins <= {"solicitation", "knowledge_base", "style_exemplar"}


def test_generate_draft_includes_all_tabs(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    draft = gen.generate_draft(include_optional=True)
    ids = {s.section_id for s in draft.sections}
    assert len(ids) == len(SECTION_SPECS)
    assert "tab_3_installation_plan" in ids
    assert "tab_4_technical_comments" in ids


def test_regeneration_records_feedback(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    spec = next(s for s in SECTION_SPECS if s.section_id == "tab_2_equipment_descriptive")
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
    assert "## Tab 1 — Executive Summary" in md
    assert "## Document Sources and References" in md
    # Sources are not inlined in section bodies.
    exec_section = draft.get_section("tab_1_executive_summary")
    assert "Sources:" not in exec_section.content

    docx_bytes = draft_to_docx_bytes(draft)
    assert docx_bytes[:2] == b"PK"  # .docx is a zip container
    assert len(docx_bytes) > 0

    import io

    import docx

    doc = docx.Document(io.BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    assert "**" not in full_text
    assert "Proposal Draft —" not in full_text  # no AI wrapper title
    assert "Title Page" in full_text or "Technical Proposal" in full_text
    assert "Document Sources and References" in full_text
    # Per-section Sources headings should not appear before the appendix.
    tab_indices = [i for i, t in enumerate(paragraphs) if t.startswith("Tab ")]
    sources_indices = [i for i, t in enumerate(paragraphs) if t == "Sources"]
    assert not sources_indices or (
        tab_indices and sources_indices[0] > max(tab_indices)
    )


def test_clean_proposal_content_strips_artifacts():
    dirty = (
        "**Tab 2 — Equipment**\n\n"
        "We propose the ## Basler ECS 2100 system.\n\n"
        "**Sources:**\n- [KB: Capabilities #1]\n- [Solicitation: SOW #2]"
    )
    cleaned = clean_proposal_content(dirty)
    assert "**" not in cleaned
    assert "##" not in cleaned
    assert "Sources:" not in cleaned

    exported = clean_proposal_content(dirty, for_export=True)
    assert "[KB:" not in exported
    assert "Basler ECS 2100" in exported


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
    section = draft.get_section("tab_2_equipment_descriptive")
    strengthened = gen.strengthen_section_for_requirement(
        section,
        "The contractor shall provide 24/7 incident response.",
        source_label="Solicitation: SOW.txt #1",
    )
    assert strengthened.section_id == "tab_2_equipment_descriptive"
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
    spec = next(s for s in SECTION_SPECS if s.section_id == "tab_5_past_performance")
    assert spec.honest_transferable is True
    # Retrieval is kept intentionally small now that a stronger model reasons from
    # a few relevant cues rather than a large pile of chunks.
    assert spec.kb_k >= 3


# --------------------------------------------------------------------------- #
# Style matching
# --------------------------------------------------------------------------- #
def _styled_kb(settings) -> KnowledgeBase:
    """A KB with category-tagged documents that have a distinctive house style."""
    kb = KnowledgeBase(settings=settings, backend=HashingEmbedding(256))
    kb.add_document(
        "Technical Volume.txt",
        "TECHNICAL APPROACH. Our crews execute exciter replacement and switchgear "
        "modernization for hydroelectric dams using a disciplined, phase-gated "
        "methodology. We sequence work as follows: assessment, design, "
        "installation, and commissioning, with hold points at each gate.",
        category="Technical",
    )
    kb.add_document(
        "Past Performance Vol.txt",
        "RELEVANT EXPERIENCE. For the Army Corps of Engineers we delivered a dam "
        "exciter replacement on schedule, restoring generator output and improving "
        "grid reliability for the installation.",
        category="Past Performance",
    )
    return kb


class _RecordingLLM:
    """Fake LLM that records the kwargs of the last complete() call."""

    def __init__(self):
        self.available = True
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)

        class _R:
            text = "We (the company) will perform the work. [KB style: Technical Volume.txt #1]"

        return _R()


def test_section_spec_has_style_categories():
    tech = next(s for s in SECTION_SPECS if s.section_id == "tab_2_equipment_descriptive")
    pp = next(s for s in SECTION_SPECS if s.section_id == "tab_5_past_performance")
    assert "Technical" in tech.style_categories
    assert "Past Performance" in pp.style_categories


def test_style_exemplars_drawn_from_prioritized_category(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    kb = _styled_kb(settings)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=kb
    )
    spec = next(s for s in SECTION_SPECS if s.section_id == "tab_2_equipment_descriptive")
    exemplars = gen._gather_style_exemplars(spec)
    assert exemplars
    assert all(e.origin == "style_exemplar" for e in exemplars)
    # The Technical-category document should be the source of the style exemplar.
    assert any("Technical Volume" in e.label for e in exemplars)


def test_deterministic_title_page_and_toc(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    title_spec = next(s for s in SECTION_SPECS if s.section_id == "title_page")
    toc_spec = next(s for s in SECTION_SPECS if s.section_id == "table_of_contents")
    title = gen.generate_section(title_spec)
    toc = gen.generate_section(toc_spec)
    assert title.used_llm is False
    assert toc.used_llm is False
    assert "Title Page" in title.content
    assert "Volume 1" in title.content
    assert "Technical Proposal" in title.content
    assert "Table of Contents" in toc.content
    assert "Tab 1 Executive Summary" in toc.content
    assert "Appendix 1" in toc.content


def test_generate_section_uses_proposal_style_prompt(tmp_path, sample_opportunity_payload):
    from dragonpulse.processors.proposal import _PROPOSAL_SYSTEM_PROMPT

    settings = _settings(tmp_path)
    kb = _styled_kb(settings)
    fake = _RecordingLLM()
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=kb, llm=fake
    )
    gen.load_solicitation([("SOW.txt", "The contractor shall replace the exciter. " * 8)])
    spec = next(s for s in SECTION_SPECS if s.section_id == "tab_2_equipment_descriptive")
    section = gen.generate_section(spec)

    assert section.used_llm is True
    assert fake.calls, "the LLM should have been called"
    call = fake.calls[-1]
    # The proposal-specific, style-aware system prompt is used.
    assert call["system_prompt"] == _PROPOSAL_SYSTEM_PROMPT
    assert call["max_tokens"] == spec.max_tokens
    # The context surfaces the style block (learning material, not facts to copy).
    assert "COMPANY WRITING STYLE" in call["context"]
    # The instruction reinforces writing in the company's voice for THIS opportunity.
    assert "VOLUME TAB STRUCTURE" in call["context"]
    assert "company's voice" in call["instruction"]
    # The section records style references separately from factual KB sources.
    assert section.style_sources


def test_low_relevance_kb_facts_are_dropped(tmp_path, sample_opportunity_payload):
    """Fact chunks below the relevance floor must not be injected as context."""
    from types import SimpleNamespace

    from dragonpulse.processors.proposal import _KB_RELEVANCE_FLOOR

    settings = _settings(tmp_path)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=_kb(settings)
    )
    spec = next(s for s in SECTION_SPECS if s.section_id == "tab_4_technical_comments")

    high = _KB_RELEVANCE_FLOOR + 0.3
    low = _KB_RELEVANCE_FLOOR - 0.1

    def fake_search(query, k=None, *, categories=None):
        if categories:  # style-exemplar lookups are not gated by the floor
            return [
                SimpleNamespace(
                    citation="Style.txt #1",
                    chunk=SimpleNamespace(text="company house style sample"),
                    score=0.5,
                )
            ]
        return [
            SimpleNamespace(
                citation="Relevant.txt #1",
                chunk=SimpleNamespace(text="clearly relevant company content"),
                score=high,
            ),
            SimpleNamespace(
                citation="Irrelevant.txt #1",
                chunk=SimpleNamespace(text="totally unrelated content"),
                score=low,
            ),
        ]

    gen.kb.search = fake_search
    evidence = gen._gather_evidence(spec)
    kb_labels = [e.label for e in evidence if e.origin == "knowledge_base"]
    assert any("Relevant.txt" in lbl for lbl in kb_labels)
    assert not any("Irrelevant.txt" in lbl for lbl in kb_labels)


def test_primary_proposal_document_prefers_full_proposal(tmp_path):
    """The largest technical/past-performance volume beats line cards."""
    settings = _settings(tmp_path)
    kb = KnowledgeBase(settings=settings, backend=HashingEmbedding(256))
    kb.add_document("Line Card.txt", "Short capability blurb.", category="Capabilities")
    kb.add_document(
        "Excitation Technical Proposal.pdf",
        "EXECUTIVE SUMMARY\nWe propose exciter replacement.\n\n"
        "TECHNICAL APPROACH\nDetailed methodology " * 20,
        category="Technical",
    )
    doc = _primary_proposal_document(kb)
    assert doc is not None
    assert "Excitation" in doc.name


def test_section_match_bonus_prefers_heading_keywords():
    assert _section_match_bonus("EXECUTIVE SUMMARY", ("executive summary",)) > 0
    assert _section_match_bonus("Pricing Table", ("executive summary",)) == 0


def test_style_exemplars_prefer_template_section(tmp_path, sample_opportunity_payload):
    settings = _settings(tmp_path)
    kb = _styled_kb(settings)
    gen = ProposalGenerator(
        _opp(sample_opportunity_payload), settings=settings, knowledge_base=kb
    )
    spec = next(s for s in SECTION_SPECS if s.section_id == "tab_5_past_performance")
    exemplars = gen._gather_style_exemplars(spec)
    assert exemplars
    assert all(e.origin == "style_exemplar" for e in exemplars)


def test_category_filtered_kb_search(tmp_path):
    settings = _settings(tmp_path)
    kb = _styled_kb(settings)
    hits = kb.search("exciter replacement methodology", k=5, categories=["Technical"])
    assert hits
    assert all("Technical Volume" in h.chunk.doc_name for h in hits)
    # A category with no documents yields nothing (caller decides the fallback).
    assert kb.search("anything", k=5, categories=["Pricing"]) == []

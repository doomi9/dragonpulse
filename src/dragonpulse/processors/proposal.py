"""Grounded proposal draft generation.

Pipeline (per section):

1. **Solicitation context** — the opportunity's attachment text is chunked and
   embedded into an *ephemeral* in-memory index (reusing the knowledge base's
   embedding backend). A section-specific query retrieves the most relevant
   solicitation passages.
2. **Company context** — the same query retrieves the most relevant chunks from
   the persistent RAG knowledge base (past proposals/performance).
3. **Generation** — both contexts (each labeled for citation) plus opportunity
   metadata are handed to the grounded LLM, which is instructed to use only the
   provided context and cite it. If no LLM is available, a deterministic
   *evidence scaffold* is produced instead, so the feature still works offline.

Every section records the exact evidence it used, so the whole draft is
auditable and never asserts a capability that is not in the cited context.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.models.proposal import CitationEvidence, ProposalDraft, ProposalSection
from dragonpulse.processors.embeddings import EmbeddingBackend, get_embedding_backend
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.llm import LLMClient, LLMUnavailable
from dragonpulse.processors.text_extract import chunk_text

logger = get_logger(__name__)

_SNIPPET_CHARS = 320
_MAX_SOLICITATION_CHARS = 60_000  # safety cap before chunking


@dataclass
class SectionSpec:
    """Definition of a proposal section to generate."""

    section_id: str
    title: str
    query: str          # retrieval query (semantic) for solicitation + KB
    instruction: str    # what to write
    optional: bool = False


# Order matters: this is the draft's section order.
SECTION_SPECS: List[SectionSpec] = [
    SectionSpec(
        section_id="executive_summary",
        title="Executive Summary",
        query="objective scope purpose summary requirement mission need",
        instruction=(
            "Write a concise Executive Summary (2-4 short paragraphs) that shows "
            "the company understands the requirement and is well suited to it. "
            "Reference the actual scope from the solicitation and the company's "
            "relevant strengths from the knowledge base."
        ),
    ),
    SectionSpec(
        section_id="technical_approach",
        title="Technical Approach",
        query="technical approach tasks methodology requirements performance work "
        "statement deliverables",
        instruction=(
            "Write a Technical Approach that maps the company's methods to the "
            "specific tasks/requirements in the solicitation excerpts. Use clear "
            "subsections or bullet points tied to stated requirements. Only claim "
            "methods/capabilities supported by the knowledge base context."
        ),
    ),
    SectionSpec(
        section_id="management_staffing",
        title="Management & Staffing Plan",
        query="management plan staffing key personnel project management quality control schedule",
        instruction=(
            "Write a Management & Staffing Plan covering project management "
            "approach, key roles, quality control, and communication. Ground "
            "staffing/role claims in the knowledge base; align oversight with any "
            "management or reporting requirements in the solicitation."
        ),
    ),
    SectionSpec(
        section_id="past_performance",
        title="Relevant Past Performance",
        query="past performance relevant experience similar projects prior contracts results",
        instruction=(
            "Write a Relevant Past Performance section citing specific prior work "
            "from the knowledge base that is similar in scope to this requirement. "
            "Do NOT invent projects, customers, dollar values, or outcomes — use "
            "only what the knowledge base context provides. If evidence is thin, "
            "say so plainly."
        ),
    ),
    SectionSpec(
        section_id="differentiators",
        title="Differentiators — Why Dragon Infrastructure",
        query="differentiators strengths unique capabilities competitive advantages certifications",
        instruction=(
            "Write a Differentiators section explaining why the company is the "
            "right choice, grounded strictly in capabilities/certifications present "
            "in the knowledge base and tied to what the solicitation values."
        ),
    ),
    SectionSpec(
        section_id="pricing_strategy",
        title="Pricing Strategy (High-Level Notes)",
        query="pricing cost price contract type labor rates budget estimate funding",
        instruction=(
            "Write high-level Pricing Strategy NOTES (not actual prices): the "
            "contract type if stated, cost drivers, and a sensible pricing posture. "
            "Do NOT fabricate dollar amounts or rates. These are internal notes."
        ),
        optional=True,
    ),
]


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text[:_SNIPPET_CHARS] + ("…" if len(text) > _SNIPPET_CHARS else "")


class SolicitationIndex:
    """Ephemeral, in-memory semantic index over the solicitation attachments."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.chunks: List[str] = []
        self.labels: List[str] = []
        self.vectors: np.ndarray = np.zeros((0, backend.dimension), dtype=np.float32)

    @property
    def is_empty(self) -> bool:
        return self.vectors.shape[0] == 0

    def add(self, attachments: List[Tuple[str, str]], chunk_chars: int, overlap: int) -> int:
        """Add ``(filename, text)`` pairs. Returns total chunks indexed."""
        all_chunks: List[str] = []
        all_labels: List[str] = []
        for name, text in attachments:
            if not text:
                continue
            text = text[:_MAX_SOLICITATION_CHARS]
            pieces = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
            for i, piece in enumerate(pieces):
                all_chunks.append(piece)
                all_labels.append(f"Solicitation: {name} #{i + 1}")
        if not all_chunks:
            return 0
        self.chunks = all_chunks
        self.labels = all_labels
        self.vectors = self.backend.embed(all_chunks)
        logger.info("Solicitation index built: %d chunks", len(all_chunks))
        return len(all_chunks)

    def search(self, query: str, k: int = 4) -> List[CitationEvidence]:
        if self.is_empty or not query.strip():
            return []
        q = self.backend.embed([query])[0]
        # Vectors are L2-normalized, so this dot product is cosine similarity.
        # errstate guards against spurious FP warnings from macOS Accelerate BLAS.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self.vectors @ q
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        top = np.argsort(scores)[::-1][:k]
        return [
            CitationEvidence(
                label=self.labels[i],
                origin="solicitation",
                snippet=_snippet(self.chunks[i]),
                score=float(scores[i]),
            )
            for i in top
        ]


class ProposalGenerator:
    """Generates grounded, cited proposal sections for an opportunity."""

    def __init__(
        self,
        opportunity: Opportunity,
        *,
        settings: Optional[Settings] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        llm: Optional[LLMClient] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.opp = opportunity
        self.settings = settings or get_settings()
        self.kb = knowledge_base or KnowledgeBase(self.settings)
        self.llm = llm or LLMClient(self.settings)
        self.backend = backend or self.kb.backend or get_embedding_backend(self.settings)
        self.solicitation = SolicitationIndex(self.backend)

    # ------------------------------------------------------------------ #
    # Context
    # ------------------------------------------------------------------ #
    def load_solicitation(self, attachments: List[Tuple[str, str]]) -> int:
        """Build the ephemeral solicitation index from extracted attachments."""
        return self.solicitation.add(
            attachments,
            chunk_chars=self.settings.rag_chunk_chars,
            overlap=self.settings.rag_chunk_overlap,
        )

    def _opportunity_block(self) -> str:
        opp = self.opp
        deadline = opp.response_deadline
        return "\n".join(
            [
                "== OPPORTUNITY METADATA ==",
                f"Title: {opp.title or 'N/A'}",
                f"Notice ID: {opp.notice_id}",
                f"Solicitation #: {opp.solicitation_number or 'N/A'}",
                f"Agency: {opp.agency or 'N/A'} / Office: {opp.office or 'N/A'}",
                f"NAICS: {opp.naics_code or 'N/A'}",
                f"Set-aside: {opp.set_aside_description or 'None'}",
                f"Response deadline: {deadline.strftime('%Y-%m-%d') if deadline else 'N/A'}",
            ]
        )

    def _gather_evidence(self, spec: SectionSpec, k: int = 4) -> List[CitationEvidence]:
        sol = self.solicitation.search(spec.query, k=k)
        kb_hits = self.kb.search(spec.query, k=k)
        kb_ev = [
            CitationEvidence(
                label=f"KB: {h.citation}",
                origin="knowledge_base",
                snippet=_snippet(h.chunk.text),
                score=h.score,
            )
            for h in kb_hits
        ]
        return sol + kb_ev

    @staticmethod
    def _context_block(evidence: List[CitationEvidence]) -> str:
        sol = [e for e in evidence if e.origin == "solicitation"]
        kb = [e for e in evidence if e.origin == "knowledge_base"]
        parts: List[str] = []
        if sol:
            parts.append("== SOLICITATION EXCERPTS ==")
            parts += [f"[{e.label}]\n{e.snippet}" for e in sol]
        if kb:
            parts.append("\n== COMPANY KNOWLEDGE BASE ==")
            parts += [f"[{e.label}]\n{e.snippet}" for e in kb]
        if not parts:
            parts.append("(No grounding context was retrieved for this section.)")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate_section(
        self,
        spec: SectionSpec,
        *,
        feedback: Optional[str] = None,
        prior: Optional[ProposalSection] = None,
    ) -> ProposalSection:
        """Generate (or regenerate) a single section, grounded and cited."""
        evidence = self._gather_evidence(spec)
        context = f"{self._opportunity_block()}\n\n{self._context_block(evidence)}"

        instruction = spec.instruction
        if feedback:
            instruction += (
                f"\n\nUSER FEEDBACK for this revision: {feedback}\n"
                "Apply the feedback while staying grounded in the CONTEXT."
            )
        if prior and prior.content:
            instruction += (
                "\n\nThe previous draft of this section is below; improve it per the "
                f"feedback rather than starting over:\n---\n{prior.content[:2000]}\n---"
            )

        used_llm = False
        content: str
        if self.llm.available:
            try:
                result = self.llm.complete(
                    instruction=instruction,
                    context=context,
                    sources=[e.label for e in evidence],
                    max_tokens=900,
                )
                content = result.text.strip() or self._scaffold(spec, evidence)
                used_llm = bool(result.text.strip())
            except LLMUnavailable as exc:
                logger.info("LLM unavailable for %s: %s", spec.section_id, exc)
                content = self._scaffold(spec, evidence)
        else:
            content = self._scaffold(spec, evidence)

        history = list(prior.feedback_history) if prior else []
        if feedback:
            history.append(feedback)

        return ProposalSection(
            section_id=spec.section_id,
            title=spec.title,
            content=content,
            sources=evidence,
            used_llm=used_llm,
            optional=spec.optional,
            feedback_history=history,
        )

    @staticmethod
    def _scaffold(spec: SectionSpec, evidence: List[CitationEvidence]) -> str:
        """Deterministic, grounded scaffold used when no LLM is available."""
        lines = [
            f"_Draft scaffold for **{spec.title}** — enable a local LLM "
            "(e.g. Ollama) for full prose. The grounded evidence below is what the "
            "section would be written from; nothing here is invented._",
            "",
            f"**What to address:** {spec.instruction}",
            "",
        ]
        sol = [e for e in evidence if e.origin == "solicitation"]
        kb = [e for e in evidence if e.origin == "knowledge_base"]
        if sol:
            lines.append("**Relevant solicitation requirements:**")
            lines += [f"- [{e.label}] {e.snippet}" for e in sol]
            lines.append("")
        if kb:
            lines.append("**Supporting company evidence (knowledge base):**")
            lines += [f"- [{e.label}] {e.snippet}" for e in kb]
            lines.append("")
        if not sol and not kb:
            lines.append(
                "_No grounding context retrieved. Load solicitation attachments "
                "and index relevant company documents, then regenerate._"
            )
        return "\n".join(lines)

    def generate_draft(self, *, include_optional: bool = False) -> ProposalDraft:
        """Generate all sections into a :class:`ProposalDraft`."""
        specs = [s for s in SECTION_SPECS if include_optional or not s.optional]
        sections = [self.generate_section(spec) for spec in specs]
        return ProposalDraft(
            notice_id=self.opp.notice_id,
            opportunity_title=self.opp.title or self.opp.notice_id,
            agency=self.opp.agency,
            solicitation_number=self.opp.solicitation_number,
            sections=sections,
            llm_model=self.settings.llm_model if self.llm.available else None,
        )


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def draft_to_docx_bytes(draft: ProposalDraft) -> bytes:
    """Render a :class:`ProposalDraft` to a .docx file (bytes)."""
    import docx

    document = docx.Document()
    document.add_heading(f"Proposal Draft — {draft.opportunity_title}", level=0)
    meta = document.add_paragraph()
    meta.add_run(f"Notice ID: {draft.notice_id}\n")
    if draft.solicitation_number:
        meta.add_run(f"Solicitation #: {draft.solicitation_number}\n")
    if draft.agency:
        meta.add_run(f"Agency: {draft.agency}\n")
    meta.add_run(f"Generated: {draft.generated_at}\n")
    meta.add_run(f"Drafting model: {draft.llm_model or 'grounded template (no LLM)'}")
    document.add_paragraph(draft.grounded_note, style="Intense Quote")

    for section in draft.sections:
        document.add_heading(section.title, level=1)
        for para in section.content.split("\n\n"):
            if para.strip():
                document.add_paragraph(para.strip())
        if section.sources:
            document.add_heading("Sources", level=2)
            for s in section.sources:
                document.add_paragraph(s.label, style="List Bullet")

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()

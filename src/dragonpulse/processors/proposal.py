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
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.models.proposal import (
    STATUS_ADDRESSED,
    STATUS_NOT_ADDRESSED,
    STATUS_PARTIAL,
    CitationEvidence,
    ComplianceItem,
    ComplianceMatrix,
    ProposalDraft,
    ProposalSection,
)
from dragonpulse.processors.embeddings import EmbeddingBackend, get_embedding_backend
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.llm import LLMClient, LLMUnavailable
from dragonpulse.processors.text_extract import chunk_text

logger = get_logger(__name__)

_SNIPPET_CHARS = 320
_MAX_SOLICITATION_CHARS = 60_000  # safety cap before chunking

# Retrieval queries that surface the compliance-relevant parts of a solicitation.
_COMPLIANCE_QUERIES = [
    "Section L instructions to offerors proposal submission requirements format page limit",
    "Section M evaluation criteria factors basis for award technical evaluation",
    "the contractor shall perform mandatory tasks requirements deliverables",
    "offeror must submit provide required documentation certifications registration",
]
_VALID_CATEGORIES = ("Section L", "Section M", "Evaluation", "SOW (shall)", "Other")
# Sentence-level detector for mandatory/evaluation language (rules fallback).
_REQUIREMENT_RE = re.compile(
    r"[^.\n]*\b(shall|must|will be evaluated|is required to|are required to|"
    r"offeror[s]? (?:shall|must|will)|the government will evaluate)\b[^.\n]*\.",
    re.IGNORECASE,
)
# Status thresholds on cosine similarity between a requirement and its best
# matching proposal section. Tuned for L2-normalized semantic embeddings; these
# are intentionally conservative so the auto-status under-claims rather than
# over-claims coverage. They are only a starting point the user edits in the UI.
_ADDRESSED_THRESHOLD = 0.58
_PARTIAL_THRESHOLD = 0.45
# Below this best-KB-cosine, treat direct past-performance matches as "weak" and
# steer the model toward an honest transferable-experience framing.
_KB_WEAK_THRESHOLD = 0.45


@dataclass
class SectionSpec:
    """Definition of a proposal section to generate."""

    section_id: str
    title: str
    query: str          # retrieval query (semantic) for solicitation + KB
    instruction: str    # what to write
    optional: bool = False
    sol_k: int = 4      # solicitation chunks to retrieve
    kb_k: int = 4       # knowledge-base chunks to retrieve
    honest_transferable: bool = False  # add transferable-experience honesty guidance


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
        query="past performance relevant experience similar projects prior contracts "
        "customers scope outcomes results capabilities delivered",
        instruction=(
            "Write a Relevant Past Performance section that connects the company's "
            "actual prior work (from the knowledge base) to THIS requirement. For "
            "each example, briefly state what was done and why it is relevant to the "
            "solicitation's scope. Do NOT invent projects, customers, dollar values, "
            "or outcomes — use only what the knowledge base context provides."
        ),
        kb_k=6,
        honest_transferable=True,
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


def _parse_json_object(text: str) -> Dict:
    """Best-effort parse of an LLM JSON reply (handles stray prose/fences)."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_category(value: Optional[str]) -> str:
    if not value:
        return "Other"
    v = str(value).strip()
    for cat in _VALID_CATEGORIES:
        if v.lower() == cat.lower():
            return cat
    return _categorize(v)


def _categorize(text: str) -> str:
    """Heuristic category from requirement text."""
    low = text.lower()
    if "section l" in low or "instructions to offeror" in low or "submission" in low:
        return "Section L"
    if "section m" in low or "basis for award" in low:
        return "Section M"
    if "evaluat" in low:
        return "Evaluation"
    if "shall" in low or "must" in low:
        return "SOW (shall)"
    return "Other"


def _prioritize(items: List["ComplianceItem"]) -> List["ComplianceItem"]:
    """Order rules-extracted items so evaluation/Section M surface first."""
    order = {"Section M": 0, "Evaluation": 1, "Section L": 2, "SOW (shall)": 3, "Other": 4}
    return sorted(items, key=lambda it: order.get(it.category, 9))


def kb_evidence_is_weak(
    evidence: List[CitationEvidence], threshold: float = _KB_WEAK_THRESHOLD
) -> bool:
    """True when the strongest knowledge-base match is below ``threshold``.

    Used to decide whether the Past Performance section should pivot to an honest
    "transferable experience" framing instead of claiming direct matches.
    """
    kb_scores = [
        e.score for e in evidence if e.origin == "knowledge_base" and e.score is not None
    ]
    if not kb_scores:
        return True
    return max(kb_scores) < threshold


class SolicitationIndex:
    """Ephemeral, in-memory semantic index over the solicitation attachments."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.chunks: List[str] = []
        self.labels: List[str] = []
        self.by_label: Dict[str, str] = {}
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
        self.by_label = dict(zip(all_labels, all_chunks))
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

    def _gather_evidence(self, spec: SectionSpec) -> List[CitationEvidence]:
        sol = self.solicitation.search(spec.query, k=spec.sol_k)
        kb_hits = self.kb.search(spec.query, k=spec.kb_k)
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
        if spec.honest_transferable and kb_evidence_is_weak(evidence):
            instruction += (
                "\n\nIMPORTANT — the knowledge base has LIMITED directly-matching "
                "past performance for this requirement. Be honest about that: do not "
                "overstate or fabricate direct experience. Instead, identify the most "
                "RELEVANT and TRANSFERABLE capabilities/projects from the provided "
                "context and explain specifically how that experience transfers to "
                "this requirement. If coverage is genuinely thin, say so plainly and "
                "frame it as transferable/adjacent experience rather than a direct match."
            )
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
        if spec.honest_transferable and kb_evidence_is_weak(evidence):
            lines.append(
                "_Note: direct past-performance matches are limited. Frame the most "
                "relevant items below as **transferable experience** and be honest "
                "about coverage rather than claiming an exact match._"
            )
            lines.append("")
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

    def strengthen_section_for_requirement(
        self,
        section: ProposalSection,
        requirement: str,
        *,
        source_label: Optional[str] = None,
    ) -> ProposalSection:
        """Regenerate ``section`` so it better addresses a specific requirement.

        Keeps the section's existing substance (passed as ``prior``) and adds
        targeted feedback, so the rest of the draft is untouched.
        """
        spec = next((s for s in SECTION_SPECS if s.section_id == section.section_id), None)
        if spec is None:
            return section
        src = f" (see {source_label})" if source_label else ""
        feedback = (
            "Strengthen this section so it explicitly and unmistakably addresses "
            f'the following solicitation requirement{src}: "{requirement.strip()}". '
            "A reviewer should be able to see exactly how the proposal satisfies it. "
            "Preserve the section's existing strengths and stay grounded in the CONTEXT; "
            "do not invent capabilities."
        )
        return self.generate_section(spec, feedback=feedback, prior=section)

    # ------------------------------------------------------------------ #
    # Compliance matrix
    # ------------------------------------------------------------------ #
    def _compliance_evidence(self, k_per_query: int = 4) -> List[CitationEvidence]:
        """Dedup'd solicitation chunks most relevant to compliance (L/M/shall)."""
        seen: Dict[str, CitationEvidence] = {}
        for query in _COMPLIANCE_QUERIES:
            for ev in self.solicitation.search(query, k=k_per_query):
                if ev.label not in seen:
                    seen[ev.label] = ev
        return list(seen.values())

    def extract_compliance_matrix(
        self, draft: ProposalDraft, *, max_items: int = 15
    ) -> ComplianceMatrix:
        """Extract key requirements and map them to the draft's sections."""
        evidence = self._compliance_evidence()
        items: List[ComplianceItem] = []
        used_llm = False
        if evidence:
            if self.llm.available:
                items = self._extract_requirements_llm(evidence, max_items)
                used_llm = bool(items)
            if not items:
                items = self._extract_requirements_rules(evidence, max_items)
                used_llm = False
        self._map_requirements_to_sections(items, draft)
        return ComplianceMatrix(
            notice_id=draft.notice_id, items=items, used_llm=used_llm
        )

    def _extract_requirements_llm(
        self, evidence: List[CitationEvidence], max_items: int
    ) -> List[ComplianceItem]:
        labels = [e.label for e in evidence]
        context_parts = []
        for e in evidence:
            full = self.solicitation.by_label.get(e.label, e.snippet)
            context_parts.append(f"[{e.label}]\n{full[:1500]}")
        context = "\n\n".join(context_parts)
        instruction = (
            f"Extract up to {max_items} of the MOST IMPORTANT compliance "
            "requirements an offeror must satisfy, drawn ONLY from the CONTEXT. "
            "Prioritize Section L submission instructions, Section M evaluation "
            "factors, and mandatory 'shall'/'must' statements. Be concise and "
            "specific; do not invent requirements. Return a JSON object of the "
            'form {"requirements": [{"requirement": str, "category": one of '
            '["Section L","Section M","Evaluation","SOW (shall)","Other"], '
            '"source_label": one of the bracketed labels shown in the CONTEXT}]}. '
            "Use the exact source_label that the requirement came from."
        )
        try:
            result = self.llm.complete(
                instruction=instruction,
                context=context,
                sources=labels,
                max_tokens=1400,
                json_mode=True,
            )
        except LLMUnavailable as exc:
            logger.info("Compliance LLM extraction unavailable: %s", exc)
            return []

        raw = _parse_json_object(result.text)
        if not raw:
            return []
        reqs = raw.get("requirements") or raw.get("items") or []
        items: List[ComplianceItem] = []
        label_set = set(labels)
        for entry in reqs:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("requirement") or "").strip()
            if not text:
                continue
            category = _normalize_category(entry.get("category"))
            label = str(entry.get("source_label") or "").strip()
            if label not in label_set:
                # Model returned an unknown label; re-ground via semantic search.
                best = self.solicitation.search(text, k=1)
                label = best[0].label if best else (labels[0] if labels else "Solicitation")
            snippet = _snippet(self.solicitation.by_label.get(label, ""))
            items.append(
                ComplianceItem(
                    requirement=text,
                    category=category,
                    source_label=label,
                    source_snippet=snippet,
                )
            )
            if len(items) >= max_items:
                break
        return items

    def _extract_requirements_rules(
        self, evidence: List[CitationEvidence], max_items: int
    ) -> List[ComplianceItem]:
        """Regex-based fallback: pull 'shall'/'must'/evaluation sentences."""
        items: List[ComplianceItem] = []
        seen_text: set = set()
        # Evaluation/Section M language first, then general mandatory statements.
        for ev in evidence:
            full = self.solicitation.by_label.get(ev.label, ev.snippet)
            for match in _REQUIREMENT_RE.finditer(full):
                sentence = " ".join(match.group(0).split()).strip()
                if len(sentence) < 25 or len(sentence) > 400:
                    continue
                key = sentence.lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                items.append(
                    ComplianceItem(
                        requirement=sentence,
                        category=_categorize(sentence),
                        source_label=ev.label,
                        source_snippet=_snippet(full),
                    )
                )
                if len(items) >= max_items:
                    return _prioritize(items)
        return _prioritize(items)

    def remap_compliance(self, matrix: ComplianceMatrix, draft: ProposalDraft) -> None:
        """Re-derive each requirement's mapped section + status from current text.

        Preserves any status/notes the user manually edited away from the
        auto-default is *not* attempted here — call after section content changes
        (e.g. "strengthen") to refresh the auto-mapping.
        """
        self._map_requirements_to_sections(matrix.items, draft)

    def _map_requirements_to_sections(
        self, items: List[ComplianceItem], draft: ProposalDraft
    ) -> None:
        """Assign each requirement to its best-matching section + a status."""
        if not items or not draft.sections:
            return
        sections = draft.sections
        sec_texts = [f"{s.title}. {s.content}" for s in sections]
        sec_vecs = self.backend.embed(sec_texts)
        req_vecs = self.backend.embed([it.requirement for it in items])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = req_vecs @ sec_vecs.T
        sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
        for i, item in enumerate(items):
            row = sims[i]
            j = int(np.argmax(row))
            best = float(row[j])
            item.section_id = sections[j].section_id
            item.section_title = sections[j].title
            item.match_score = best
            if best >= _ADDRESSED_THRESHOLD:
                item.status = STATUS_ADDRESSED
            elif best >= _PARTIAL_THRESHOLD:
                item.status = STATUS_PARTIAL
            else:
                item.status = STATUS_NOT_ADDRESSED


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

    if draft.compliance and draft.compliance.items:
        _add_compliance_table_to_doc(document, draft.compliance)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _add_compliance_table_to_doc(document, matrix: ComplianceMatrix) -> None:
    """Append the compliance matrix as a Word table."""
    document.add_page_break()
    document.add_heading("Compliance Matrix", level=1)
    counts = matrix.status_counts()
    document.add_paragraph(
        f"Summary: {counts[STATUS_ADDRESSED]} addressed · "
        f"{counts[STATUS_PARTIAL]} partial · "
        f"{counts[STATUS_NOT_ADDRESSED]} not addressed. "
        "Status is an auto-generated starting point — review and adjust.",
        style="Intense Quote",
    )
    headers = ["#", "Requirement / Factor", "Category", "Source", "Section", "Status", "Notes"]
    table = document.add_table(rows=1, cols=len(headers))
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:  # pragma: no cover - style availability varies
        pass
    for cell, head in zip(table.rows[0].cells, headers):
        cell.text = head
    for i, item in enumerate(matrix.items, start=1):
        cells = table.add_row().cells
        cells[0].text = str(i)
        cells[1].text = item.requirement
        cells[2].text = item.category
        cells[3].text = item.source_label
        cells[4].text = item.section_title or "—"
        cells[5].text = item.status
        cells[6].text = item.notes or ""


def compliance_matrix_to_xlsx_bytes(matrix: ComplianceMatrix) -> bytes:
    """Render the compliance matrix to an .xlsx workbook (bytes)."""
    import pandas as pd

    rows = [
        {
            "#": i,
            "Requirement / Factor": item.requirement,
            "Category": item.category,
            "Source": item.source_label,
            "Source Excerpt": item.source_snippet,
            "Proposal Section": item.section_title or "",
            "Status": item.status,
            "Match Score": round(item.match_score, 3) if item.match_score is not None else "",
            "Notes": item.notes or "",
        }
        for i, item in enumerate(matrix.items, start=1)
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "#", "Requirement / Factor", "Category", "Source", "Source Excerpt",
            "Proposal Section", "Status", "Match Score", "Notes",
        ],
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Compliance Matrix")
        worksheet = writer.sheets["Compliance Matrix"]
        widths = {"A": 5, "B": 60, "C": 14, "D": 32, "E": 50, "F": 26, "G": 14, "H": 12, "I": 40}
        for col, width in widths.items():
            worksheet.column_dimensions[col].width = width
    return buffer.getvalue()

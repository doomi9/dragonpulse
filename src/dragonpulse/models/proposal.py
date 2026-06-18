"""Models for the grounded proposal generator.

A :class:`ProposalDraft` is a set of :class:`ProposalSection` objects. Each
section records not only its generated text but the exact evidence it was
grounded in — solicitation excerpts and knowledge-base chunks — as
:class:`CitationEvidence`. This keeps the whole draft auditable: every claim can
be traced to either the solicitation or one of the company's own documents.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Compliance status values (order = severity for sorting/summary).
STATUS_ADDRESSED = "Addressed"
STATUS_PARTIAL = "Partial"
STATUS_NOT_ADDRESSED = "Not Addressed"
COMPLIANCE_STATUSES = [STATUS_ADDRESSED, STATUS_PARTIAL, STATUS_NOT_ADDRESSED]


class CitationEvidence(BaseModel):
    """A single grounded source used by a section."""

    label: str          # e.g. "Solicitation: SOW.pdf #2" or "KB: Capabilities #1"
    origin: str         # "solicitation" | "knowledge_base" | "style_exemplar"
    snippet: str        # short excerpt for transparency
    score: Optional[float] = None


class ProposalSection(BaseModel):
    """One generated proposal section with its grounding and history."""

    model_config = ConfigDict(extra="ignore")

    section_id: str
    title: str
    content: str = ""
    sources: List[CitationEvidence] = Field(default_factory=list)
    used_llm: bool = False
    optional: bool = False
    feedback_history: List[str] = Field(default_factory=list)

    @property
    def solicitation_sources(self) -> List[CitationEvidence]:
        return [s for s in self.sources if s.origin == "solicitation"]

    @property
    def kb_sources(self) -> List[CitationEvidence]:
        return [
            s for s in self.sources
            if s.origin in ("knowledge_base", "style_exemplar")
        ]

    @property
    def style_sources(self) -> List[CitationEvidence]:
        """Knowledge-base chunks used as writing-style exemplars."""
        return [s for s in self.sources if s.origin == "style_exemplar"]

    def to_markdown(self) -> str:
        return f"## {self.title}\n\n{self.content.strip()}\n"


class ComplianceItem(BaseModel):
    """One extracted solicitation requirement and how the draft covers it."""

    model_config = ConfigDict(extra="ignore")

    requirement: str               # the requirement / evaluation factor text
    category: str = "Other"        # Section L | Section M | Evaluation | SOW (shall) | Other
    source_label: str              # citation, e.g. "Solicitation: SOW.pdf #3"
    source_snippet: str = ""       # excerpt the requirement was drawn from
    section_id: Optional[str] = None    # proposal section that addresses it
    section_title: Optional[str] = None
    status: str = STATUS_NOT_ADDRESSED  # Addressed | Partial | Not Addressed
    notes: str = ""                # user-editable comments
    match_score: Optional[float] = None  # semantic similarity to the mapped section


class ComplianceMatrix(BaseModel):
    """A focused requirement-traceability matrix for a proposal draft."""

    model_config = ConfigDict(extra="ignore")

    notice_id: str
    items: List[ComplianceItem] = Field(default_factory=list)
    used_llm: bool = False
    generated_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    )

    def status_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in COMPLIANCE_STATUSES}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_markdown(self) -> str:
        lines = [
            "## Compliance Matrix",
            "",
            "| # | Requirement / Factor | Category | Source | Proposal Section | Status | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for i, item in enumerate(self.items, start=1):
            req = item.requirement.replace("|", "\\|").replace("\n", " ")
            notes = (item.notes or "").replace("|", "\\|").replace("\n", " ")
            section = item.section_title or "—"
            lines.append(
                f"| {i} | {req} | {item.category} | {item.source_label} | "
                f"{section} | {item.status} | {notes} |"
            )
        counts = self.status_counts()
        lines += [
            "",
            f"_Summary: {counts[STATUS_ADDRESSED]} addressed · "
            f"{counts[STATUS_PARTIAL]} partial · "
            f"{counts[STATUS_NOT_ADDRESSED]} not addressed. Status is an "
            "auto-generated starting point — review and adjust._",
            "",
        ]
        return "\n".join(lines)


class ProposalDraft(BaseModel):
    """A full proposal draft for one opportunity."""

    model_config = ConfigDict(extra="ignore")

    notice_id: str
    opportunity_title: str
    agency: Optional[str] = None
    solicitation_number: Optional[str] = None
    sections: List[ProposalSection] = Field(default_factory=list)
    compliance: Optional[ComplianceMatrix] = None
    generated_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    )
    llm_model: Optional[str] = None
    grounded_note: str = (
        "All content is grounded in the cited solicitation excerpts and company "
        "knowledge-base documents. Review and verify before submission."
    )

    def get_section(self, section_id: str) -> Optional[ProposalSection]:
        for s in self.sections:
            if s.section_id == section_id:
                return s
        return None

    def replace_section(self, section: ProposalSection) -> None:
        for i, s in enumerate(self.sections):
            if s.section_id == section.section_id:
                self.sections[i] = section
                return
        self.sections.append(section)

    def to_markdown(self) -> str:
        header = [
            f"# Proposal Draft — {self.opportunity_title}",
            "",
            f"- **Notice ID:** {self.notice_id}",
        ]
        if self.solicitation_number:
            header.append(f"- **Solicitation #:** {self.solicitation_number}")
        if self.agency:
            header.append(f"- **Agency:** {self.agency}")
        header += [
            f"- **Generated:** {self.generated_at}",
            f"- **Drafting model:** {self.llm_model or 'grounded template (no LLM)'}",
            "",
            f"> {self.grounded_note}",
            "",
            "---",
            "",
        ]
        body = "\n".join(s.to_markdown() for s in self.sections)
        out = "\n".join(header) + body
        sourced = [(s.title, s.sources) for s in self.sections if s.sources]
        if sourced:
            out += "\n\n---\n\n## Document Sources and References\n\n"
            for title, sources in sourced:
                out += f"### {title}\n\n"
                for src in sources:
                    out += f"- {src.label}\n"
                out += "\n"
        if self.compliance and self.compliance.items:
            out += "\n\n---\n\n" + self.compliance.to_markdown()
        return out


class SavedDraft(BaseModel):
    """A named, persisted snapshot of a :class:`ProposalDraft`."""

    model_config = ConfigDict(extra="ignore")

    draft_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    notice_id: str
    opportunity_title: str = ""
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    )
    modified_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    )
    version: int = 1
    draft: ProposalDraft

    def touch(self) -> None:
        """Bump the modified timestamp and version (a new save of same draft)."""
        self.modified_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        self.version += 1

    @property
    def section_count(self) -> int:
        return len(self.draft.sections)

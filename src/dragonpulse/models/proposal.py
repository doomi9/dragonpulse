"""Models for the grounded proposal generator.

A :class:`ProposalDraft` is a set of :class:`ProposalSection` objects. Each
section records not only its generated text but the exact evidence it was
grounded in — solicitation excerpts and knowledge-base chunks — as
:class:`CitationEvidence`. This keeps the whole draft auditable: every claim can
be traced to either the solicitation or one of the company's own documents.
"""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CitationEvidence(BaseModel):
    """A single grounded source used by a section."""

    label: str          # e.g. "Solicitation: SOW.pdf #2" or "KB: Capabilities #1"
    origin: str         # "solicitation" | "knowledge_base"
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
        return [s for s in self.sources if s.origin == "knowledge_base"]

    def to_markdown(self) -> str:
        lines = [f"## {self.title}", "", self.content.strip(), ""]
        if self.sources:
            lines.append("**Sources:**")
            for s in self.sources:
                lines.append(f"- [{s.label}]")
            lines.append("")
        return "\n".join(lines)


class ProposalDraft(BaseModel):
    """A full proposal draft for one opportunity."""

    model_config = ConfigDict(extra="ignore")

    notice_id: str
    opportunity_title: str
    agency: Optional[str] = None
    solicitation_number: Optional[str] = None
    sections: List[ProposalSection] = Field(default_factory=list)
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
        return "\n".join(header) + body

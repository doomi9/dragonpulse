"""Outreach email drafting ("Who to Reach Out To").

Produces a personalized, professional outreach email to an opportunity's point
of contact. Like the checklist, this is **grounded**: every fact in the draft
comes from the opportunity metadata or the user-supplied company profile, and
the result records its sources.

- With no LLM: a clean, deterministic template is filled from metadata.
- With an opted-in LLM: a more natural draft is generated, still constrained to
  the provided context and required to cite it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.common import PointOfContact
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.processors.llm import LLMClient, LLMUnavailable

logger = get_logger(__name__)


@dataclass
class CompanyProfile:
    """Minimal sender profile used to personalize outreach.

    Defaults are placeholders; the UI lets the user edit these. Nothing here is
    persisted to disk by default.
    """

    company_name: str = "[Your Company]"
    sender_name: str = "[Your Name]"
    sender_title: str = "[Your Title]"
    sender_email: str = "[your.email@company.com]"
    sender_phone: str = "[Your Phone]"
    capabilities: str = "[your core capabilities]"


@dataclass
class OutreachDraft:
    """A draft outreach email with provenance."""

    subject: str
    body: str
    to_name: Optional[str]
    to_email: Optional[str]
    used_llm: bool
    sources: List[str] = field(default_factory=list)


def _subject(opp: Opportunity) -> str:
    ref = opp.solicitation_number or opp.notice_id
    title = opp.title or "your recent opportunity"
    return f"Question regarding {title} ({ref})"


def build_template_draft(
    opp: Opportunity, poc: PointOfContact, profile: CompanyProfile
) -> OutreachDraft:
    """Deterministic, fully-grounded outreach draft (no LLM)."""
    greeting_name = poc.full_name or "Contracting Officer"
    ref = opp.solicitation_number or opp.notice_id
    agency = opp.agency or "your agency"
    naics = f" (NAICS {opp.naics_code})" if opp.naics_code else ""
    deadline = opp.response_deadline
    deadline_line = (
        f"We note the response deadline of {deadline.strftime('%B %d, %Y')}. "
        if deadline
        else ""
    )

    body = (
        f"Dear {greeting_name},\n\n"
        f"My name is {profile.sender_name}, {profile.sender_title} at "
        f"{profile.company_name}. I am writing regarding the opportunity "
        f'"{opp.title or ref}" ({ref}) posted by {agency}{naics}.\n\n'
        f"{profile.company_name} specializes in {profile.capabilities}, which aligns "
        f"closely with the requirement described in this notice. {deadline_line}"
        "We would welcome the opportunity to confirm a few details to ensure our "
        "response fully addresses your needs:\n\n"
        "  1. Could you confirm the preferred submission method and format?\n"
        "  2. Are there any anticipated amendments or a scheduled industry day?\n"
        "  3. Is the set-aside designation final as posted?\n\n"
        "Thank you for your time and for the information provided in this notice. "
        "Please let me know if a brief call would be helpful.\n\n"
        "Best regards,\n"
        f"{profile.sender_name}\n"
        f"{profile.sender_title}, {profile.company_name}\n"
        f"{profile.sender_email} | {profile.sender_phone}"
    )

    sources = ["opportunity metadata: title, solicitationNumber, agency"]
    if opp.naics_code:
        sources.append("opportunity metadata: naicsCode")
    if deadline:
        sources.append("opportunity metadata: responseDeadLine")
    if poc.full_name:
        sources.append("opportunity metadata: pointOfContact.fullName")

    return OutreachDraft(
        subject=_subject(opp),
        body=body,
        to_name=poc.full_name,
        to_email=poc.email,
        used_llm=False,
        sources=sources,
    )


def _context(opp: Opportunity, poc: PointOfContact, profile: CompanyProfile) -> str:
    deadline = opp.response_deadline
    return "\n".join(
        [
            "== Opportunity ==",
            f"Title: {opp.title or 'N/A'}",
            f"Solicitation/Notice: {opp.solicitation_number or opp.notice_id}",
            f"Agency: {opp.agency or 'N/A'}",
            f"Office: {opp.office or 'N/A'}",
            f"NAICS: {opp.naics_code or 'N/A'}",
            f"Set-aside: {opp.set_aside_description or 'None'}",
            f"Response deadline: {deadline.strftime('%Y-%m-%d %H:%M') if deadline else 'N/A'}",
            "== Contact ==",
            f"Name: {poc.full_name or 'N/A'}",
            f"Title: {poc.title or 'N/A'}",
            f"Email: {poc.email or 'N/A'}",
            "== Sender (our company) ==",
            f"Company: {profile.company_name}",
            f"Sender: {profile.sender_name}, {profile.sender_title}",
            f"Capabilities: {profile.capabilities}",
            f"Contact: {profile.sender_email} | {profile.sender_phone}",
        ]
    )


def generate_outreach_email(
    opp: Opportunity,
    poc: PointOfContact,
    profile: Optional[CompanyProfile] = None,
    *,
    use_llm: bool = False,
    llm: Optional[LLMClient] = None,
) -> OutreachDraft:
    """Generate an outreach draft, using the LLM only if available and requested.

    Falls back to the deterministic template on any LLM error, so the UI always
    gets a usable draft.
    """
    profile = profile or CompanyProfile()
    template = build_template_draft(opp, poc, profile)
    if not use_llm:
        return template

    llm = llm or LLMClient()
    if not llm.available:
        return template

    try:
        instruction = (
            "Write a concise, professional outreach email (120-180 words) from the "
            "sender to the contact about this opportunity. Use ONLY facts in the "
            "CONTEXT. Do not fabricate capabilities, past performance, or details. "
            "Include 2-3 specific, relevant clarifying questions. Cite context facts "
            "in brackets where natural. Return only the email body (no subject line)."
        )
        result = llm.complete(
            instruction=instruction,
            context=_context(opp, poc, profile),
            sources=["opportunity metadata", "company profile"],
            max_tokens=500,
        )
    except LLMUnavailable as exc:
        logger.info("Outreach LLM unavailable, using template: %s", exc)
        return template

    return OutreachDraft(
        subject=_subject(opp),
        body=result.text or template.body,
        to_name=poc.full_name,
        to_email=poc.email,
        used_llm=True,
        sources=result.sources,
    )

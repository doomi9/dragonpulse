"""Action-checklist generation ("What Needs to Be Done").

Hybrid approach
---------------
1. **Rules engine (always on, deterministic, grounded):** maps the opportunity's
   notice type and metadata to a concrete, dated action checklist. Every item
   cites the opportunity fact that produced it.
2. **LLM enrichment (optional):** when the user has opted into an LLM, we ask it
   to *add* tailored, opportunity-specific steps — still grounded in, and citing,
   the provided context. The rule-based items are never discarded.

This guarantees a useful, source-cited checklist even with no LLM, while
allowing richer tailoring when one is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.common import NoticeType
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.processors.llm import LLMClient, LLMUnavailable

logger = get_logger(__name__)


@dataclass
class ChecklistItem:
    """A single, grounded action item."""

    action: str
    detail: str
    due: Optional[str] = None  # ISO date or human phrase
    priority: str = "normal"  # high | normal | low
    source: str = "rules engine"
    done: bool = False


# Notice-type-specific guidance. Each entry is (action, detail, priority).
_NOTICE_PLAYBOOK = {
    NoticeType.SOURCES_SOUGHT.value: [
        (
            "Decide whether to submit a capability statement",
            "Sources Sought / RFI responses are market research, not bids. "
            "Responding signals interest and can shape the eventual solicitation.",
            "high",
        ),
        (
            "Tailor your capability statement to the stated requirement",
            "Mirror the keywords and NAICS in the notice; highlight relevant past performance.",
            "high",
        ),
        (
            "Ask clarifying questions to the listed POC",
            "Use the contact(s) below to confirm scope, anticipated set-aside, and timeline.",
            "normal",
        ),
    ],
    NoticeType.PRESOLICITATION.value: [
        (
            "Confirm the anticipated solicitation release date",
            "Presolicitations announce an upcoming RFP/RFQ. Track when the solicitation drops.",
            "high",
        ),
        (
            "Begin teaming / subcontractor outreach early",
            "Use the lead time before solicitation to line up partners and key personnel.",
            "normal",
        ),
    ],
    NoticeType.SOLICITATION.value: [
        (
            "Read Sections L (instructions) and M (evaluation) first",
            "These define how to submit and how you are scored. Build the proposal "
            "outline from Section M.",
            "high",
        ),
        (
            "Build a compliance matrix from the SOW/PWS",
            "Extract every 'shall' requirement and map it to a proposal response.",
            "high",
        ),
        (
            "Submit clarifying questions before the Q&A deadline",
            "Note the question cutoff; it is usually well before the proposal due date.",
            "high",
        ),
        (
            "Verify registrations are active",
            "Confirm SAM.gov registration, required certs, and that the set-aside "
            "eligibility applies to you.",
            "normal",
        ),
    ],
    NoticeType.COMBINED_SYNOPSIS_SOLICITATION.value: [
        (
            "Treat as a live solicitation — quote may be due soon",
            "Combined Synopsis/Solicitations (often FAR 13 simplified) can have short turnarounds.",
            "high",
        ),
        (
            "Build a line-item compliant quote",
            "Address every CLIN and submission instruction exactly as stated.",
            "high",
        ),
    ],
    NoticeType.SPECIAL_NOTICE.value: [
        (
            "Determine the notice's purpose",
            "Special Notices cover industry days, pre-bid conferences, or program "
            "updates. Identify the ask.",
            "normal",
        ),
        (
            "Register for any event / RSVP",
            "If an industry day or site visit is mentioned, register before the stated cutoff.",
            "normal",
        ),
    ],
    NoticeType.AWARD_NOTICE.value: [
        (
            "Capture competitive intelligence",
            "Record the awardee, amount, and NAICS for your pricing history and future bids.",
            "normal",
        ),
        (
            "Consider a post-award debrief request (if you bid)",
            "If you competed and lost, request a debrief within the FAR-allowed window.",
            "normal",
        ),
    ],
}

# Generic items appended for every active opportunity.
_GENERIC_TAIL = [
    (
        "Confirm submission method and format",
        "Check whether the response goes via email, SAM.gov, or a portal, and the "
        "required file formats.",
        "normal",
    ),
    (
        "Download and review all attachments",
        "Pull every resource link (SOW/PWS, attachments, amendments) and scan for requirements.",
        "high",
    ),
]


def _deadline_items(opp: Opportunity) -> List[ChecklistItem]:
    """Build date-anchored items from the response deadline, if present."""
    items: List[ChecklistItem] = []
    deadline = opp.response_deadline
    if deadline is None:
        return items

    days_left = opp.days_until_deadline()
    due_str = deadline.strftime("%Y-%m-%d %H:%M")
    priority = "high" if (days_left is not None and days_left <= 7) else "normal"
    overdue = days_left is not None and days_left < 0

    items.append(
        ChecklistItem(
            action="Submit the response before the deadline" if not overdue
            else "Deadline has passed — verify if still open",
            detail=(
                f"Response deadline: {due_str}."
                + (f" ~{days_left} day(s) remaining." if days_left is not None else "")
            ),
            due=deadline.strftime("%Y-%m-%d"),
            priority="high" if overdue else priority,
            source="opportunity metadata: responseDeadLine",
        )
    )

    # Suggest an internal "pencils down" milestone two days before the deadline.
    if days_left is not None and days_left > 2:
        pencils_down = (deadline - timedelta(days=2)).strftime("%Y-%m-%d")
        items.append(
            ChecklistItem(
                action="Internal 'pencils down' / final review",
                detail="Reserve the final 48 hours for compliance review and submission mechanics.",
                due=pencils_down,
                priority="normal",
                source="rules engine (derived from responseDeadLine)",
            )
        )
    return items


def build_rule_checklist(opp: Opportunity) -> List[ChecklistItem]:
    """Build the deterministic, fully-grounded base checklist."""
    items: List[ChecklistItem] = []

    # 1) Deadline-anchored items.
    items.extend(_deadline_items(opp))

    # 2) Notice-type playbook.
    code = None
    if opp.notice_type:
        # Map human label back to a code, else try direct code.
        for nt in NoticeType:
            if nt.label.lower() == opp.notice_type.lower() or nt.value == opp.notice_type:
                code = nt.value
                break
    playbook = _NOTICE_PLAYBOOK.get(code, [])
    src_label = f"notice type: {opp.notice_type}" if opp.notice_type else "notice type: unknown"
    for action, detail, priority in playbook:
        items.append(
            ChecklistItem(action=action, detail=detail, priority=priority, source=src_label)
        )

    # 3) Set-aside eligibility reminder, grounded in the metadata.
    if opp.set_aside_description or opp.set_aside_code:
        sa = opp.set_aside_description or opp.set_aside_code
        items.append(
            ChecklistItem(
                action="Confirm set-aside eligibility",
                detail=(
                    f"This is restricted to: {sa}. Verify your firm qualifies "
                    "before investing effort."
                ),
                priority="high",
                source="opportunity metadata: typeOfSetAside",
            )
        )

    # 4) Generic tail.
    for action, detail, priority in _GENERIC_TAIL:
        items.append(
            ChecklistItem(action=action, detail=detail, priority=priority, source="rules engine")
        )

    return items


def _opp_context(opp: Opportunity) -> str:
    """Compact, grounded context block describing the opportunity for the LLM."""
    deadline = opp.response_deadline
    lines = [
        f"Title: {opp.title or 'N/A'}",
        f"Notice ID: {opp.notice_id}",
        f"Notice type: {opp.notice_type or 'N/A'}",
        f"Agency: {opp.agency or 'N/A'}",
        f"Office: {opp.office or 'N/A'}",
        f"NAICS: {opp.naics_code or 'N/A'}",
        f"Set-aside: {opp.set_aside_description or opp.set_aside_code or 'None'}",
        f"Posted: {opp.posted_date_raw or 'N/A'}",
        f"Response deadline: {deadline.strftime('%Y-%m-%d %H:%M') if deadline else 'N/A'}",
        f"Number of attachments: {len(opp.resource_links)}",
    ]
    return "\n".join(lines)


def enrich_with_llm(
    opp: Opportunity, base_items: List[ChecklistItem], llm: Optional[LLMClient] = None
) -> List[ChecklistItem]:
    """Optionally append LLM-suggested, grounded items. Never raises."""
    llm = llm or LLMClient()
    if not llm.available:
        return base_items
    try:
        context = _opp_context(opp)
        instruction = (
            "Based ONLY on the opportunity metadata in the CONTEXT, suggest up to "
            "4 additional, specific action items a contractor should take that are "
            "not already obvious. For each, return one line as 'ACTION :: DETAIL'. "
            "Cite the relevant context fact in brackets within the DETAIL."
        )
        result = llm.complete(
            instruction=instruction,
            context=context,
            sources=["opportunity metadata"],
            max_tokens=400,
        )
    except LLMUnavailable as exc:
        logger.info("Skipping LLM checklist enrichment: %s", exc)
        return base_items

    extra: List[ChecklistItem] = []
    for line in result.text.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if "::" not in line:
            continue
        action, _, detail = line.partition("::")
        extra.append(
            ChecklistItem(
                action=action.strip(),
                detail=detail.strip(),
                priority="normal",
                source=f"LLM ({result.model}), grounded in opportunity metadata",
            )
        )
    return base_items + extra


def build_checklist(
    opp: Opportunity,
    *,
    use_llm: bool = False,
    llm: Optional[LLMClient] = None,
) -> List[ChecklistItem]:
    """Public entry point: rules-based checklist, optionally LLM-enriched."""
    items = build_rule_checklist(opp)
    if use_llm:
        items = enrich_with_llm(opp, items, llm=llm)
    return items

"""Load a SAM.gov opportunity from its public page — **zero keyed-API calls**.

A sam.gov opportunity URL points at an Angular single-page app whose HTML carries
no data; the page is populated in the browser from SAM.gov's *public frontend*
JSON endpoints (``https://sam.gov/api/prod/...``). Those are the same calls the
website makes for any visitor — they are **not** the rate-limited
``api.sam.gov`` / data.gov endpoints and do **not** consume the user's API key
budget.

This module fetches those public endpoints and parses them into an
:class:`~dragonpulse.models.opportunity.Opportunity`, so a user who is out of API
requests (or just found something on SAM.gov) can paste the link and start
drafting immediately.

Everything here is best-effort and defensive: SAM.gov's frontend shapes vary by
notice type and change over time, so missing fields degrade gracefully rather
than raising.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import requests

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.common import (
    SET_ASIDE_CHOICES,
    Address,
    NoticeType,
    PointOfContact,
    ResourceLink,
)
from dragonpulse.models.opportunity import Opportunity

logger = get_logger(__name__)

# Public SAM.gov frontend endpoints (no API key; not the data.gov budget).
_FRONTEND_OPP = "https://sam.gov/api/prod/opps/v2/opportunities/{oid}"
_FRONTEND_ORG = "https://sam.gov/api/prod/federalorganizations/v1/organizations/{org}"
_FRONTEND_RES = "https://sam.gov/api/prod/opps/v3/opportunities/{oid}/resources"
_FRONTEND_DL = "https://sam.gov/api/prod/opps/v3/opportunities/resources/files/{rid}/download"
_FRONTEND_SEARCH = "https://sam.gov/api/prod/sgs/v1/search/"
_PUBLIC_VIEW = "https://sam.gov/opp/{oid}/view"

_OPP_ID_RE = re.compile(r"/opp/([A-Za-z0-9]+)")
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9]{16,}$")
_USER_AGENT = "Mozilla/5.0 (compatible; DragonPulse/1.0)"
_DEFAULT_TIMEOUT = 30
_MAX_DESCRIPTION_CHARS = 40_000

# data2.type single-letter code -> human label (reuse the canonical mapping).
_NOTICE_TYPE_BY_CODE = {nt.value: nt.label for nt in NoticeType}


class SamScrapeError(RuntimeError):
    """Raised when a SAM.gov link cannot be fetched or parsed."""


@dataclass
class ScrapedOpportunity:
    """Result of loading an opportunity from a SAM.gov link."""

    opportunity: Opportunity
    description: str = ""
    attachments: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Link parsing
# --------------------------------------------------------------------------- #
def parse_sam_link(raw: str) -> str:
    """Extract the opportunity id from a SAM.gov URL (or a bare id).

    Accepts the workspace URL
    (``https://sam.gov/workspace/contract/opp/<id>/view``), the classic
    ``https://sam.gov/opp/<id>/view`` form, or a bare opportunity id.

    Raises
    ------
    SamScrapeError
        If no opportunity id can be found.
    """
    text = (raw or "").strip()
    if not text:
        raise SamScrapeError("Paste a SAM.gov opportunity link.")
    match = _OPP_ID_RE.search(text)
    if match:
        return match.group(1)
    if _BARE_ID_RE.match(text):
        return text
    raise SamScrapeError(
        "That doesn't look like a SAM.gov opportunity link. Expected something "
        "like https://sam.gov/workspace/contract/opp/<ID>/view"
    )


# --------------------------------------------------------------------------- #
# Fetch + parse
# --------------------------------------------------------------------------- #
def fetch_opportunity_from_link(
    raw: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    resolve_org: bool = True,
    fetch_attachments: bool = True,
) -> ScrapedOpportunity:
    """Load an :class:`Opportunity` from a SAM.gov link via public page data.

    No rate-limited ``api.sam.gov`` calls are made; only the public frontend
    endpoints the website itself uses.

    Raises
    ------
    SamScrapeError
        On an invalid link, network failure, or unparseable response.
    """
    oid = parse_sam_link(raw)
    http = session or requests.Session()
    # SAM.gov's frontend returns 406 for an explicit application/json Accept;
    # a permissive Accept matches what the browser sends.
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}

    try:
        resp = http.get(_FRONTEND_OPP.format(oid=oid), headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise SamScrapeError(f"Couldn't reach SAM.gov: {exc}") from exc
    if resp.status_code == 404:
        raise SamScrapeError(
            "Couldn't find that opportunity on SAM.gov. The link may be wrong, or "
            "the notice may have been archived or removed."
        )
    if resp.status_code >= 400:
        raise SamScrapeError(
            f"SAM.gov returned HTTP {resp.status_code} for that link. Try again, or "
            "upload the solicitation PDF instead."
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise SamScrapeError(
            "SAM.gov did not return readable opportunity data for that link."
        ) from exc

    data2 = payload.get("data2") or {}
    if not data2:
        raise SamScrapeError(
            "That page didn't contain opportunity data. Double-check the link points "
            "to a single opportunity (…/opp/<ID>/view)."
        )

    title = _clean(data2.get("title"))
    sol_number = _clean(data2.get("solicitationNumber"))
    naics = _first_naics(data2.get("naics"))
    notice_label = _NOTICE_TYPE_BY_CODE.get(str(data2.get("type") or "").lower())
    sa_code, sa_desc = _set_aside(data2)
    deadline = _deadline(data2)
    archive_date = _clean((data2.get("archive") or {}).get("date"))
    posted = _clean(payload.get("postedDate"))
    pocs = _points_of_contact(data2.get("pointOfContact"))
    place = _place_of_performance(data2.get("placeOfPerformance"))
    description = _description(payload)

    org_path: Optional[str] = None
    if resolve_org and data2.get("organizationId"):
        org_path = _resolve_org(http, str(data2["organizationId"]), headers, timeout)

    res_links: List[ResourceLink] = []
    if fetch_attachments:
        res_links = _attachments(http, oid, headers, timeout)

    opp = Opportunity(
        notice_id=oid,
        title=title or f"SAM.gov opportunity {oid}",
        solicitation_number=sol_number,
        full_parent_path_name=org_path,
        notice_type=notice_label,
        base_type=notice_label,
        naics_code=naics,
        set_aside_code=sa_code,
        set_aside_description=sa_desc,
        posted_date_raw=posted,
        response_deadline_raw=deadline,
        archive_date_raw=archive_date,
        points_of_contact=pocs,
        place_of_performance=place,
        resource_links=res_links,
        ui_link=_PUBLIC_VIEW.format(oid=oid),
        manual_entry=True,
        loaded_via="sam_link",
    )
    logger.info(
        "Scraped SAM.gov opportunity %s (naics=%s, attachments=%d) — no keyed API used",
        oid, naics, len(res_links),
    )
    return ScrapedOpportunity(
        opportunity=opp,
        description=description,
        attachments=[r.name or r.url for r in res_links],
    )


# --------------------------------------------------------------------------- #
# Field helpers (all defensive)
# --------------------------------------------------------------------------- #
def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_naics(naics) -> Optional[str]:
    if not isinstance(naics, list):
        return None
    # Prefer the primary entry; codes are nested as {"code": ["237130"], ...}.
    ordered = sorted(naics, key=lambda n: 0 if (n or {}).get("type") == "primary" else 1)
    for entry in ordered:
        code = (entry or {}).get("code")
        if isinstance(code, list) and code:
            return _clean(code[0])
        if isinstance(code, str):
            return _clean(code)
    return None


def _set_aside(data2: dict) -> "tuple[Optional[str], Optional[str]]":
    sol = data2.get("solicitation") or {}
    code = (
        data2.get("typeOfSetAside")
        or sol.get("setAside")
        or sol.get("setAsideType")
        or data2.get("setAside")
    )
    code = _clean(code)
    if not code:
        return None, None
    return code, SET_ASIDE_CHOICES.get(code, code)


def _deadline(data2: dict) -> Optional[str]:
    deadlines = (data2.get("solicitation") or {}).get("deadlines") or {}
    return _clean(deadlines.get("response"))


def _points_of_contact(pocs) -> List[PointOfContact]:
    out: List[PointOfContact] = []
    if not isinstance(pocs, list):
        return out
    for poc in pocs:
        if isinstance(poc, dict):
            try:
                out.append(PointOfContact.model_validate(poc))
            except Exception:  # noqa: BLE001 - skip malformed contacts
                continue
    return out


def _place_of_performance(pop) -> Optional[Address]:
    if not isinstance(pop, dict) or not pop:
        return None

    def _name(v):
        return v.get("name") or v.get("code") if isinstance(v, dict) else v

    try:
        return Address(
            city=_name(pop.get("city")),
            state=_name(pop.get("state")),
            zipcode=_clean(pop.get("zip")),
            country_code=_name(pop.get("country")),
        )
    except Exception:  # noqa: BLE001
        return None


def _description(payload: dict) -> str:
    desc = payload.get("description")
    body = ""
    if isinstance(desc, list) and desc:
        body = (desc[0] or {}).get("body") or ""
    elif isinstance(desc, dict):
        body = desc.get("body") or ""
    elif isinstance(desc, str):
        body = desc
    return _html_to_text(body)[:_MAX_DESCRIPTION_CHARS]


def _html_to_text(raw_html: str) -> str:
    """Convert SAM's HTML description body to readable plain text."""
    if not raw_html:
        return ""
    text = raw_html
    # Preserve block structure before stripping tags.
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</li\s*>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "• ", text)
    text = re.sub(r"<[^>]+>", "", text)  # drop remaining tags
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    # Collapse excess blank lines / trailing spaces.
    lines = [ln.strip() for ln in text.splitlines()]
    out: List[str] = []
    blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- #
# Keyless search crawl (public frontend) — used as a fallback when the keyed
# data.gov budget is exhausted.
# --------------------------------------------------------------------------- #
def search_opportunities_via_frontend(
    query: str,
    *,
    naics_codes: Optional[List[str]] = None,
    posted_from: Optional[date] = None,
    posted_to: Optional[date] = None,
    limit: int = 25,
    active_only: bool = True,
    session: Optional[requests.Session] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> List[Opportunity]:
    """Search SAM.gov's **public frontend** for ``query`` — no keyed API used.

    This hits the same JSON search endpoint the sam.gov website uses for any
    visitor, so it does **not** consume the rate-limited data.gov budget. Results
    are mapped to :class:`Opportunity` records (flagged ``loaded_via='sam_crawl'``)
    and filtered to the posted-date window client-side. Best-effort: any failure
    returns an empty list rather than raising.
    """
    http = session or requests.Session()
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    params = {
        "index": "opp",
        "page": 0,
        "size": max(1, min(limit, 100)),
        "sort": "-modifiedDate",
        "mode": "search",
        "q": query,
        "qMode": "ALL",
    }
    if active_only:
        params["is_active"] = "true"
    if naics_codes:
        params["naics"] = ",".join(naics_codes)

    try:
        resp = http.get(_FRONTEND_SEARCH, params=params, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            logger.info("Frontend search HTTP %s for %r", resp.status_code, query)
            return []
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.info("Frontend search failed for %r: %s", query, exc)
        return []

    results = (data.get("_embedded") or {}).get("results") or []
    out: List[Opportunity] = []
    for raw in results:
        opp = _opp_from_search_result(raw)
        if opp is None:
            continue
        if (posted_from or posted_to) and not _within_window(opp, posted_from, posted_to):
            continue
        out.append(opp)
    logger.info("Frontend crawl for %r → %d opportunities (no keyed API)", query, len(out))
    return out


def _within_window(opp: Opportunity, start: Optional[date], end: Optional[date]) -> bool:
    posted = opp.posted_date
    if posted is None:
        return True  # don't drop undated results
    day = posted.date()
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def _opp_from_search_result(raw: dict) -> Optional[Opportunity]:
    """Map one public search-result record to an :class:`Opportunity`."""
    if not isinstance(raw, dict):
        return None
    oid = _clean(raw.get("_id") or raw.get("id") or raw.get("noticeId"))
    if not oid:
        return None
    type_obj = raw.get("type") or {}
    notice_label = None
    if isinstance(type_obj, dict):
        notice_label = _clean(type_obj.get("value")) or _NOTICE_TYPE_BY_CODE.get(
            str(type_obj.get("code") or "").lower()
        )
    agency = _agency_from_hierarchy(raw.get("organizationHierarchy"))
    naics = _first_naics(raw.get("naics")) or _clean(raw.get("naicsCode"))
    sa_code = _clean(raw.get("typeOfSetAside"))
    sa_desc = _clean(raw.get("typeOfSetAsideDescription")) or (
        SET_ASIDE_CHOICES.get(sa_code, sa_code) if sa_code else None
    )
    try:
        return Opportunity(
            notice_id=oid,
            title=_clean(raw.get("title")) or f"SAM.gov opportunity {oid}",
            solicitation_number=_clean(raw.get("solicitationNumber")),
            full_parent_path_name=agency,
            notice_type=notice_label,
            base_type=notice_label,
            naics_code=naics,
            set_aside_code=sa_code,
            set_aside_description=sa_desc,
            posted_date_raw=_clean(raw.get("publishDate")),
            response_deadline_raw=_clean(raw.get("responseDate")),
            ui_link=_PUBLIC_VIEW.format(oid=oid),
            manual_entry=True,
            loaded_via="sam_crawl",
        )
    except Exception:  # noqa: BLE001 - skip malformed records
        return None


def _agency_from_hierarchy(hierarchy) -> Optional[str]:
    """Join organization hierarchy names into a 'DEPT.SUBAGENCY.OFFICE' path."""
    if not isinstance(hierarchy, list) or not hierarchy:
        return None
    names = []
    for level in sorted(hierarchy, key=lambda h: (h or {}).get("level", 0)):
        name = _clean((level or {}).get("name"))
        if name:
            names.append(name)
    return ".".join(names) if names else None


def _resolve_org(http, org_id: str, headers: dict, timeout: int) -> Optional[str]:
    """Resolve an organizationId to its full parent path name (best-effort)."""
    try:
        resp = http.get(
            _FRONTEND_ORG.format(org=org_id),
            params={"sort": "name"},
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    embedded = data.get("_embedded")
    org = None
    if isinstance(embedded, list) and embedded:
        org = (embedded[0] or {}).get("org")
    elif isinstance(embedded, dict):
        org = embedded.get("org")
    if isinstance(org, dict):
        return _clean(org.get("fullParentPathName")) or _clean(org.get("name"))
    return None


def _attachments(http, oid: str, headers: dict, timeout: int) -> List[ResourceLink]:
    """List attachments from the public resources endpoint (best-effort)."""
    try:
        resp = http.get(
            _FRONTEND_RES.format(oid=oid),
            params={"withScanResult": "false", "excludeDeleted": "true"},
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            return []
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    links: List[ResourceLink] = []
    embedded = data.get("_embedded") or {}
    groups = embedded.get("opportunityAttachmentList") or []
    if isinstance(groups, dict):
        groups = [groups]
    for group in groups:
        for att in (group or {}).get("attachments") or []:
            if not isinstance(att, dict):
                continue
            rid = att.get("resourceId") or att.get("id")
            name = _clean(att.get("name"))
            if att.get("type") == "link" and att.get("uri"):
                links.append(ResourceLink(url=att["uri"], name=name or att["uri"]))
            elif rid:
                links.append(
                    ResourceLink(url=_FRONTEND_DL.format(rid=rid), name=name)
                )
        for url in (group or {}).get("resourceLinks") or []:
            if isinstance(url, str):
                links.append(ResourceLink.from_url(url))
    return links

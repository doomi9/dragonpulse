"""Opportunity models for the SAM.gov Opportunities v2 API.

The Opportunities v2 ``/search`` endpoint returns records under the
``opportunitiesData`` key. Field names use camelCase; we alias them to
snake_case and keep everything optional because coverage varies widely by
notice type.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dragonpulse.models.common import Address, PointOfContact, ResourceLink, _Tolerant


def _parse_dt(value: Any) -> Optional[datetime]:
    """Best-effort parse of the various date formats SAM.gov emits."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    # Common SAM formats, in order of likelihood.
    fmts = (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
    )
    # Normalize a trailing "Z" to +0000 for %z parsing.
    normalized = text.replace("Z", "+0000")
    for fmt in fmts:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    try:  # last resort: ISO parser
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class AwardSummary(_Tolerant):
    """The ``award`` sub-object present on award notices.

    SAM.gov nests the awardee as ``{"name": ..., "ueiSAM": ...}`` under an
    ``awardee`` key; :meth:`_flatten_awardee` lifts those into flat fields.
    """

    number: Optional[str] = None
    amount: Optional[str] = None
    date: Optional[str] = None
    awardee_name: Optional[str] = None
    awardee_uei: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_awardee(cls, data: Any) -> Any:
        if isinstance(data, dict):
            awardee = data.get("awardee")
            if isinstance(awardee, dict):
                data = dict(data)  # avoid mutating caller's payload
                data.setdefault("awardee_name", awardee.get("name"))
                data.setdefault(
                    "awardee_uei", awardee.get("ueiSAM") or awardee.get("uei")
                )
            elif isinstance(awardee, str):
                data = dict(data)
                data.setdefault("awardee_name", awardee)
        return data


class Opportunity(BaseModel):
    """A single SAM.gov opportunity notice.

    Only ``notice_id`` is effectively guaranteed; everything else is optional to
    survive the API's inconsistencies. Use the convenience properties for UI.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    notice_id: str = Field(alias="noticeId")
    title: Optional[str] = None
    solicitation_number: Optional[str] = Field(default=None, alias="solicitationNumber")

    # Organization hierarchy.
    full_parent_path_name: Optional[str] = Field(default=None, alias="fullParentPathName")
    full_parent_path_code: Optional[str] = Field(default=None, alias="fullParentPathCode")
    organization_type: Optional[str] = Field(default=None, alias="organizationType")

    # Classification.
    notice_type: Optional[str] = Field(default=None, alias="type")
    base_type: Optional[str] = Field(default=None, alias="baseType")
    naics_code: Optional[str] = Field(default=None, alias="naicsCode")
    classification_code: Optional[str] = Field(default=None, alias="classificationCode")
    set_aside_code: Optional[str] = Field(default=None, alias="typeOfSetAside")
    set_aside_description: Optional[str] = Field(
        default=None, alias="typeOfSetAsideDescription"
    )

    # Dates (kept as raw strings + parsed datetimes).
    posted_date_raw: Optional[str] = Field(default=None, alias="postedDate")
    response_deadline_raw: Optional[str] = Field(default=None, alias="responseDeadLine")
    archive_date_raw: Optional[str] = Field(default=None, alias="archiveDate")
    archive_type: Optional[str] = Field(default=None, alias="archiveType")
    active: Optional[str] = None

    # Contacts / locations / links.
    points_of_contact: List[PointOfContact] = Field(
        default_factory=list, alias="pointOfContact"
    )
    office_address: Optional[Address] = Field(default=None, alias="officeAddress")
    place_of_performance: Optional[Address] = Field(
        default=None, alias="placeOfPerformance"
    )
    resource_links: List[ResourceLink] = Field(
        default_factory=list, alias="resourceLinks"
    )
    ui_link: Optional[str] = Field(default=None, alias="uiLink")

    # The ``description`` field is usually a URL pointing to the full text.
    description_link: Optional[str] = Field(default=None, alias="description")
    additional_info_link: Optional[str] = Field(default=None, alias="additionalInfoLink")

    award: Optional[AwardSummary] = None

    # ------------------------------------------------------------------ #
    # Validators / normalizers
    # ------------------------------------------------------------------ #
    @field_validator("resource_links", mode="before")
    @classmethod
    def _normalize_resource_links(cls, v: Any) -> Any:
        """Accept a list of bare URL strings or dicts and normalize to models."""
        if not v:
            return []
        normalized: List[Any] = []
        for item in v:
            if isinstance(item, str):
                normalized.append(ResourceLink.from_url(item))
            else:
                normalized.append(item)
        return normalized

    @field_validator("place_of_performance", "office_address", mode="before")
    @classmethod
    def _empty_dict_to_none(cls, v: Any) -> Any:
        if isinstance(v, dict) and not v:
            return None
        return v

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def posted_date(self) -> Optional[datetime]:
        return _parse_dt(self.posted_date_raw)

    @property
    def response_deadline(self) -> Optional[datetime]:
        return _parse_dt(self.response_deadline_raw)

    @property
    def archive_date(self) -> Optional[datetime]:
        return _parse_dt(self.archive_date_raw)

    @property
    def agency(self) -> Optional[str]:
        """Top-level department/agency from the parent path, if available."""
        if not self.full_parent_path_name:
            return None
        return self.full_parent_path_name.split(".")[0].strip()

    @property
    def office(self) -> Optional[str]:
        """Most specific office from the parent path, if available."""
        if not self.full_parent_path_name:
            return None
        return self.full_parent_path_name.split(".")[-1].strip()

    @property
    def primary_contact(self) -> Optional[PointOfContact]:
        for poc in self.points_of_contact:
            if (poc.poc_type or "").lower() == "primary":
                return poc
        return self.points_of_contact[0] if self.points_of_contact else None

    @property
    def sam_url(self) -> str:
        """A direct link to the opportunity on sam.gov."""
        if self.ui_link:
            return self.ui_link
        return f"https://sam.gov/opp/{self.notice_id}/view"

    def days_until_deadline(self, now: Optional[datetime] = None) -> Optional[int]:
        """Whole days until the response deadline (negative if past)."""
        deadline = self.response_deadline
        if deadline is None:
            return None
        now = now or datetime.now(tz=deadline.tzinfo)
        try:
            return (deadline - now).days
        except TypeError:
            # tz-aware vs naive mismatch; compare naively.
            return (deadline.replace(tzinfo=None) - datetime.now()).days

    def to_table_row(self) -> Dict[str, Any]:
        """Flatten to a row for the search results table."""
        deadline = self.response_deadline
        return {
            "Title": self.title or "(untitled)",
            "Agency": self.agency or "",
            "Office": self.office or "",
            "Type": self.notice_type or "",
            "Set-Aside": self.set_aside_description or self.set_aside_code or "",
            "NAICS": self.naics_code or "",
            "Response Deadline": deadline.strftime("%Y-%m-%d %H:%M") if deadline else "",
            "Days Left": self.days_until_deadline(),
            "Notice ID": self.notice_id,
            "Link": self.sam_url,
        }


class OpportunitySearchResult(BaseModel):
    """Parsed result of an Opportunities v2 ``/search`` call."""

    total_records: int = Field(default=0, alias="totalRecords")
    limit: int = 0
    offset: int = 0
    opportunities: List[Opportunity] = Field(
        default_factory=list, alias="opportunitiesData"
    )
    from_cache: bool = False
    fetched_at: Optional[datetime] = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @property
    def count(self) -> int:
        return len(self.opportunities)

    @property
    def has_more(self) -> bool:
        return (self.offset + self.count) < self.total_records

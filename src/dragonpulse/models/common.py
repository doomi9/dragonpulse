"""Shared models and enumerations used across DragonPulse.

These mirror the shapes returned by the SAM.gov Opportunities v2 API but are
tolerant of missing/extra fields (the live API is inconsistent across notice
types). Parsing therefore favors ``Optional`` fields and ``extra="ignore"``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NoticeType(str, Enum):
    """SAM.gov procurement notice types and their single-letter ``ptype`` codes.

    The ``ptype`` query parameter on the Opportunities v2 API uses these codes.
    """

    SOLICITATION = "o"
    COMBINED_SYNOPSIS_SOLICITATION = "k"
    PRESOLICITATION = "p"
    SOURCES_SOUGHT = "r"
    SPECIAL_NOTICE = "s"
    AWARD_NOTICE = "a"
    JUSTIFICATION = "u"
    INTENT_TO_BUNDLE = "i"
    SALE_OF_SURPLUS = "g"

    @property
    def label(self) -> str:
        return _NOTICE_LABELS[self]


_NOTICE_LABELS = {
    NoticeType.SOLICITATION: "Solicitation",
    NoticeType.COMBINED_SYNOPSIS_SOLICITATION: "Combined Synopsis/Solicitation",
    NoticeType.PRESOLICITATION: "Presolicitation",
    NoticeType.SOURCES_SOUGHT: "Sources Sought",
    NoticeType.SPECIAL_NOTICE: "Special Notice",
    NoticeType.AWARD_NOTICE: "Award Notice",
    NoticeType.JUSTIFICATION: "Justification (J&A)",
    NoticeType.INTENT_TO_BUNDLE: "Intent to Bundle Requirements",
    NoticeType.SALE_OF_SURPLUS: "Sale of Surplus Property",
}


# Common set-aside codes accepted by the ``typeOfSetAside`` query parameter.
# (value -> human label). Not exhaustive, but covers the usual small-business cases.
SET_ASIDE_CHOICES = {
    "SBA": "Total Small Business Set-Aside",
    "SBP": "Partial Small Business Set-Aside",
    "8A": "8(a) Set-Aside",
    "8AN": "8(a) Sole Source",
    "HZC": "HUBZone Set-Aside",
    "HZS": "HUBZone Sole Source",
    "SDVOSBC": "Service-Disabled Veteran-Owned Set-Aside",
    "SDVOSBS": "SDVOSB Sole Source",
    "WOSB": "Women-Owned Small Business Set-Aside",
    "WOSBSS": "WOSB Sole Source",
    "EDWOSB": "Economically Disadvantaged WOSB Set-Aside",
    "EDWOSBSS": "EDWOSB Sole Source",
    "LAS": "Local Area Set-Aside",
    "IEE": "Indian Economic Enterprise",
    "ISBEE": "Indian Small Business Economic Enterprise",
}


class SetAside(str, Enum):
    """Convenience enum of the most common set-aside codes."""

    TOTAL_SMALL_BUSINESS = "SBA"
    EIGHT_A = "8A"
    HUBZONE = "HZC"
    SDVOSB = "SDVOSBC"
    WOSB = "WOSB"
    EDWOSB = "EDWOSB"


class _Tolerant(BaseModel):
    """Base model that ignores unexpected fields from the live API."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Address(_Tolerant):
    """A postal address (office or place of performance)."""

    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = Field(default=None, alias="zip")
    country_code: Optional[str] = Field(default=None, alias="countryCode")
    street_address: Optional[str] = Field(default=None, alias="streetAddress")

    def one_line(self) -> str:
        parts = [self.street_address, self.city, self.state, self.zipcode, self.country_code]
        return ", ".join(p for p in parts if p)


class PointOfContact(_Tolerant):
    """A single point-of-contact entry from an opportunity.

    SAM.gov uses ``fullName`` (not first/last) and a ``type`` such as
    ``"primary"`` or ``"secondary"``.
    """

    full_name: Optional[str] = Field(default=None, alias="fullName")
    title: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None
    poc_type: Optional[str] = Field(default=None, alias="type")

    @property
    def display_name(self) -> str:
        return self.full_name or self.email or "(unnamed contact)"

    def has_contact_method(self) -> bool:
        return bool(self.email or self.phone)


class ResourceLink(_Tolerant):
    """A downloadable attachment / resource associated with an opportunity.

    The live API often returns ``resourceLinks`` as a list of bare URL strings;
    :meth:`from_url` normalizes those into this model.
    """

    url: str
    name: Optional[str] = None
    file_type: Optional[str] = None

    @classmethod
    def from_url(cls, url: str) -> "ResourceLink":
        """Build a ResourceLink from a bare URL, inferring a display name."""
        # SAM resource URLs look like .../files/<uuid>/download — not human friendly,
        # so we keep the URL but leave name to be enriched later (Content-Disposition).
        return cls(url=url, name=None, file_type=None)

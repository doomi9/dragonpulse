"""Search filter model.

``OpportunityFilters`` is the single source of truth for what the user has
selected in the sidebar. It knows how to translate itself into SAM.gov
Opportunities v2 query parameters via :meth:`to_query_params`.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class OpportunityFilters(BaseModel):
    """User-selected search filters for the Opportunities v2 endpoint."""

    keyword: Optional[str] = Field(default=None, description="Free-text title search.")
    naics_codes: List[str] = Field(default_factory=list)
    set_aside_codes: List[str] = Field(default_factory=list)
    notice_type_codes: List[str] = Field(
        default_factory=list, description="Single-letter ptype codes."
    )
    department_name: Optional[str] = Field(
        default=None, description="Top-level department/agency name (deptname)."
    )
    posted_from: date = Field(
        default_factory=lambda: date.today() - timedelta(days=30)
    )
    posted_to: date = Field(default_factory=date.today)
    limit: int = Field(default=25, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)

    @field_validator("naics_codes", "set_aside_codes", "notice_type_codes", mode="before")
    @classmethod
    def _drop_blanks(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return v

    @field_validator("posted_to")
    @classmethod
    def _validate_range(cls, v: date, info: Any) -> date:
        posted_from = info.data.get("posted_from")
        if posted_from and v < posted_from:
            raise ValueError("posted_to must be on or after posted_from")
        # SAM.gov enforces a maximum 1-year window.
        if posted_from and (v - posted_from).days > 365:
            raise ValueError("Date range cannot exceed 1 year (SAM.gov limit)")
        return v

    def to_query_params(self) -> Dict[str, Any]:
        """Translate filters into Opportunities v2 query parameters.

        Note: SAM.gov uses ``MM/dd/yyyy`` for ``postedFrom``/``postedTo`` and
        requires both. Multi-value filters (NAICS, set-aside, notice type) are
        passed as the first selected value here; the client expands them into
        repeated parameters where the API supports it.
        """
        params: Dict[str, Any] = {
            "postedFrom": self.posted_from.strftime("%m/%d/%Y"),
            "postedTo": self.posted_to.strftime("%m/%d/%Y"),
            "limit": self.limit,
            "offset": self.offset,
        }
        if self.keyword:
            params["title"] = self.keyword
        if self.naics_codes:
            params["ncode"] = ",".join(self.naics_codes)
        if self.set_aside_codes:
            params["typeOfSetAside"] = ",".join(self.set_aside_codes)
        if self.notice_type_codes:
            params["ptype"] = ",".join(self.notice_type_codes)
        if self.department_name:
            params["deptname"] = self.department_name
        return params

    def cache_signature(self) -> Dict[str, Any]:
        """A JSON-serializable view used for cache keying and display."""
        return self.model_dump(mode="json")

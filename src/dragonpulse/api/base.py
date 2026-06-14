"""Base SAM.gov HTTP client.

Responsibilities
----------------
- Inject the API key (never logged in full).
- Consult the disk cache before any network call.
- Enforce the daily live-request budget (protects the 10/day basic key).
- Translate HTTP/transport failures into typed exceptions.
- Be rate-limit aware: detect 429s and SAM's "rate limit exceeded" bodies.

This base class is endpoint-agnostic; concrete clients (Opportunities, Awards)
build on top of :meth:`get_json`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import requests

from dragonpulse.cache.disk_cache import DiskCache
from dragonpulse.cache.request_budget import RequestBudget, RequestBudgetExceeded
from dragonpulse.config.logging_config import get_logger, redact
from dragonpulse.config.settings import Settings, get_settings

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30  # seconds


class SamApiError(RuntimeError):
    """Base class for all SAM.gov API errors."""


class SamAuthError(SamApiError):
    """Raised on 401/403 — usually a missing or invalid API key."""


class SamRateLimitError(SamApiError):
    """Raised when SAM.gov reports the request rate/quota was exceeded."""


class SamClient:
    """Thin, cache-first HTTP client for the SAM.gov APIs."""

    BASE_URL = "https://api.sam.gov"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[DiskCache] = None,
        budget: Optional[RequestBudget] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or DiskCache(
            cache_dir=self.settings.cache_dir,
            default_ttl_seconds=self.settings.cache_ttl_seconds,
            disabled=self.settings.cache_disabled,
        )
        self.budget = budget or RequestBudget(
            state_dir=self.settings.cache_dir,
            daily_budget=self.settings.daily_request_budget,
        )
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "DragonPulse/0.1"})

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def get_json(
        self,
        path: str,
        params: Dict[str, Any],
        *,
        ttl_seconds: Optional[int] = None,
        force_refresh: bool = False,
        allow_network: bool = True,
    ) -> Tuple[Dict[str, Any], bool]:
        """GET ``path`` with ``params``, using the cache when possible.

        Parameters
        ----------
        path:
            API path beginning with ``/`` (e.g. ``/opportunities/v2/search``).
        params:
            Query parameters (without the API key — it is injected here).
        ttl_seconds:
            Override the cache TTL for this call.
        force_refresh:
            Skip the cache read (still writes the fresh result back).
        allow_network:
            If False, only cached data is acceptable; raises if not cached.

        Returns
        -------
        (payload, from_cache)
            The decoded JSON body and whether it came from the disk cache.

        Raises
        ------
        SamAuthError, SamRateLimitError, SamApiError, RequestBudgetExceeded
        """
        endpoint = f"{self.BASE_URL}{path}"

        # 1) Cache read.
        if not force_refresh:
            entry = self.cache.get(endpoint, params, ttl_seconds=ttl_seconds)
            if entry is not None:
                logger.info("Serving cached response for %s (age=%.0fs)", path, entry.age_seconds)
                return entry.payload, True

        if not allow_network:
            raise SamApiError(
                f"No fresh cached data for {path} and network access is disabled."
            )

        # 2) Pre-flight checks.
        if not self.settings.has_api_key:
            raise SamAuthError(
                "No SAM.gov API key configured. Set DRAGONPULSE_SAM_API_KEY_BASIC in .env."
            )
        self.budget.check()

        # 3) Network call.
        payload = self._request(endpoint, params)

        # 4) Record the spend and cache the result.
        self.budget.record(1)
        self.cache.set(endpoint, params, payload, ttl_seconds=ttl_seconds)
        return payload, False

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual HTTP GET and normalize errors."""
        request_params = dict(params)
        request_params["api_key"] = self.settings.active_api_key

        logger.info(
            "LIVE GET %s params=%s key=%s (budget %d/%d used today)",
            url,
            {k: v for k, v in params.items() if k != "api_key"},
            self.settings.masked_api_key(),
            self.budget.used_today(),
            self.budget.daily_budget,
        )

        try:
            resp = self.session.get(url, params=request_params, timeout=DEFAULT_TIMEOUT)
        except requests.Timeout as exc:
            raise SamApiError(f"Request to {url} timed out after {DEFAULT_TIMEOUT}s") from exc
        except requests.RequestException as exc:
            raise SamApiError(f"Network error calling {url}: {exc}") from exc

        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: requests.Response) -> Dict[str, Any]:
        status = resp.status_code

        if status == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise SamApiError("SAM.gov returned non-JSON body on 200") from exc

        # Try to extract a useful message from the error body.
        detail = SamClient._extract_error_detail(resp)

        if status in (401, 403):
            raise SamAuthError(f"Authentication failed ({status}): {detail}")
        if status == 429:
            raise SamRateLimitError(f"Rate limit exceeded ({status}): {detail}")
        # SAM sometimes returns 400/403 with a rate-limit message in the body.
        if "rate limit" in detail.lower() or "over rate limit" in detail.lower():
            raise SamRateLimitError(f"Rate limit exceeded: {detail}")
        raise SamApiError(f"SAM.gov error ({status}): {detail}")

    @staticmethod
    def _extract_error_detail(resp: requests.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return (resp.text or "").strip()[:300] or resp.reason
        if isinstance(body, dict):
            for key in ("error", "message", "errormessage", "errorMessage", "description"):
                val = body.get(key)
                if isinstance(val, dict):
                    val = val.get("message") or val.get("code")
                if val:
                    return str(val)
        return str(body)[:300]

    def cache_only_get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return cached payload for a request without hitting the network."""
        endpoint = f"{self.BASE_URL}{path}"
        entry = self.cache.get(endpoint, params, ttl_seconds=-1)  # ignore TTL
        return entry.payload if entry else None


def safe_sleep(seconds: float) -> None:
    """Sleep helper (extracted for easy patching in tests)."""
    time.sleep(seconds)


# Re-export for convenience.
__all__ = [
    "SamClient",
    "SamApiError",
    "SamAuthError",
    "SamRateLimitError",
    "RequestBudgetExceeded",
    "redact",
]

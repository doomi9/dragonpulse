"""Daily live-request budget tracker.

The basic SAM.gov key allows only 10 requests/day. To avoid accidentally
burning through that quota during development, DragonPulse tracks how many
*live* (non-cached) requests have been made today and refuses to exceed a
configurable soft budget.

State is persisted to a small JSON file keyed by local date so the count
survives app restarts within the same day.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Dict

from dragonpulse.config.logging_config import get_logger

logger = get_logger(__name__)


class RequestBudgetExceeded(RuntimeError):
    """Raised when a live request would exceed the configured daily budget."""


class RequestBudget:
    """Persisted per-day counter of live API requests."""

    def __init__(self, state_dir: Path, daily_budget: int) -> None:
        self.state_path = Path(state_dir) / "request_budget.json"
        self.daily_budget = daily_budget
        self._lock = Lock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.localtime())

    def _load(self) -> Dict[str, int]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return {}

    def _save(self, data: Dict[str, int]) -> None:
        # Keep only the last few days to avoid unbounded growth.
        trimmed = dict(sorted(data.items())[-7:])
        self.state_path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")

    def used_today(self) -> int:
        return self._load().get(self._today(), 0)

    def remaining(self) -> int:
        return max(self.daily_budget - self.used_today(), 0)

    def check(self) -> None:
        """Raise :class:`RequestBudgetExceeded` if no budget remains."""
        if self.remaining() <= 0:
            raise RequestBudgetExceeded(
                f"Daily live-request budget ({self.daily_budget}) reached. "
                "Results will come from cache only until tomorrow, or raise "
                "DRAGONPULSE_DAILY_REQUEST_BUDGET if you have a higher-tier key."
            )

    def record(self, n: int = 1) -> int:
        """Increment today's counter by ``n`` and return the new total."""
        with self._lock:
            data = self._load()
            today = self._today()
            data[today] = data.get(today, 0) + n
            self._save(data)
            logger.debug("Live request recorded: %d/%d today", data[today], self.daily_budget)
            return data[today]

    def reset_today(self) -> None:
        with self._lock:
            data = self._load()
            data.pop(self._today(), None)
            self._save(data)

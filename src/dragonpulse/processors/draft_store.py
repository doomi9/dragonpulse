"""Local, on-disk store for saved proposal drafts + version history.

Drafts are persisted as individual JSON files under ``data/drafts/`` so they
survive restarts and stay fully local (nothing leaves the machine). Each file is
a :class:`~dragonpulse.models.proposal.SavedDraft` — a named snapshot of a
:class:`~dragonpulse.models.proposal.ProposalDraft` with created/modified
timestamps and a version counter.

The store is intentionally simple (one file per saved draft, listed/filtered by
``notice_id``); a contractor's draft library is small enough that this is fast
and trivially auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings
from dragonpulse.models.proposal import ProposalDraft, SavedDraft

logger = get_logger(__name__)


class DraftStore:
    """Persist and retrieve named proposal drafts on local disk."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.dir = self.settings.drafts_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, draft_id: str) -> Path:
        return self.dir / f"{draft_id}.json"

    def save(self, name: str, draft: ProposalDraft) -> SavedDraft:
        """Persist ``draft`` under ``name`` as a new saved draft."""
        saved = SavedDraft(
            name=name.strip() or "Untitled draft",
            notice_id=draft.notice_id,
            opportunity_title=draft.opportunity_title,
            draft=draft,
        )
        self._write(saved)
        logger.info("Saved draft '%s' (%s) for notice %s", saved.name, saved.draft_id,
                    saved.notice_id)
        return saved

    def update(self, draft_id: str, draft: ProposalDraft) -> Optional[SavedDraft]:
        """Overwrite an existing saved draft, bumping version + modified time."""
        saved = self.get(draft_id)
        if saved is None:
            return None
        saved.draft = draft
        saved.opportunity_title = draft.opportunity_title
        saved.touch()
        self._write(saved)
        logger.info("Updated draft '%s' to v%d", saved.draft_id, saved.version)
        return saved

    def _write(self, saved: SavedDraft) -> None:
        self._path(saved.draft_id).write_text(
            json.dumps(saved.model_dump(), indent=2), "utf-8"
        )

    def get(self, draft_id: str) -> Optional[SavedDraft]:
        path = self._path(draft_id)
        if not path.exists():
            return None
        try:
            return SavedDraft.model_validate(json.loads(path.read_text("utf-8")))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Saved draft %s unreadable: %s", draft_id, exc)
            return None

    def list_drafts(self, notice_id: Optional[str] = None) -> List[SavedDraft]:
        """Return saved drafts (optionally filtered to one opportunity).

        Sorted newest-written first. File modification time (high resolution) is
        used so rapid successive saves within the same second still order
        correctly, falling back to the ``modified_at`` field.
        """
        rows: List[tuple] = []
        for path in self.dir.glob("*.json"):
            try:
                saved = SavedDraft.model_validate(json.loads(path.read_text("utf-8")))
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            if notice_id is None or saved.notice_id == notice_id:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                rows.append((mtime, saved))
        rows.sort(key=lambda r: (r[0], r[1].modified_at), reverse=True)
        return [saved for _, saved in rows]

    def delete(self, draft_id: str) -> bool:
        path = self._path(draft_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted saved draft %s", draft_id)
            return True
        return False

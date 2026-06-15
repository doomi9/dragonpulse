"""Tests for the local saved-draft store + version history."""

from __future__ import annotations

from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.models.proposal import ProposalDraft, ProposalSection
from dragonpulse.processors.draft_store import DraftStore


def _settings(tmp_path) -> Settings:
    return Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        llm_enabled=False,
    )


def _draft(notice_id="n1", title="Opp One") -> ProposalDraft:
    return ProposalDraft(
        notice_id=notice_id,
        opportunity_title=title,
        sections=[ProposalSection(section_id="exec", title="Executive Summary", content="Hi")],
    )


def test_save_and_list_by_notice(tmp_path):
    store = DraftStore(_settings(tmp_path))
    store.save("First draft", _draft("n1", "Opp One"))
    store.save("Other opp", _draft("n2", "Opp Two"))

    n1 = store.list_drafts("n1")
    assert len(n1) == 1
    assert n1[0].name == "First draft"
    assert n1[0].opportunity_title == "Opp One"
    assert len(store.list_drafts()) == 2  # both, unfiltered


def test_load_roundtrip(tmp_path):
    store = DraftStore(_settings(tmp_path))
    saved = store.save("Draft A", _draft("n1"))
    loaded = store.get(saved.draft_id)
    assert loaded is not None
    assert loaded.draft.notice_id == "n1"
    assert loaded.draft.sections[0].content == "Hi"


def test_update_bumps_version_and_modified(tmp_path):
    store = DraftStore(_settings(tmp_path))
    saved = store.save("Draft A", _draft("n1"))
    assert saved.version == 1

    new_draft = _draft("n1")
    new_draft.sections[0].content = "Updated content"
    updated = store.update(saved.draft_id, new_draft)
    assert updated is not None
    assert updated.version == 2
    assert updated.draft.sections[0].content == "Updated content"
    # Persisted.
    reloaded = store.get(saved.draft_id)
    assert reloaded.version == 2


def test_delete(tmp_path):
    store = DraftStore(_settings(tmp_path))
    saved = store.save("Draft A", _draft("n1"))
    assert store.delete(saved.draft_id) is True
    assert store.get(saved.draft_id) is None
    assert store.delete("nonexistent") is False


def test_list_sorted_newest_first(tmp_path):
    store = DraftStore(_settings(tmp_path))
    a = store.save("A", _draft("n1"))
    b = store.save("B", _draft("n1"))
    # Bump A so it becomes most-recently-modified.
    store.update(a.draft_id, _draft("n1"))
    ordered = store.list_drafts("n1")
    assert [d.draft_id for d in ordered][0] == a.draft_id
    assert b.draft_id in [d.draft_id for d in ordered]

"""Fail-closed export paths at the Phase 2/Phase 3 analysis boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neurallm.providers.fake import FakeProvider
from neurallm.reporting import export_closed_run, scientific_result_sha256
from neurallm.reporting.artifacts import _result_row
from neurallm.storage import RunFinalization, SQLiteRunStore, StoredTurn, TurnState
from tests.storage.helpers import complete_request, make_manifest, make_request


def _closed_run(
    path: Path,
    *,
    phase3: bool = False,
) -> tuple[Path, RunFinalization, StoredTurn]:
    path.mkdir()
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity).model_copy(
        update={"decision_rule_version": "phase2-no-scientific-decision-v1"}
    )
    if phase3:
        manifest = manifest.model_copy(
            update={
                "decision_rule_version": "phase3-baseline-evaluator-v1",
                "phase3_analysis_contract_sha256": "a" * 64,
            }
        )
    with SQLiteRunStore(path / "run.sqlite3", manifest) as store:
        request = make_request(provider.provider_identity)
        complete_request(store, provider, request)
        finalization = store.finalize_run(
            (request.condition_id,), scientific_result_sha256(store.list_turns())
        )
        turn = store.list_turns()[0]
    return path, finalization, turn


def test_phase3_export_requires_finalized_analysis(tmp_path: Path) -> None:
    run_directory, _, _ = _closed_run(tmp_path / "phase3", phase3=True)

    with pytest.raises(ValueError, match="missing finalized analysis evidence"):
        export_closed_run(run_directory)


def test_pre_phase3_export_rejects_unexpected_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory, _, _ = _closed_run(tmp_path / "phase2")
    monkeypatch.setattr(SQLiteRunStore, "get_analysis", lambda _store: object())

    with pytest.raises(ValueError, match="Phase 2 run unexpectedly"):
        export_closed_run(run_directory)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty", "at least one turn"),
        ("incomplete", "non-committed turns"),
    ],
)
def test_export_rejects_impossible_finalized_turn_surfaces(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory, finalization, turn = _closed_run(tmp_path / case)
    turns = () if case == "empty" else (replace(turn, state=TurnState.PREPARED),)
    monkeypatch.setattr(SQLiteRunStore, "verify_integrity", lambda _store: None)
    monkeypatch.setattr(SQLiteRunStore, "get_finalization", lambda _store: finalization)
    monkeypatch.setattr(SQLiteRunStore, "list_turns", lambda _store: turns)

    with pytest.raises(ValueError, match=message):
        export_closed_run(run_directory)


def test_result_views_reject_incomplete_committed_evidence(tmp_path: Path) -> None:
    _, _, turn = _closed_run(tmp_path / "phase2")
    incomplete = replace(turn, response=None)

    with pytest.raises(ValueError, match="missing response"):
        _result_row(incomplete)
    with pytest.raises(ValueError, match="incomplete turn evidence"):
        scientific_result_sha256((incomplete,))

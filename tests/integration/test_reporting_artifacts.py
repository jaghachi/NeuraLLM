"""Integration checks for the compact closed-run artifact boundary."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import (
    CLOSED_RUN_ARTIFACTS,
    export_closed_run,
    scientific_result_sha256,
)
from neurallm.storage import SQLiteRunStore
from tests.storage.helpers import complete_request, make_manifest, make_request


def _closed_run(path: Path) -> Path:
    path.mkdir()
    provider = FakeProvider()
    with SQLiteRunStore(path / "run.sqlite3", make_manifest(provider.provider_identity)) as store:
        request = make_request(provider.provider_identity)
        complete_request(store, provider, request)
        store.finalize_run(
            (request.condition_id,),
            scientific_result_sha256(store.list_turns()),
        )
    return path


def test_closed_run_exports_exact_compact_deterministic_artifact_set(tmp_path: Path) -> None:
    run_directory = _closed_run(tmp_path / "run")

    summary = export_closed_run(run_directory)
    derived_names = CLOSED_RUN_ARTIFACTS - {"run.sqlite3"}
    first_bytes = {name: (run_directory / name).read_bytes() for name in sorted(derived_names)}
    repeated = export_closed_run(run_directory)
    second_bytes = {name: (run_directory / name).read_bytes() for name in sorted(derived_names)}

    assert {item.name for item in run_directory.iterdir()} == CLOSED_RUN_ARTIFACTS
    assert summary == repeated
    assert summary.committed_turns == 1
    assert first_bytes == second_bytes

    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((run_directory / "decision.json").read_text(encoding="utf-8"))
    assert manifest["provider_identity"]["provider_type"] == "fake"
    assert decision["implementation_phase"] == 2
    assert decision["scientific_decision"] is None
    assert decision["claim_scope"] == "engineering_validation_only"
    assert decision["comparison_status"] == "not_available_until_phase_3"
    assert decision["scientific_result_sha256"] == summary.scientific_result_sha256
    assert len(summary.scientific_result_sha256) == 64

    with (run_directory / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["policy_id"] == "test-policy"
    assert rows[0]["response_text"].startswith("fake-response:")

    with (run_directory / "comparisons.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert "does not establish policy efficacy" in report
    assert "Phase 3" in report


def test_export_input_and_missing_identity_guards(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        export_closed_run(str(tmp_path))  # type: ignore[arg-type]

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("not a run", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        export_closed_run(file_path)

    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()
    with pytest.raises(FileNotFoundError, match="run.sqlite3"):
        export_closed_run(empty_directory)

    unbound_directory = tmp_path / "unbound"
    unbound_directory.mkdir()
    with SQLiteRunStore(unbound_directory / "run.sqlite3"):
        pass
    with pytest.raises(ValueError, match="does not contain a manifest"):
        export_closed_run(unbound_directory)

    with pytest.raises(ValueError, match="at least one committed turn"):
        scientific_result_sha256(())


def test_export_fails_closed_for_noncommitted_run(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(
        run_directory / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        store.prepare_turn(request)

    with pytest.raises(ValueError, match="not finalized"):
        export_closed_run(run_directory)


def test_export_rejects_unexpected_run_directory_artifacts(tmp_path: Path) -> None:
    run_directory = _closed_run(tmp_path / "run")
    (run_directory / "request-0001.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected artifacts"):
        export_closed_run(run_directory)


def test_export_rejects_finalized_scientific_result_hash_drift(tmp_path: Path) -> None:
    run_directory = _closed_run(tmp_path / "run")
    database_path = run_directory / "run.sqlite3"
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT finalization_json FROM run_finalization WHERE singleton_id = 1"
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        payload["scientific_result_sha256"] = "f" * 64
        connection.execute(
            """
            UPDATE run_finalization
            SET finalization_json = ?, finalization_sha256 = ?
            WHERE singleton_id = 1
            """,
            (canonical_json(payload), canonical_sha256(payload)),
        )

    with pytest.raises(ValueError, match="does not match the recomputed output"):
        export_closed_run(run_directory)

"""Schema ownership, migration, manifest, and compact-store tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.fake import FakeProvider
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    ManifestMismatchError,
    SchemaVersionError,
    SQLiteRunStore,
    StoreInvariantError,
)
from neurallm.storage.migrations import APPLICATION_ID, MIGRATIONS
from tests.storage.helpers import make_manifest, make_request


def test_new_store_has_explicit_versioned_schema_and_bound_manifest(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)

    with SQLiteRunStore(database, manifest) as store:
        assert store.schema_version == CURRENT_SCHEMA_VERSION
        assert store.get_manifest() == manifest
        store.verify_integrity()

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA application_id").fetchone() == (APPLICATION_ID,)
        assert connection.execute("PRAGMA user_version").fetchone() == (CURRENT_SCHEMA_VERSION,)
        assert connection.execute("SELECT version, name FROM schema_migrations").fetchall() == [
            (1, "canonical_transactional_turn_store"),
            (2, "phase3_analysis_evidence"),
        ]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
    assert {
        "run_manifest",
        "run_finalization",
        "turns",
        "responses",
        "turn_metrics",
        "history_commitments",
        "turn_inputs",
        "analysis_manifest",
        "comparison_results",
        "guardrail_results",
        "analysis_decision",
        "analysis_finalization",
    } <= tables
    assert {
        "turns_forward_state_guard",
        "turns_insert_after_finalization_guard",
        "turn_inputs_insert_after_run_finalization_guard",
        "comparisons_insert_after_analysis_finalization_guard",
        "guardrails_insert_after_analysis_finalization_guard",
    } <= triggers


def test_manifest_cannot_be_rebound(tmp_path: Path) -> None:
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    changed = manifest.model_copy(
        update={"experiment_config_hash": "f" * 64},
    )

    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        with pytest.raises(ManifestMismatchError, match="another manifest"):
            store.bind_manifest(changed)
        assert store.get_manifest() == manifest


def test_v1_store_migrates_additively_and_retains_its_frozen_manifest(tmp_path: Path) -> None:
    database = tmp_path / "phase2.sqlite3"
    provider = FakeProvider()
    phase2_manifest = make_manifest(provider.provider_identity).model_copy(
        update={"database_schema_version": 1}
    )
    with sqlite3.connect(database, isolation_level=None) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for statement in MIGRATIONS[0].statements:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
            (MIGRATIONS[0].version, MIGRATIONS[0].name),
        )
        connection.execute(
            """
            INSERT INTO run_manifest(singleton_id, manifest_json, manifest_sha256)
            VALUES (1, ?, ?)
            """,
            (canonical_json(phase2_manifest), canonical_sha256(phase2_manifest)),
        )
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute("PRAGMA user_version = 1")
        connection.execute("COMMIT")

    with SQLiteRunStore(database) as store:
        assert store.schema_version == CURRENT_SCHEMA_VERSION
        assert store.get_manifest() == phase2_manifest
        store.verify_integrity()


def test_unmanaged_sqlite_database_is_not_adopted(tmp_path: Path) -> None:
    database = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")

    with pytest.raises(SchemaVersionError, match="unmanaged"):
        SQLiteRunStore(database)


def test_newer_schema_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")

    with pytest.raises(SchemaVersionError, match="newer"):
        SQLiteRunStore(database)


def test_missing_migrated_schema_object_fails_closed(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER turns_forward_state_guard")

    with pytest.raises(SchemaVersionError, match="schema objects"):
        SQLiteRunStore(database)


def test_turns_require_a_bound_manifest(tmp_path: Path) -> None:
    provider = FakeProvider()

    with SQLiteRunStore(tmp_path / "run.sqlite3") as store:
        with pytest.raises(StoreInvariantError, match="manifest must be bound"):
            store.prepare_turn(make_request(provider.provider_identity))


def test_compact_keeps_one_canonical_artifact(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    with SQLiteRunStore(database, make_manifest(provider.provider_identity)) as store:
        store.compact()

    assert tuple(path.name for path in tmp_path.iterdir()) == ("run.sqlite3",)

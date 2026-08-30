"""Adversarial on-disk corruption tests for fail-closed reads and resumes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neurallm.control.policy import PolicyState
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.fake import FakeProvider
from neurallm.storage import (
    SchemaVersionError,
    SQLiteRunStore,
    StoreCorruptionError,
)
from tests.storage.helpers import complete_request, make_manifest, make_request


class StateWithAnotherShape(PolicyState):
    """Incompatible state shape used to exercise typed rehydration checks."""

    required_name: str


@pytest.mark.parametrize(
    ("request_json", "request_sha256", "message"),
    [
        ("{", "f" * 64, "invalid JSON"),
        ('{ "value":1}', canonical_sha256({"value": 1}), "not canonical JSON"),
        ("{}", canonical_sha256({}), "typed validation"),
    ],
)
def test_malformed_noncanonical_and_untyped_request_json_fail_closed(
    tmp_path: Path,
    request_json: str,
    request_sha256: str,
    message: str,
) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, manifest) as store:
        store.prepare_turn(request)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE turns SET request_json = ?, request_sha256 = ? WHERE condition_id = ?",
            (request_json, request_sha256, request.condition_id),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match=message):
            store.get_turn(request.condition_id)


def test_condition_columns_cannot_disagree_with_canonical_condition(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, make_manifest(provider.provider_identity)) as store:
        store.prepare_turn(request)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE turns SET policy_id = 'tampered' WHERE condition_id = ?",
            (request.condition_id,),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="policy_id disagrees"):
            store.get_turn(request.condition_id)


def test_checkpoint_state_cannot_hide_premature_response_evidence(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    response = provider.generate(request)
    with SQLiteRunStore(database, make_manifest(provider.provider_identity)) as store:
        store.prepare_turn(request)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO responses(condition_id, response_json, response_sha256)
            VALUES (?, ?, ?)
            """,
            (request.condition_id, canonical_json(response), canonical_sha256(response)),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="checkpoint state"):
            store.resume_action(request.condition_id)


def test_policy_trace_and_history_commitment_tampering_are_detected(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, make_manifest(provider.provider_identity)) as store:
        complete_request(store, provider, request)
    wrong_trace = {
        "action": {
            "presence_penalty_delta": 0.0,
            "temperature_delta": 0.0,
            "top_k_delta": 0,
            "top_p_delta": 0.0,
        },
        "policy_id": "wrong-policy",
        "turn_index": 0,
    }
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE history_commitments
            SET policy_trace_json = ?, policy_trace_sha256 = ?
            WHERE condition_id = ?
            """,
            (canonical_json(wrong_trace), canonical_sha256(wrong_trace), request.condition_id),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="wrong policy_id"):
            store.get_turn(request.condition_id)

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE history_commitments
            SET policy_trace_json = ?, policy_trace_sha256 = ?,
                history_commitment_sha256 = ?
            WHERE condition_id = ?
            """,
            (
                canonical_json(
                    {
                        **wrong_trace,
                        "policy_id": request.condition.policy_id,
                    }
                ),
                canonical_sha256(
                    {
                        **wrong_trace,
                        "policy_id": request.condition.policy_id,
                    }
                ),
                "f" * 64,
                request.condition_id,
            ),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="commitment hash"):
            store.get_turn(request.condition_id)


def test_policy_state_must_rehydrate_as_callers_declared_type(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        complete_request(store, provider, request)
        with pytest.raises(StoreCorruptionError, match="declared model"):
            store.load_policy_state(request.condition_id, StateWithAnotherShape)


def test_stored_manifest_schema_version_is_revalidated(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    with SQLiteRunStore(database, manifest):
        pass
    wrong = manifest.model_copy(update={"database_schema_version": 2})
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE run_manifest SET manifest_json = ?, manifest_sha256 = ?
            WHERE singleton_id = 1
            """,
            (canonical_json(wrong), canonical_sha256(wrong)),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="another database schema"):
            store.get_manifest()


def test_application_and_migration_identity_are_enforced(tmp_path: Path) -> None:
    wrong_app = tmp_path / "wrong-app.sqlite3"
    with SQLiteRunStore(wrong_app):
        pass
    with sqlite3.connect(wrong_app) as connection:
        connection.execute("PRAGMA application_id = 123")
    with pytest.raises(SchemaVersionError, match="another application"):
        SQLiteRunStore(wrong_app)

    wrong_migration = tmp_path / "wrong-migration.sqlite3"
    with SQLiteRunStore(wrong_migration):
        pass
    with sqlite3.connect(wrong_migration) as connection:
        connection.execute("UPDATE schema_migrations SET name = 'tampered'")
    with pytest.raises(SchemaVersionError, match="migration history"):
        SQLiteRunStore(wrong_migration)


def test_non_sqlite_file_is_reported_as_corruption(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    database.write_bytes(b"not a sqlite database")

    with pytest.raises(StoreCorruptionError, match="unable to open"):
        SQLiteRunStore(database)


def test_foreign_key_corruption_is_detected_before_materialization(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO responses(condition_id, response_json, response_sha256)
            VALUES (?, ?, ?)
            """,
            ("f" * 64, "{}", canonical_sha256({})),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="foreign-key"):
            store.verify_integrity()


def test_stored_turn_zero_history_is_rejected_even_if_checks_were_bypassed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, make_manifest(provider.provider_identity)) as store:
        store.prepare_turn(request)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE turns
            SET previous_condition_id = ?, previous_history_commitment_sha256 = ?
            WHERE condition_id = ?
            """,
            ("e" * 64, "f" * 64, request.condition_id),
        )

    with pytest.raises(StoreCorruptionError, match="quick_check"):
        SQLiteRunStore(database)


def test_stored_history_cannot_be_redirected_to_uncommitted_predecessor(
    tmp_path: Path,
) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    focal_zero = make_request(provider.provider_identity, prompt_sequence_id="focal")
    focal_one = make_request(
        provider.provider_identity,
        prompt_sequence_id="focal",
        turn_index=1,
    )
    uncommitted = make_request(provider.provider_identity, prompt_sequence_id="other")
    with SQLiteRunStore(database, manifest) as store:
        complete_request(store, provider, focal_zero)
        store.prepare_turn(focal_one, store.history_binding_for(focal_zero.condition_id))
        store.prepare_turn(uncommitted)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE turns
            SET previous_condition_id = ?, previous_history_commitment_sha256 = ?
            WHERE condition_id = ?
            """,
            (uncommitted.condition_id, "f" * 64, focal_one.condition_id),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="not committed"):
            store.get_turn(focal_one.condition_id)


def test_stored_history_cannot_cross_policy_axis_after_coherent_disk_tampering(
    tmp_path: Path,
) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    turn_zero = make_request(provider.provider_identity, turn_index=0)
    turn_one = make_request(provider.provider_identity, turn_index=1)
    with SQLiteRunStore(database, manifest) as store:
        complete_request(store, provider, turn_zero)
        store.prepare_turn(turn_one, store.history_binding_for(turn_zero.condition_id))

    tampered_condition = turn_one.condition.model_copy(update={"policy_id": "other-policy"})
    tampered_request = turn_one.model_copy(update={"condition": tampered_condition})
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE turns
            SET condition_id = ?, condition_json = ?, request_json = ?,
                request_sha256 = ?, policy_id = ?
            WHERE condition_id = ?
            """,
            (
                tampered_request.condition_id,
                canonical_json(tampered_condition),
                canonical_json(tampered_request),
                canonical_sha256(tampered_request),
                tampered_condition.policy_id,
                turn_one.condition_id,
            ),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="crosses unmatched condition axes"):
            store.get_turn(tampered_request.condition_id)

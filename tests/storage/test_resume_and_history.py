"""Crash-safe resume, history matching, and corruption tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ConfigDict

from neurallm.control.policy import PolicyState
from neurallm.providers.fake import FakeProvider
from neurallm.storage import (
    HistoryBinding,
    HistoryMismatchError,
    ResumeAction,
    SQLiteRunStore,
    StoreCorruptionError,
    TurnState,
    UncertainDispatchError,
)
from tests.storage.helpers import (
    complete_request,
    make_manifest,
    make_metrics,
    make_request,
)


class CounterState(PolicyState):
    """Concrete controller state used to verify typed rehydration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    count: int


def test_prepared_crash_resumes_without_duplicate_request(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, manifest) as store:
        store.prepare_turn(request)

    with SQLiteRunStore(database, manifest) as resumed:
        assert resumed.resume_action(request.condition_id) is ResumeAction.DISPATCH_PREPARED
        assert resumed.prepare_turn(request).state is TurnState.PREPARED
        assert len(resumed.list_turns()) == 1
        complete_request(resumed, provider, request)
        assert len(resumed.list_turns()) == 1


def test_dispatching_crash_becomes_terminal_uncertain_and_is_never_retried(
    tmp_path: Path,
) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, manifest) as store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)

    with SQLiteRunStore(database, manifest) as resumed:
        with pytest.raises(UncertainDispatchError, match="cannot be retried"):
            resumed.resume_action(request.condition_id)
        turn = resumed.get_turn(request.condition_id)
        assert turn.state is TurnState.UNCERTAIN_DISPATCH
        assert turn.uncertain_reason is not None
        with pytest.raises(UncertainDispatchError, match="cannot be retried"):
            resumed.begin_dispatch(request.condition_id)


def test_response_and_metric_checkpoints_have_safe_distinct_resume_actions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    request = make_request(provider.provider_identity)
    response = provider.generate(request)
    with SQLiteRunStore(database, manifest) as store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        store.persist_response(request.condition_id, response)

    with SQLiteRunStore(database, manifest) as resumed:
        assert resumed.resume_action(request.condition_id) is ResumeAction.COMPUTE_METRICS
        resumed.persist_metrics(request.condition_id, make_metrics(response))

    with SQLiteRunStore(database, manifest) as resumed_again:
        assert resumed_again.resume_action(request.condition_id) is ResumeAction.COMMIT


def test_nonzero_turn_requires_exact_committed_matched_history(tmp_path: Path) -> None:
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    turn_zero = make_request(provider.provider_identity, turn_index=0)
    turn_one = make_request(provider.provider_identity, turn_index=1)
    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        with pytest.raises(HistoryMismatchError, match="requires exact"):
            store.prepare_turn(turn_one)
        committed = complete_request(
            store,
            provider,
            turn_zero,
            policy_state=CounterState(count=4),
        )
        wrong = HistoryBinding(
            previous_condition_id=turn_zero.condition_id,
            previous_history_commitment_sha256="f" * 64,
        )
        with pytest.raises(HistoryMismatchError, match="does not match"):
            store.prepare_turn(turn_one, wrong)

        binding = store.history_binding_for(turn_zero.condition_id)
        prepared = store.prepare_turn(turn_one, binding)
        assert prepared.history == binding
        assert store.load_policy_state(turn_zero.condition_id, CounterState) == CounterState(
            count=4
        )
        assert committed.history_commitment_sha256 == binding.previous_history_commitment_sha256


def test_history_from_unmatched_sequence_fails_closed(tmp_path: Path) -> None:
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    previous = make_request(
        provider.provider_identity,
        turn_index=0,
        prompt_sequence_id="sequence-a",
    )
    current = make_request(
        provider.provider_identity,
        turn_index=1,
        prompt_sequence_id="sequence-b",
    )
    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        complete_request(store, provider, previous)
        with pytest.raises(HistoryMismatchError, match="different matched conditions"):
            store.prepare_turn(current, store.history_binding_for(previous.condition_id))


def test_history_from_another_policy_fails_closed(tmp_path: Path) -> None:
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    previous = make_request(provider.provider_identity, turn_index=0, policy_id="test-policy")
    current = make_request(provider.provider_identity, turn_index=1, policy_id="other-policy")
    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        complete_request(store, provider, previous)
        with pytest.raises(HistoryMismatchError, match="different matched conditions"):
            store.prepare_turn(current, store.history_binding_for(previous.condition_id))


def test_tampered_request_json_fails_hash_validation(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    request = make_request(provider.provider_identity)
    with SQLiteRunStore(database, manifest) as store:
        store.prepare_turn(request)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE turns SET request_json = '{}' WHERE condition_id = ?",
            (request.condition_id,),
        )

    with SQLiteRunStore(database, manifest) as store:
        with pytest.raises(StoreCorruptionError, match="generation request digest"):
            store.resume_action(request.condition_id)
        with pytest.raises(StoreCorruptionError):
            store.verify_integrity()


def test_tampered_history_binding_fails_closed_on_resume(tmp_path: Path) -> None:
    database = tmp_path / "run.sqlite3"
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    turn_zero = make_request(provider.provider_identity, turn_index=0)
    turn_one = make_request(provider.provider_identity, turn_index=1)
    with SQLiteRunStore(database, manifest) as store:
        complete_request(store, provider, turn_zero)
        store.prepare_turn(turn_one, store.history_binding_for(turn_zero.condition_id))
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE turns
            SET previous_history_commitment_sha256 = ?
            WHERE condition_id = ?
            """,
            ("f" * 64, turn_one.condition_id),
        )

    with SQLiteRunStore(database, manifest) as store:
        with pytest.raises(StoreCorruptionError, match="commitment is mismatched"):
            store.resume_action(turn_one.condition_id)

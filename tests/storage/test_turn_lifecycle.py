"""Transactional checkpoint and logical-request uniqueness tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neurallm.control.policy import PolicyState
from neurallm.domain.models import ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import GenerationResponse
from neurallm.providers.fake import FakeProvider
from neurallm.storage import (
    DuplicateLogicalRequestError,
    ResumeAction,
    SQLiteRunStore,
    StateTransitionError,
    StoreInvariantError,
    TurnState,
)
from tests.storage.helpers import (
    complete_request,
    make_manifest,
    make_metrics,
    make_request,
    make_trace,
)


def test_full_lifecycle_persists_raw_evidence_and_commits_once(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(
        database,
        make_manifest(provider.provider_identity),
    ) as store:
        prepared = store.prepare_turn(request)
        assert prepared.state is TurnState.PREPARED
        assert store.resume_action(request.condition_id) is ResumeAction.DISPATCH_PREPARED

        assert store.begin_dispatch(request.condition_id).state is TurnState.DISPATCHING
        response = provider.generate(request)
        persisted = store.persist_response(request.condition_id, response)
        assert persisted.state is TurnState.RESPONSE_PERSISTED
        assert persisted.response == response
        assert store.resume_action(request.condition_id) is ResumeAction.COMPUTE_METRICS

        metrics = make_metrics(response)
        measured = store.persist_metrics(request.condition_id, metrics)
        assert measured.state is TurnState.METRICS_COMPUTED
        assert measured.metrics == metrics
        assert store.resume_action(request.condition_id) is ResumeAction.COMMIT

        history = store.commit_turn(
            request.condition_id,
            PolicyState(),
            make_trace(request),
        )
        assert history.metrics == metrics
        assert len(history.history_commitment_sha256) == 64
        assert store.resume_action(request.condition_id) is ResumeAction.SKIP_COMMITTED
        assert store.prepare_turn(request).state is TurnState.COMMITTED
        with pytest.raises(StateTransitionError, match="cannot dispatch"):
            store.begin_dispatch(request.condition_id)
        assert len(store.list_turns()) == 1
        store.verify_integrity()

    with sqlite3.connect(database) as connection:
        raw_request, raw_response = connection.execute(
            """
            SELECT t.request_json, r.response_json
            FROM turns AS t
            JOIN responses AS r ON r.condition_id = t.condition_id
            WHERE t.condition_id = ?
            """,
            (request.condition_id,),
        ).fetchone()
    assert raw_request == canonical_json(request)
    assert f'"text":"{response.text}"' in raw_response


def test_same_condition_cannot_bind_a_different_request(tmp_path: Path) -> None:
    provider = FakeProvider()
    original = make_request(provider.provider_identity, prompt="first prompt")
    conflicting = make_request(provider.provider_identity, prompt="changed prompt")
    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        store.prepare_turn(original)
        with pytest.raises(DuplicateLogicalRequestError, match="different request"):
            store.prepare_turn(conflicting)
        assert store.get_turn(original.condition_id).request == original


def test_invalid_response_rolls_back_without_partial_evidence(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    other_identity = ProviderIdentity(
        provider_type="fake",
        implementation_version="1.0.0",
        model_alias="different-model",
        build_id="builtin",
        provider_config_hash=canonical_sha256("other-provider"),
    )
    good = provider.generate(request)
    invalid = GenerationResponse(
        text=good.text,
        provider_identity=other_identity,
        effective_parameters=good.effective_parameters,
        raw_metadata=good.raw_metadata,
    )
    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        with pytest.raises(StoreInvariantError, match="provider identity"):
            store.persist_response(request.condition_id, invalid)
        turn = store.get_turn(request.condition_id)
        assert turn.state is TurnState.DISPATCHING
        assert turn.response is None


def test_each_checkpoint_rejects_out_of_order_writes(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    response = provider.generate(request)
    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        store.prepare_turn(request)
        with pytest.raises(StateTransitionError, match="persist a response"):
            store.persist_response(request.condition_id, response)
        with pytest.raises(StateTransitionError, match="persist metrics"):
            store.persist_metrics(request.condition_id, make_metrics(response))
        with pytest.raises(StateTransitionError, match="commit"):
            store.commit_turn(request.condition_id, PolicyState(), make_trace(request))


def test_list_turns_has_deterministic_schedule_order(tmp_path: Path) -> None:
    provider = FakeProvider()
    first = make_request(provider.provider_identity, prompt_sequence_id="a")
    second = make_request(provider.provider_identity, prompt_sequence_id="b")
    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        store.prepare_turn(second)
        complete_request(store, provider, first)
        turns = store.list_turns()
        assert tuple(turn.condition.prompt_sequence_id for turn in turns) == ("a", "b")
        assert len(store.list_turns(TurnState.COMMITTED)) == 1

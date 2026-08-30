"""Public API validation and fail-closed edge-case tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurallm.control.policy import PolicyState
from neurallm.domain.models import ControllerAction, DecodingParameters, ProviderIdentity
from neurallm.domain.serialization import canonical_sha256
from neurallm.providers.base import GenerationMetadata
from neurallm.providers.fake import FakeProvider
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    HistoryBinding,
    HistoryMismatchError,
    ResumeAction,
    SchemaVersionError,
    SQLiteRunStore,
    StateTransitionError,
    StoreInvariantError,
    TurnState,
    UncertainDispatchError,
)
from tests.storage.helpers import (
    complete_request,
    make_manifest,
    make_metrics,
    make_request,
    make_trace,
)


def _store(tmp_path: Path) -> tuple[SQLiteRunStore, FakeProvider]:
    provider = FakeProvider()
    return (
        SQLiteRunStore(tmp_path / "run.sqlite3", make_manifest(provider.provider_identity)),
        provider,
    )


def test_constructor_and_closed_store_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        SQLiteRunStore(tmp_path / "run.sqlite3", timeout_seconds=0)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="must not be a directory"):
        SQLiteRunStore(directory)

    store = SQLiteRunStore(tmp_path / "closed.sqlite3")
    assert store.path == tmp_path / "closed.sqlite3"
    store.close()
    store.close()
    with pytest.raises(StoreInvariantError, match="closed"):
        store.list_turns()


def test_manifest_and_prepare_type_guards(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    with store:
        with pytest.raises(TypeError, match="RunManifest"):
            store.bind_manifest(object())  # type: ignore[arg-type]
        wrong_version = make_manifest(provider.provider_identity).model_copy(
            update={"database_schema_version": CURRENT_SCHEMA_VERSION + 1}
        )
        with pytest.raises(SchemaVersionError, match="not supported"):
            store.bind_manifest(wrong_version)
        with pytest.raises(TypeError, match="GenerationRequest"):
            store.prepare_turn(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="HistoryBinding"):
            store.prepare_turn(request, object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TurnState"):
            store.list_turns("PREPARED")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="condition_id"):
            store.get_turn(123)  # type: ignore[arg-type]


def test_request_provider_must_match_manifest(tmp_path: Path) -> None:
    store, _provider = _store(tmp_path)
    other_identity = ProviderIdentity(
        provider_type="fake",
        implementation_version="1.0.0",
        model_alias="other-fake",
        build_id="builtin",
        provider_config_hash=canonical_sha256("other"),
    )
    with store:
        with pytest.raises(StoreInvariantError, match="bound run manifest"):
            store.prepare_turn(make_request(other_identity))


def test_repeated_begin_dispatch_is_terminal_uncertain(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    with store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        with pytest.raises(UncertainDispatchError, match="cannot be retried"):
            store.begin_dispatch(request.condition_id)
        assert store.get_turn(request.condition_id).state is TurnState.UNCERTAIN_DISPATCH


def test_explicit_uncertain_dispatch_validates_reason_and_state(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    with store:
        store.prepare_turn(request)
        with pytest.raises(TypeError, match="reason"):
            store.mark_dispatch_uncertain(request.condition_id, 1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="blank"):
            store.mark_dispatch_uncertain(request.condition_id, "  ")
        with pytest.raises(StateTransitionError, match="only a DISPATCHING"):
            store.mark_dispatch_uncertain(request.condition_id, "provider failed")
        store.begin_dispatch(request.condition_id)
        uncertain = store.mark_dispatch_uncertain(request.condition_id, "provider failed")
        assert uncertain.state is TurnState.UNCERTAIN_DISPATCH
        assert uncertain.uncertain_reason == "provider failed"
        with pytest.raises(UncertainDispatchError):
            store.resume_action(request.condition_id)


def test_response_and_metric_type_guards(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    with store:
        store.prepare_turn(request)
        with pytest.raises(TypeError, match="GenerationResponse"):
            store.persist_response(request.condition_id, object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ResponseMetrics"):
            store.persist_metrics(request.condition_id, object())  # type: ignore[arg-type]


@pytest.mark.parametrize("mismatch", ["parameters", "metadata"])
def test_response_must_bind_exact_request(tmp_path: Path, mismatch: str) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    good = provider.generate(request)
    if mismatch == "parameters":
        parameters = DecodingParameters(
            temperature=0.9,
            top_p=good.effective_parameters.top_p,
            top_k=good.effective_parameters.top_k,
            presence_penalty=good.effective_parameters.presence_penalty,
            max_tokens=good.effective_parameters.max_tokens,
            seed=good.effective_parameters.seed,
        )
        response = good.model_copy(update={"effective_parameters": parameters})
        message = "effective parameters"
    else:
        metadata = GenerationMetadata(request_sha256="f" * 64)
        response = good.model_copy(update={"raw_metadata": metadata})
        message = "metadata"
    with store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        with pytest.raises(StoreInvariantError, match=message):
            store.persist_response(request.condition_id, response)
        assert store.get_turn(request.condition_id).response is None


def test_narrow_float32_effective_echo_is_persisted_without_normalization(
    tmp_path: Path,
) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    good = provider.generate(request)
    observed = good.effective_parameters.model_copy(
        update={
            "temperature": 0.800000011920929,
            "top_p": 0.949999988079071,
            "presence_penalty": 0.00000000596046448,
        }
    )
    response = good.model_copy(update={"effective_parameters": observed})

    with store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        persisted = store.persist_response(request.condition_id, response)

    assert persisted.response is not None
    assert persisted.response.effective_parameters == observed


def test_commit_validates_trace_and_action_bounds(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    response = provider.generate(request)
    with store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        store.persist_response(request.condition_id, response)
        store.persist_metrics(request.condition_id, make_metrics(response))
        with pytest.raises(TypeError, match="policy_state"):
            store.commit_turn(request.condition_id, object(), make_trace(request))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="policy_trace"):
            store.commit_turn(request.condition_id, PolicyState(), object())  # type: ignore[arg-type]
        wrong_policy = make_trace(request).model_copy(update={"policy_id": "other"})
        with pytest.raises(StoreInvariantError, match="policy_id"):
            store.commit_turn(request.condition_id, PolicyState(), wrong_policy)
        wrong_turn = make_trace(request).model_copy(update={"turn_index": 1})
        with pytest.raises(StoreInvariantError, match="turn_index"):
            store.commit_turn(request.condition_id, PolicyState(), wrong_turn)
        out_of_bounds = make_trace(request).model_copy(
            update={
                "action": ControllerAction(
                    temperature_delta=0.5,
                    top_p_delta=0.0,
                    top_k_delta=0,
                    presence_penalty_delta=0.0,
                )
            }
        )
        with pytest.raises(StoreInvariantError, match="exceeds manifest bounds"):
            store.commit_turn(request.condition_id, PolicyState(), out_of_bounds)


def test_history_lookup_and_state_type_fail_closed(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    request = make_request(provider.provider_identity)
    with store:
        store.prepare_turn(request)
        with pytest.raises(HistoryMismatchError, match="has not committed"):
            store.get_committed_history(request.condition_id)
        with pytest.raises(StoreInvariantError, match="unknown"):
            store.get_turn("f" * 64)
        complete_request(store, provider, request)
        with pytest.raises(TypeError, match="PolicyState subclass"):
            store.load_policy_state(request.condition_id, object)  # type: ignore[type-var]


def test_history_binding_rejects_turn_zero_uncommitted_and_skipped_turn(tmp_path: Path) -> None:
    store, provider = _store(tmp_path)
    turn_zero = make_request(provider.provider_identity, turn_index=0)
    binding = HistoryBinding(
        previous_condition_id="e" * 64,
        previous_history_commitment_sha256="f" * 64,
    )
    with store:
        with pytest.raises(HistoryMismatchError, match="turn zero"):
            store.prepare_turn(turn_zero, binding)
        store.prepare_turn(turn_zero)
        turn_one = make_request(provider.provider_identity, turn_index=1)
        uncommitted = HistoryBinding(
            previous_condition_id=turn_zero.condition_id,
            previous_history_commitment_sha256="f" * 64,
        )
        with pytest.raises(HistoryMismatchError, match="not committed"):
            store.prepare_turn(turn_one, uncommitted)
        complete_request(store, provider, turn_zero)
        turn_two = make_request(provider.provider_identity, turn_index=2)
        with pytest.raises(HistoryMismatchError, match="immediately previous"):
            store.prepare_turn(turn_two, store.history_binding_for(turn_zero.condition_id))


def test_resume_action_values_are_exhaustive() -> None:
    assert set(ResumeAction) == {
        ResumeAction.DISPATCH_PREPARED,
        ResumeAction.COMPUTE_METRICS,
        ResumeAction.COMMIT,
        ResumeAction.SKIP_COMMITTED,
    }

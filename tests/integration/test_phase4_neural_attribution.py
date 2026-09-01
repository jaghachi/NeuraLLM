"""Fake-provider causal proof for persistent and matched-reset neural arms."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from neurallm.control import (
    NeuralPolicyState,
    NeuralPolicyTrace,
    NeuralSubstrateState,
    PolicyState,
    PolicyTrace,
)
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    DecodingParameters,
    PromptFeatures,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.experiments import (
    CausalAppliedPolicyTrace,
    GitProvenance,
    PlannedTurn,
    build_run_manifest,
    execute_plan,
    prepare_experiment,
)
from neurallm.providers.base import GenerationRequest, GenerationResponse
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import export_closed_run, scientific_result_sha256
from neurallm.reporting.artifacts import _validate_phase4_mechanism_evidence
from neurallm.storage import SQLiteRunStore, StoredTurn, TurnState
from tests.storage.helpers import make_metrics

_PERSISTENT = "neural_persistent"
_RESET = "neural_matched_history_state_reset"


class _FailIfCalledProvider:
    """Expose the fake identity while making every dispatch an assertion failure."""

    def __init__(self) -> None:
        self._identity = FakeProvider().provider_identity
        self.calls = 0

    @property
    def provider_identity(self) -> ProviderIdentity:
        return self._identity

    def generate(self, _request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        raise AssertionError("provider must not be called for an invalid causal schedule")


class _CountingFakeProvider:
    """Count deterministic fake dispatches across a simulated process restart."""

    def __init__(self) -> None:
        self._delegate = FakeProvider()
        self.calls = 0

    @property
    def provider_identity(self) -> ProviderIdentity:
        return self._delegate.provider_identity

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return self._delegate.generate(request)


def _prepared_phase4():
    root = Path(__file__).resolve().parents[2]
    return prepare_experiment(
        root / "configs" / "experiments" / "phase4-neural-causal-smoke.yaml",
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )


def _completed_phase4_run(root: Path) -> tuple[Path, Path]:
    prepared = _prepared_phase4()
    provider = FakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    run_directory = root / "phase4-finalized-run"
    run_directory.mkdir()
    database = run_directory / "run.sqlite3"
    summary = execute_plan(
        prepared.plan,
        manifest,
        provider,
        prepared.policy_runtimes,
        database,
    )
    assert summary.planned_turns == summary.committed_turns == 6
    with SQLiteRunStore(database) as store:
        assert store.get_finalization() is not None
    return run_directory, database


def _rehash_neural_trace(
    trace: NeuralPolicyTrace,
    **updates: object,
) -> NeuralPolicyTrace:
    updated = trace.model_copy(update=updates)
    mechanism_sha256 = canonical_sha256(
        {
            "turn_index": updated.turn_index,
            "observation_encoding": updated.observation_encoding,
            "stored_substrate_state": updated.stored_substrate_state,
            "effective_substrate_state": updated.effective_substrate_state,
            "substrate_transition": updated.substrate_transition,
            "decoder_version": updated.decoder_version,
            "action_magnitude_version": updated.action_magnitude_version,
            "decoder_activation": updated.decoder_activation,
            "action": updated.action,
            "action_magnitude": updated.action_magnitude,
            "state_reset_applied": updated.state_reset_applied,
        }
    )
    return updated.model_copy(update={"mechanism_sha256": mechanism_sha256})


def _replace_causal_trace(
    turn: StoredTurn,
    *,
    outer_updates: dict[str, object] | None = None,
    neural_updates: dict[str, object] | None = None,
) -> StoredTurn:
    assert turn.policy_trace_json is not None
    trace = CausalAppliedPolicyTrace.model_validate_json(turn.policy_trace_json)
    if neural_updates:
        trace = trace.model_copy(
            update={
                "policy_trace": _rehash_neural_trace(
                    trace.policy_trace,
                    **neural_updates,
                )
            }
        )
    if outer_updates:
        trace = trace.model_copy(update=outer_updates)
    return replace(turn, policy_trace_json=canonical_json(trace))


def _tamper_terminal_action_application(database: Path, field: str) -> None:
    with SQLiteRunStore(database) as store:
        target = next(
            turn
            for turn in store.list_turns()
            if turn.condition.policy_id == _PERSISTENT and turn.condition.turn_index == 2
        )
        finalization = store.get_finalization()

    assert finalization is not None
    assert target.response is not None
    assert target.metrics is not None
    assert target.policy_state_json is not None
    assert target.policy_trace_json is not None
    trace = CausalAppliedPolicyTrace.model_validate_json(target.policy_trace_json)
    application = trace.action_application
    request = target.request
    response = target.response

    if field == "step":
        temperature_delta = application.step_clamped_action.temperature_delta
        tampered_step = application.step_clamped_action.model_copy(
            update={"temperature_delta": 0.0 if temperature_delta != 0.0 else 0.05}
        )
        application = application.model_copy(update={"step_clamped_action": tampered_step})
        trace = trace.model_copy(
            update={"action": tampered_step, "action_application": application}
        )
    elif field == "final":
        final_parameters = application.final_decoding_parameters
        tampered_temperature = (
            final_parameters.temperature + 0.01
            if final_parameters.temperature < 1.99
            else final_parameters.temperature - 0.01
        )
        tampered_parameters = final_parameters.model_copy(
            update={"temperature": tampered_temperature}
        )
        application = application.model_copy(
            update={"final_decoding_parameters": tampered_parameters}
        )
        trace = trace.model_copy(update={"action_application": application})
        request = request.model_copy(update={"decoding_parameters": tampered_parameters})
        request_sha256 = canonical_sha256(request)
        response = response.model_copy(
            update={
                "effective_parameters": tampered_parameters,
                "raw_metadata": response.raw_metadata.model_copy(
                    update={"request_sha256": request_sha256}
                ),
            }
        )
    elif field == "saturation":
        indicator = application.saturation.temperature
        tampered_saturation = application.saturation.model_copy(
            update={
                "temperature": indicator.model_copy(
                    update={"step_clamped": not indicator.step_clamped}
                )
            }
        )
        application = application.model_copy(update={"saturation": tampered_saturation})
        trace = trace.model_copy(update={"action_application": application})
    else:
        raise AssertionError(f"unsupported test tamper field: {field}")

    request_json = canonical_json(request)
    request_sha256 = canonical_sha256(request)
    response_json = canonical_json(response)
    response_sha256 = canonical_sha256(response)
    trace_json = canonical_json(trace)
    trace_sha256 = canonical_sha256(trace)
    metrics_sha256 = canonical_sha256(target.metrics)
    policy_state_sha256 = canonical_sha256(json.loads(target.policy_state_json))
    history_commitment_sha256 = canonical_sha256(
        {
            "condition_id": target.condition_id,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "metrics_sha256": metrics_sha256,
            "policy_state_sha256": policy_state_sha256,
            "policy_trace_sha256": trace_sha256,
            "previous_history_commitment_sha256": (
                target.history.previous_history_commitment_sha256
                if target.history is not None
                else None
            ),
        }
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE turns SET request_json = ?, request_sha256 = ? WHERE condition_id = ?",
            (request_json, request_sha256, target.condition_id),
        )
        connection.execute(
            "UPDATE responses SET response_json = ?, response_sha256 = ? WHERE condition_id = ?",
            (response_json, response_sha256, target.condition_id),
        )
        connection.execute(
            """
            UPDATE history_commitments
            SET policy_trace_json = ?,
                policy_trace_sha256 = ?,
                history_commitment_sha256 = ?
            WHERE condition_id = ?
            """,
            (
                trace_json,
                trace_sha256,
                history_commitment_sha256,
                target.condition_id,
            ),
        )

    with SQLiteRunStore(database) as store:
        store.verify_integrity()
        updated_result_sha256 = scientific_result_sha256(store.list_turns())
    updated_finalization = finalization.model_copy(
        update={"scientific_result_sha256": updated_result_sha256}
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE run_finalization
            SET finalization_json = ?, finalization_sha256 = ?
            WHERE singleton_id = 1
            """,
            (
                canonical_json(updated_finalization),
                canonical_sha256(updated_finalization),
            ),
        )

    with SQLiteRunStore(database) as store:
        store.verify_integrity()
        assert store.get_finalization() == updated_finalization


def test_fake_provider_run_proves_matched_history_and_substrate_causality(
    tmp_path: Path,
) -> None:
    prepared = _prepared_phase4()
    provider = FakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    run_directory = tmp_path / "phase4-neural-run"
    run_directory.mkdir()
    database = run_directory / "run.sqlite3"

    first = execute_plan(
        prepared.plan,
        manifest,
        provider,
        prepared.policy_runtimes,
        database,
    )
    replay = execute_plan(
        prepared.plan,
        manifest,
        provider,
        prepared.policy_runtimes,
        database,
    )

    assert first.planned_turns == first.committed_turns == 6
    assert first.provider_calls == 6
    assert replay.provider_calls == 0
    assert replay.manifest_sha256 == first.manifest_sha256
    assert dict(manifest.matched_history_policy_sources) == {_RESET: _PERSISTENT}

    artifacts = export_closed_run(run_directory)
    decision = json.loads((run_directory / "decision.json").read_text(encoding="utf-8"))
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert artifacts.implementation_phase == 4
    assert artifacts.manifest_sha256 == canonical_sha256(manifest)
    assert decision["implementation_phase"] == 4
    assert decision["claim_scope"] == "deterministic_mechanism_validation_only"
    assert decision["scientific_decision"] is None
    assert decision["scientific_result_sha256"] == artifacts.scientific_result_sha256
    assert decision["matched_history_policy_sources"] == {_RESET: _PERSISTENT}
    assert "Phase 4 Deterministic Mechanism Report" in report
    assert "does not establish neural efficacy" in report
    assert "Phase 2 Engineering Report" not in report

    with SQLiteRunStore(database) as store:
        turns = {
            (turn.condition.policy_id, turn.condition.turn_index): turn
            for turn in store.list_turns()
        }
        persistent_zero = turns[(_PERSISTENT, 0)]
        reset_zero = turns[(_RESET, 0)]
        assert persistent_zero.response is not None
        assert reset_zero.response is not None
        assert persistent_zero.response.text == reset_zero.response.text
        assert persistent_zero.request.decoding_parameters == reset_zero.request.decoding_parameters
        persistent_zero_trace = CausalAppliedPolicyTrace.model_validate_json(
            persistent_zero.policy_trace_json
        )
        reset_zero_trace = CausalAppliedPolicyTrace.model_validate_json(
            reset_zero.policy_trace_json
        )
        assert persistent_zero_trace.policy_trace.mechanism_sha256 == (
            reset_zero_trace.policy_trace.mechanism_sha256
        )
        assert persistent_zero.policy_state_json == reset_zero.policy_state_json
        assert persistent_zero_trace.observation_has_previous_response is False
        assert reset_zero_trace.observation_has_previous_response is False

        later_decoding_diverged = False
        later_response_diverged = False
        for turn_index in (1, 2):
            focal_previous = turns[(_PERSISTENT, turn_index - 1)]
            persistent = turns[(_PERSISTENT, turn_index)]
            reset = turns[(_RESET, turn_index)]
            expected_metrics_sha256 = canonical_sha256(focal_previous.metrics)

            assert persistent.history is not None
            assert reset.history is not None
            assert persistent.history.previous_condition_id == focal_previous.condition_id
            assert reset.history.previous_condition_id == focal_previous.condition_id
            assert persistent.history.previous_history_commitment_sha256 == (
                focal_previous.history_commitment_sha256
            )
            assert reset.history == persistent.history

            persistent_trace = CausalAppliedPolicyTrace.model_validate_json(
                persistent.policy_trace_json
            )
            reset_trace = CausalAppliedPolicyTrace.model_validate_json(reset.policy_trace_json)
            assert persistent_trace.history_source_policy_id == _PERSISTENT
            assert reset_trace.history_source_policy_id == _PERSISTENT
            assert persistent_trace.history_source_condition_id == (focal_previous.condition_id)
            assert reset_trace.history_source_condition_id == focal_previous.condition_id
            assert persistent_trace.history_commitment_sha256 == (
                focal_previous.history_commitment_sha256
            )
            assert reset_trace.history_commitment_sha256 == (
                focal_previous.history_commitment_sha256
            )
            assert persistent_trace.observation_metrics_sha256 == expected_metrics_sha256
            assert reset_trace.observation_metrics_sha256 == expected_metrics_sha256
            assert persistent_trace.policy_trace.observation_encoding == (
                reset_trace.policy_trace.observation_encoding
            )
            assert persistent_trace.policy_trace.stored_substrate_state == (
                reset_trace.policy_trace.stored_substrate_state
            )
            assert persistent_trace.policy_trace.effective_substrate_state == (
                persistent_trace.policy_trace.stored_substrate_state
            )
            assert reset_trace.policy_trace.effective_substrate_state != (
                reset_trace.policy_trace.stored_substrate_state
            )
            assert persistent_trace.policy_trace.state_reset_applied is False
            assert reset_trace.policy_trace.state_reset_applied is True
            assert persistent_trace.policy_trace.observation_encoding.turn_index == turn_index
            assert reset_trace.policy_trace.observation_encoding.turn_index == turn_index
            assert 0.0 <= persistent_trace.policy_trace.action_magnitude <= 1.0
            assert 0.0 <= reset_trace.policy_trace.action_magnitude <= 1.0
            assert set(
                persistent_trace.policy_trace.substrate_transition.saturation.model_dump()
            ) == {"excitation", "inhibition", "adaptation", "fatigue", "context"}
            action_saturation = reset_trace.action_application.saturation.model_dump()
            assert set(action_saturation) == {
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
            }
            assert all(
                set(indicator) == {"step_clamped", "legal_clamped"}
                for indicator in action_saturation.values()
            )

            persistent_state = NeuralPolicyState.model_validate_json(persistent.policy_state_json)
            reset_state = NeuralPolicyState.model_validate_json(reset.policy_state_json)
            assert (
                persistent_state.next_turn_index == reset_state.next_turn_index == (turn_index + 1)
            )
            assert persistent_state.controller_seed == reset_state.controller_seed
            assert persistent_state.action_bounds == reset_state.action_bounds

            later_decoding_diverged |= (
                persistent.request.decoding_parameters != reset.request.decoding_parameters
            )
            assert persistent.response is not None
            assert reset.response is not None
            later_response_diverged |= persistent.response.text != reset.response.text

        focal_one = turns[(_PERSISTENT, 1)]
        reset_one = turns[(_RESET, 1)]
        persistent_two = turns[(_PERSISTENT, 2)]
        reset_two = turns[(_RESET, 2)]
        assert focal_one.metrics is not None
        assert reset_one.metrics is not None
        assert canonical_sha256(focal_one.metrics) != canonical_sha256(reset_one.metrics)
        assert persistent_two.history is not None
        assert reset_two.history is not None
        assert persistent_two.history.previous_condition_id == focal_one.condition_id
        assert reset_two.history.previous_condition_id == focal_one.condition_id
        assert persistent_two.history.previous_condition_id != reset_one.condition_id
        assert reset_two.history.previous_condition_id != reset_one.condition_id
        assert later_decoding_diverged is True
        assert later_response_diverged is True


def test_matched_history_plan_is_turn_interleaved_before_any_dispatch() -> None:
    prepared = _prepared_phase4()

    schedule = tuple(
        (turn.condition.turn_index, turn.condition.policy_id) for turn in prepared.plan.turns
    )

    assert schedule == (
        (0, _PERSISTENT),
        (0, _RESET),
        (1, _PERSISTENT),
        (1, _RESET),
        (2, _PERSISTENT),
        (2, _RESET),
    )


def test_neural_run_resumes_from_a_durable_focal_response_without_regeneration(
    tmp_path: Path,
) -> None:
    prepared = _prepared_phase4()
    provider = _CountingFakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database = tmp_path / "phase4-neural-resume.sqlite3"
    crashed = False

    def crash_after_focal_turn_one_response(state: TurnState, turn: PlannedTurn) -> None:
        nonlocal crashed
        condition = turn.condition
        if (
            state is TurnState.RESPONSE_PERSISTED
            and condition.policy_id == _PERSISTENT
            and condition.turn_index == 1
            and not crashed
        ):
            crashed = True
            raise RuntimeError("simulated crash after durable focal response")

    with pytest.raises(RuntimeError, match="durable focal response"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database,
            checkpoint_hook=crash_after_focal_turn_one_response,
        )
    assert provider.calls == 3

    resumed_prepared = _prepared_phase4()
    assert resumed_prepared.plan == prepared.plan
    resumed = execute_plan(
        resumed_prepared.plan,
        manifest,
        provider,
        resumed_prepared.policy_runtimes,
        database,
    )

    assert resumed.committed_turns == 6
    assert resumed.provider_calls == 3
    assert provider.calls == 6
    with SQLiteRunStore(database) as store:
        turns = {
            (turn.condition.policy_id, turn.condition.turn_index): turn
            for turn in store.list_turns()
        }
        focal_one = turns[(_PERSISTENT, 1)]
        reset_two = turns[(_RESET, 2)]
        assert reset_two.history is not None
        assert reset_two.history.previous_condition_id == focal_one.condition_id


def test_forward_causal_dependency_fails_before_store_creation_or_dispatch(
    tmp_path: Path,
) -> None:
    prepared = _prepared_phase4()
    invalid_turns = tuple(
        sorted(
            prepared.plan.turns,
            key=lambda turn: (
                turn.condition.policy_id,
                turn.condition.turn_index,
            ),
        )
    )
    invalid_plan = prepared.plan.model_copy(update={"turns": invalid_turns})
    provider = _FailIfCalledProvider()
    manifest = build_run_manifest(
        invalid_plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database = tmp_path / "must-not-exist.sqlite3"

    with pytest.raises(ValueError, match="scheduled before"):
        execute_plan(
            invalid_plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database,
        )

    assert not database.exists()


def test_missing_reset_turn_fails_exact_pairing_before_store_or_dispatch(
    tmp_path: Path,
) -> None:
    prepared = _prepared_phase4()
    invalid_turns = tuple(
        turn
        for turn in prepared.plan.turns
        if not (turn.condition.policy_id == _RESET and turn.condition.turn_index == 1)
    )
    invalid_plan = prepared.plan.model_copy(update={"turns": invalid_turns})
    provider = _FailIfCalledProvider()
    manifest = build_run_manifest(
        invalid_plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database = tmp_path / "missing-reset-must-not-exist.sqlite3"

    with pytest.raises(ValueError, match="exact paired coverage"):
        execute_plan(
            invalid_plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database,
        )

    assert provider.calls == 0
    assert not database.exists()


def test_mismatched_current_attribution_inputs_fail_before_store_or_dispatch(
    tmp_path: Path,
) -> None:
    prepared = _prepared_phase4()
    invalid_turns = tuple(
        turn.model_copy(update={"prompt": f"{turn.prompt} confounded"})
        if turn.condition.policy_id == _RESET and turn.condition.turn_index == 1
        else turn
        for turn in prepared.plan.turns
    )
    invalid_plan = prepared.plan.model_copy(update={"turns": invalid_turns})
    provider = _FailIfCalledProvider()
    manifest = build_run_manifest(
        invalid_plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database = tmp_path / "mismatched-pair-must-not-exist.sqlite3"

    with pytest.raises(ValueError, match="share exact current inputs"):
        execute_plan(
            invalid_plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database,
        )

    assert not database.exists()


def test_finalized_phase4_export_rejects_missing_turn_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory, database = _completed_phase4_run(tmp_path)
    with SQLiteRunStore(database) as store:
        turn_inputs = store.list_turn_inputs()
    assert len(turn_inputs) == 6
    monkeypatch.setattr(
        SQLiteRunStore,
        "list_turn_inputs",
        lambda _store: turn_inputs[:-1],
    )

    with pytest.raises(ValueError, match="exact prompt-side evidence coverage"):
        export_closed_run(run_directory)


def test_finalized_phase4_export_rejects_mismatched_paired_turn_input_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory, database = _completed_phase4_run(tmp_path)
    with SQLiteRunStore(database) as store:
        turns = store.list_turns()
        turn_inputs = store.list_turn_inputs()
    reset_one_id = next(
        turn.condition_id
        for turn in turns
        if turn.condition.policy_id == _RESET and turn.condition.turn_index == 1
    )
    mismatched_inputs = tuple(
        evidence.model_copy(update={"prompt_case_id": "mismatched-paired-case"})
        if evidence.condition_id == reset_one_id
        else evidence
        for evidence in turn_inputs
    )
    monkeypatch.setattr(
        SQLiteRunStore,
        "list_turn_inputs",
        lambda _store: mismatched_inputs,
    )

    with pytest.raises(ValueError, match="mismatched prompt-side evidence"):
        export_closed_run(run_directory)


def test_finalized_phase4_export_reconstructs_neural_encoding_from_turn_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory, database = _completed_phase4_run(tmp_path)
    with SQLiteRunStore(database) as store:
        turns = store.list_turns()
        turn_inputs = store.list_turn_inputs()
    paired_turn_one_ids = {
        turn.condition_id
        for turn in turns
        if turn.condition.turn_index == 1 and turn.condition.policy_id in {_PERSISTENT, _RESET}
    }
    assert len(paired_turn_one_ids) == 2
    identically_modified_features = PromptFeatures({"unrecorded_prompt_signal": -64.0})
    modified_inputs = tuple(
        evidence.model_copy(update={"prompt_features": identically_modified_features})
        if evidence.condition_id in paired_turn_one_ids
        else evidence
        for evidence in turn_inputs
    )
    paired_inputs = tuple(
        evidence for evidence in modified_inputs if evidence.condition_id in paired_turn_one_ids
    )
    assert len(paired_inputs) == 2
    assert paired_inputs[0].prompt_features == paired_inputs[1].prompt_features
    monkeypatch.setattr(
        SQLiteRunStore,
        "list_turn_inputs",
        lambda _store: modified_inputs,
    )

    with pytest.raises(ValueError, match="trace does not match its prompt-side evidence"):
        export_closed_run(run_directory)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("schema", "current database schema"),
        ("policy_hashes", "exactly the persistent and reset policies"),
        ("incomplete_turn", "complete committed mechanism evidence"),
        ("outer_identity", "trace identity does not match"),
        ("nested_identity", "nested neural trace has the wrong identity"),
        ("request_parameters", "decoding parameters do not match"),
        ("step_action", "does not bind its step-clamped action"),
        ("raw_action", "does not bind its raw controller action"),
        ("history_access", "wrong history-access mode"),
        ("state_turn", "stored neural state has the wrong logical turn"),
        ("state_seed", "stored neural state has the wrong controller seed"),
        ("state_bounds", "stored neural state has the wrong action bounds"),
        ("state_substrate", "stored neural state does not match the traced transition"),
        ("substrate_equation", "does not reproduce the neural substrate equations"),
        ("action_magnitude", "wrong normalized action magnitude"),
        ("decoder", "does not reproduce the declared action decoder"),
        ("turn_zero_history", "turn zero must carry explicit null history"),
        ("turn_zero_initial_state", "turn zero does not use the declared initial state"),
        ("later_history_missing", "later turn lacks its focal history binding"),
        ("wrong_focal_prior", "history does not bind the focal prior turn"),
        ("causal_commitment", "causal trace does not match its committed focal history"),
        ("stored_substrate", "did not load the committed focal substrate"),
        ("paired_coverage", "attribution arms lack exact paired coverage"),
        ("current_prompt", "attribution pair has mismatched current prompts"),
        ("turn_zero_equivalence", "arms are not equivalent at turn zero"),
        ("paired_history", "arms do not share exact focal inputs"),
        ("persistent_intervention", "persistent arm contains an undeclared intervention"),
        ("reset_intervention", "reset arm did not apply the declared substrate reset"),
    ),
)
def test_phase4_export_rejects_specific_mechanism_evidence_corruption(
    corruption: str,
    message: str,
    tmp_path: Path,
) -> None:
    _run_directory, database = _completed_phase4_run(tmp_path)
    with SQLiteRunStore(database) as store:
        manifest = store.get_manifest()
        turns = list(store.list_turns())
        turn_inputs = list(store.list_turn_inputs())
    assert manifest is not None

    def locate(policy_id: str, turn_index: int) -> int:
        return next(
            index
            for index, turn in enumerate(turns)
            if turn.condition.policy_id == policy_id and turn.condition.turn_index == turn_index
        )

    persistent_zero_index = locate(_PERSISTENT, 0)
    reset_zero_index = locate(_RESET, 0)
    persistent_one_index = locate(_PERSISTENT, 1)
    reset_one_index = locate(_RESET, 1)
    persistent_two_index = locate(_PERSISTENT, 2)
    persistent_zero = turns[persistent_zero_index]
    reset_zero = turns[reset_zero_index]
    persistent_one = turns[persistent_one_index]
    reset_one = turns[reset_one_index]

    if corruption == "schema":
        manifest = manifest.model_copy(update={"database_schema_version": 1})
    elif corruption == "policy_hashes":
        manifest = manifest.model_copy(update={"policy_config_hashes": {_PERSISTENT: "f" * 64}})
    elif corruption == "incomplete_turn":
        turns[persistent_zero_index] = replace(persistent_zero, response=None)
    elif corruption == "outer_identity":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            outer_updates={"policy_id": _RESET},
        )
    elif corruption == "nested_identity":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            neural_updates={"policy_id": _RESET},
        )
    elif corruption == "request_parameters":
        assert persistent_zero.policy_trace_json is not None
        trace = CausalAppliedPolicyTrace.model_validate_json(persistent_zero.policy_trace_json)
        final_parameters = trace.action_application.final_decoding_parameters
        drifted_parameters = final_parameters.model_copy(
            update={"temperature": final_parameters.temperature + 0.01}
        )
        application = trace.action_application.model_copy(
            update={"final_decoding_parameters": drifted_parameters}
        )
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            outer_updates={"action_application": application},
        )
    elif corruption == "step_action":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            outer_updates={
                "action": ControllerAction(
                    temperature_delta=0.01,
                    top_p_delta=0.0,
                    top_k_delta=0,
                    presence_penalty_delta=0.0,
                )
            },
        )
    elif corruption == "raw_action":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            neural_updates={
                "action": ControllerAction(
                    temperature_delta=0.01,
                    top_p_delta=0.0,
                    top_k_delta=0,
                    presence_penalty_delta=0.0,
                )
            },
        )
    elif corruption == "history_access":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            outer_updates={"history_access": "matched_focal_previous_response"},
        )
    elif corruption.startswith("state_"):
        assert persistent_zero.policy_state_json is not None
        state = NeuralPolicyState.model_validate_json(persistent_zero.policy_state_json)
        if corruption == "state_turn":
            state = state.model_copy(update={"next_turn_index": state.next_turn_index + 1})
        elif corruption == "state_seed":
            state = state.model_copy(update={"controller_seed": state.controller_seed + 1})
        elif corruption == "state_bounds":
            state = state.model_copy(
                update={"action_bounds": ActionBounds(temperature_delta=(-0.05, 0.05))}
            )
        elif corruption == "state_substrate":
            persistent_two = turns[persistent_two_index]
            assert persistent_two.policy_state_json is not None
            state = NeuralPolicyState.model_validate_json(
                persistent_two.policy_state_json
            ).model_copy(
                update={
                    "substrate": NeuralSubstrateState(
                        excitation=1.0,
                        inhibition=1.0,
                        adaptation=1.0,
                        fatigue=1.0,
                        context=1.0,
                    )
                }
            )
        target_index = (
            persistent_two_index if corruption == "state_substrate" else persistent_zero_index
        )
        turns[target_index] = replace(
            turns[target_index],
            policy_state_json=canonical_json(state),
        )
    elif corruption == "substrate_equation":
        assert persistent_zero.policy_trace_json is not None
        trace = CausalAppliedPolicyTrace.model_validate_json(persistent_zero.policy_trace_json)
        transition = trace.policy_trace.substrate_transition
        seed_drive = 0.0 if transition.seed_drive != 0.0 else 0.001
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            neural_updates={
                "substrate_transition": transition.model_copy(update={"seed_drive": seed_drive})
            },
        )
    elif corruption == "action_magnitude":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            neural_updates={"action_magnitude": 0.5},
        )
    elif corruption == "decoder":
        assert persistent_zero.policy_trace_json is not None
        trace = CausalAppliedPolicyTrace.model_validate_json(persistent_zero.policy_trace_json)
        activation = trace.policy_trace.decoder_activation
        temperature = 0.0 if activation.temperature != 0.0 else 0.5
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            neural_updates={
                "decoder_activation": activation.model_copy(update={"temperature": temperature})
            },
        )
    elif corruption == "turn_zero_history":
        assert persistent_one.history is not None
        turns[persistent_zero_index] = replace(
            persistent_zero,
            history=persistent_one.history,
        )
    elif corruption == "turn_zero_initial_state":
        turns[persistent_zero_index] = _replace_causal_trace(
            persistent_zero,
            neural_updates={"state_reset_applied": True},
        )
    elif corruption == "later_history_missing":
        turns[persistent_one_index] = replace(persistent_one, history=None)
    elif corruption == "wrong_focal_prior":
        assert persistent_one.history is not None
        turns[persistent_one_index] = replace(
            persistent_one,
            history=persistent_one.history.model_copy(
                update={"previous_condition_id": reset_zero.condition_id}
            ),
        )
    elif corruption == "causal_commitment":
        turns[persistent_one_index] = _replace_causal_trace(
            persistent_one,
            outer_updates={"history_source_condition_id": reset_zero.condition_id},
        )
    elif corruption == "stored_substrate":
        turns[persistent_one_index] = _replace_causal_trace(
            persistent_one,
            neural_updates={
                "stored_substrate_state": NeuralSubstrateState(
                    excitation=1.0,
                    inhibition=1.0,
                    adaptation=1.0,
                    fatigue=1.0,
                    context=1.0,
                )
            },
        )
    elif corruption == "paired_coverage":
        reset_two_index = locate(_RESET, 2)
        reset_two_id = turns[reset_two_index].condition_id
        turns.pop(reset_two_index)
        turn_inputs = [
            evidence for evidence in turn_inputs if evidence.condition_id != reset_two_id
        ]
    elif corruption == "current_prompt":
        turns[reset_one_index] = replace(
            reset_one,
            request=reset_one.request.model_copy(update={"prompt": "confounded prompt"}),
        )
    elif corruption == "turn_zero_equivalence":
        assert reset_zero.response is not None
        turns[reset_zero_index] = replace(
            reset_zero,
            response=reset_zero.response.model_copy(update={"text": "different response"}),
        )
    elif corruption == "paired_history":
        assert reset_one.history is not None
        turns[reset_one_index] = replace(
            reset_one,
            history=reset_one.history.model_copy(
                update={"previous_history_commitment_sha256": "f" * 64}
            ),
        )
    elif corruption == "persistent_intervention":
        turns[persistent_one_index] = _replace_causal_trace(
            persistent_one,
            neural_updates={"state_reset_applied": True},
        )
    elif corruption == "reset_intervention":
        turns[reset_one_index] = _replace_causal_trace(
            reset_one,
            neural_updates={"state_reset_applied": False},
        )
    else:
        raise AssertionError(f"unsupported corruption: {corruption}")

    with pytest.raises(ValueError, match=message):
        _validate_phase4_mechanism_evidence(
            manifest,
            tuple(turns),
            tuple(turn_inputs),
        )


def test_finalized_phase4_export_rejects_an_extra_committed_policy_turn(
    tmp_path: Path,
) -> None:
    run_directory, database = _completed_phase4_run(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM run_finalization WHERE singleton_id = 1")

    provider = FakeProvider()
    with SQLiteRunStore(database) as store:
        persistent_zero = next(
            turn
            for turn in store.list_turns()
            if turn.condition.policy_id == _PERSISTENT and turn.condition.turn_index == 0
        )
        persistent_input = store.get_turn_input(persistent_zero.condition_id)
        assert persistent_input is not None
        extra_condition = persistent_zero.condition.model_copy(update={"policy_id": "best_static"})
        extra_request = persistent_zero.request.model_copy(update={"condition": extra_condition})
        extra_input = persistent_input.model_copy(
            update={"condition_id": extra_condition.condition_id}
        )
        store.prepare_turn(extra_request, input_evidence=extra_input)
        store.begin_dispatch(extra_condition.condition_id)
        response = provider.generate(extra_request)
        store.persist_response(extra_condition.condition_id, response)
        store.persist_metrics(extra_condition.condition_id, make_metrics(response))
        store.commit_turn(
            extra_condition.condition_id,
            PolicyState(),
            PolicyTrace(
                policy_id="best_static",
                turn_index=0,
                action=ControllerAction(
                    temperature_delta=0.0,
                    top_p_delta=0.0,
                    top_k_delta=0,
                    presence_penalty_delta=0.0,
                ),
            ),
        )
        turns = store.list_turns()
        assert len(turns) == 7
        store.finalize_run(
            tuple(turn.condition_id for turn in turns),
            scientific_result_sha256(turns),
        )
        store.verify_integrity()

    with pytest.raises(ValueError, match="store must contain exactly the two neural policies"):
        export_closed_run(run_directory)


@pytest.mark.parametrize("field", ("step", "final", "saturation"))
def test_finalized_phase4_export_recomputes_action_application_from_turn_zero(
    field: str,
    tmp_path: Path,
) -> None:
    run_directory, database = _completed_phase4_run(tmp_path)
    _tamper_terminal_action_application(database, field)

    with pytest.raises(
        ValueError,
        match="trace does not reproduce the declared action application",
    ):
        export_closed_run(run_directory)


def test_response_text_only_difference_does_not_satisfy_mechanism_divergence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run_directory, database = _completed_phase4_run(tmp_path)
    with SQLiteRunStore(database) as store:
        manifest = store.get_manifest()
        turns = store.list_turns()
        turn_inputs = store.list_turn_inputs()
    assert manifest is not None
    by_coordinate = {(turn.condition.policy_id, turn.condition.turn_index): turn for turn in turns}
    assert any(
        by_coordinate[(_PERSISTENT, turn_index)].response.text
        != by_coordinate[(_RESET, turn_index)].response.text
        for turn_index in (1, 2)
        if by_coordinate[(_PERSISTENT, turn_index)].response is not None
        and by_coordinate[(_RESET, turn_index)].response is not None
    )

    # Isolate the final attribution predicate: all independently validated mechanism
    # values compare equal while the committed provider texts remain different.
    for model_type in (NeuralSubstrateState, ControllerAction, DecodingParameters):
        monkeypatch.setattr(model_type, "__eq__", lambda _self, _other: True)
        monkeypatch.setattr(model_type, "__ne__", lambda _self, _other: False, raising=False)

    with pytest.raises(ValueError, match="requires a later paired mechanism divergence"):
        _validate_phase4_mechanism_evidence(manifest, turns, turn_inputs)

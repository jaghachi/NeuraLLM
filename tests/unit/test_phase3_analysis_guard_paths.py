"""Fail-closed unit paths for reconstruction of Phase 3 analysis evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from neurallm.control.action_space import apply_action
from neurallm.control.policy import PolicyState, PolicyTrace
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    DecodingBounds,
    PromptFeatures,
    RunManifest,
    SeedSchedule,
    UnitIntervalMetricValue,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    MatchedUnitKey,
    SequenceExpectation,
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    select_best_static,
)
from neurallm.experiments.analysis import (
    _normalized_action_magnitude,
    _trace_evidence,
    _validate_plan_manifest,
    analyze_closed_run,
    build_evaluation_design,
    build_phase3_analysis_contract_sha256,
    evaluation_records_from_store,
)
from neurallm.experiments.matching import materialize_matched_coverage
from neurallm.experiments.plan import (
    PHASE2_DECISION_RULE_VERSION,
    PHASE3_DECISION_RULE_VERSION,
    ExperimentPlan,
    PlannedTurn,
)
from neurallm.experiments.runner import DetailedAppliedPolicyTrace
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
)
from neurallm.storage import (
    AnalysisFinalization,
    AnalysisManifest,
    HistoryBinding,
    SQLiteRunStore,
    StoredTurn,
    StoreInvariantError,
    TurnInputEvidence,
    TurnState,
)
from tests.storage.helpers import make_manifest, make_metrics, make_request


def _selection_record() -> StaticSelectionRecord:
    first = StaticProfile(
        profile_id="first",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=64,
    )
    second = first.model_copy(update={"profile_id": "second", "temperature": 0.8})
    return select_best_static(
        (
            StaticCandidateResult(profile=first, unit_scores=(0.8,)),
            StaticCandidateResult(profile=second, unit_scores=(0.6,)),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=canonical_sha256("development"),
        development_unit_keys=(MatchedUnitKey(prompt_sequence_id="development-a", model_seed=7),),
    )


def _phase3_plan(*, turn_index: int = 0) -> ExperimentPlan:
    provider = FakeProvider()
    request = make_request(
        provider.provider_identity,
        turn_index=turn_index,
        policy_id="test-policy",
    )
    planned = PlannedTurn(
        condition=request.condition,
        prompt_case_id="case-a",
        prompt_family="constrained",
        prompt_features=PromptFeatures({}),
        prompt=request.prompt,
        validator=ValidatorSpec(kind="non_empty"),
        decoding_parameters=request.decoding_parameters,
    )
    spec = EvaluationSpec(
        focal_policy_id="test-policy",
        required_serious_comparator_ids=("baseline-policy",),
        bootstrap_resamples=4,
        bootstrap_seed=17,
        permutation_resamples=4,
        permutation_seed=19,
    )
    matched_units = materialize_matched_coverage(
        (request.condition,),
        experiment_id=request.condition.experiment_id,
        dataset_version=request.condition.dataset_version,
        sequence_turn_indexes={request.condition.prompt_sequence_id: (turn_index,)},
        policy_ids=(request.condition.policy_id,),
        model_seeds=(request.condition.model_seed,),
        controller_seeds=(request.condition.controller_seed,),
    )
    selection = _selection_record()
    return ExperimentPlan(
        experiment_id=request.condition.experiment_id,
        dataset_version=request.condition.dataset_version,
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        experiment_config_hash=canonical_sha256("experiment-config"),
        dataset_hash=canonical_sha256("dataset"),
        provider_identity=provider.provider_identity,
        provider_effective_configuration_json=fake_provider_effective_configuration_json(),
        action_bounds=ActionBounds(),
        decoding_bounds=DecodingBounds(),
        metric_versions={"test-metrics": "1.0.0"},
        decision_rule_version=PHASE3_DECISION_RULE_VERSION,
        database_schema_version=2,
        evaluation=spec,
        evaluation_spec_sha256=canonical_sha256(spec),
        static_selection_record=selection,
        static_selection_result_sha256=selection.selection_result_sha256,
        matched_units=matched_units,
        turns=(planned,),
    )


def _stored_turn(
    plan: ExperimentPlan,
    *,
    raw_action: ControllerAction | None = None,
) -> tuple[StoredTurn, TurnInputEvidence]:
    planned = plan.turns[0]
    action = raw_action or ControllerAction(
        temperature_delta=0.0,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )
    application = apply_action(
        planned.decoding_parameters,
        action,
        plan.action_bounds,
        plan.decoding_bounds,
    )
    request = planned.generation_request.model_copy(
        update={"decoding_parameters": application.final_decoding_parameters}
    )
    response = FakeProvider().generate(request)
    trace = DetailedAppliedPolicyTrace(
        policy_id=planned.condition.policy_id,
        turn_index=planned.condition.turn_index,
        action=application.step_clamped_action,
        action_application=application,
        history_access="none",
        observation_has_previous_response=False,
        policy_trace=PolicyTrace(
            policy_id=planned.condition.policy_id,
            turn_index=planned.condition.turn_index,
            action=action,
        ),
    )
    turn = StoredTurn(
        condition_id=planned.condition.condition_id,
        request_sha256=canonical_sha256(request),
        state=TurnState.COMMITTED,
        condition=planned.condition,
        request=request,
        history=None,
        response=response,
        metrics=make_metrics(response),
        policy_state_json=canonical_json(PolicyState()),
        policy_trace_json=canonical_json(trace),
        history_commitment_sha256=canonical_sha256("history"),
        uncertain_reason=None,
    )
    evidence = TurnInputEvidence(
        condition_id=planned.condition.condition_id,
        prompt_case_id=planned.prompt_case_id,
        prompt_family=planned.prompt_family,
        prompt_features=planned.prompt_features,
        validator=planned.validator,
    )
    return turn, evidence


def _synthetic_manifest() -> AnalysisManifest:
    provider = FakeProvider()
    spec = EvaluationSpec(
        focal_policy_id="focal",
        required_serious_comparator_ids=("static",),
        bootstrap_resamples=4,
        bootstrap_seed=1,
        permutation_resamples=4,
        permutation_seed=2,
    )
    design = ExpectedEvaluationDesign(
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        dataset_sha256=canonical_sha256("dataset"),
        provider_identity_id=provider.provider_identity.identity_id,
        sequences=(SequenceExpectation(prompt_sequence_id="sequence-a", turn_count=1),),
        model_seeds=(7,),
        controller_seeds=(11,),
        policy_ids=("focal", "static"),
    )
    selection = _selection_record()
    return AnalysisManifest(
        run_manifest_sha256=canonical_sha256("manifest"),
        run_finalization_sha256=canonical_sha256("finalization"),
        scientific_result_sha256=canonical_sha256("result"),
        experiment_plan_sha256=canonical_sha256("plan"),
        evaluation_spec=spec,
        evaluation_spec_sha256=canonical_sha256(spec),
        static_selection_record=selection,
        static_selection_result_sha256=selection.selection_result_sha256,
        evaluation_design=design,
        dataset_sha256=design.dataset_sha256,
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        evaluation_input_sha256=canonical_sha256("input"),
    )


def _manifest_matching_phase3_plan(plan: ExperimentPlan) -> RunManifest:
    return make_manifest(plan.provider_identity).model_copy(
        update={
            "experiment_config_hash": plan.experiment_config_hash,
            "dataset_hash": plan.dataset_hash,
            "provider_effective_configuration_json": (plan.provider_effective_configuration_json),
            "policy_config_hashes": {"test-policy": canonical_sha256("test-policy")},
            "metric_versions": dict(plan.metric_versions),
            "seed_schedule": SeedSchedule(model_seeds=(7,), controller_seeds=(11,)),
            "action_bounds": plan.action_bounds,
            "decoding_bounds": plan.decoding_bounds,
            "decision_rule_version": PHASE3_DECISION_RULE_VERSION,
            "database_schema_version": plan.database_schema_version,
            "phase3_analysis_contract_sha256": build_phase3_analysis_contract_sha256(plan),
        }
    )


def _validate_dataset_boundary(manifest: AnalysisManifest) -> AnalysisManifest:
    validator = cast(
        Callable[[], AnalysisManifest],
        manifest.validate_dataset_boundary,
    )
    return validator()


def test_normalized_action_magnitude_handles_zero_width_bounds() -> None:
    zero = ControllerAction(
        temperature_delta=0.0,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )
    zero_bounds = ActionBounds(
        temperature_delta=(0.0, 0.0),
        top_p_delta=(0.0, 0.0),
        top_k_delta=(0, 0),
        presence_penalty_delta=(0.0, 0.0),
    )

    assert _normalized_action_magnitude(zero, zero_bounds) == 0.0
    with pytest.raises(StoreInvariantError, match="zero-width"):
        _normalized_action_magnitude(
            zero.model_copy(update={"temperature_delta": 0.1}),
            zero_bounds,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "missing its policy trace"),
        ("invalid-json", "not valid JSON"),
        ("not-object", "not a JSON object"),
        ("shape", "unexpected evidence shape"),
        ("noncanonical", "not canonical JSON"),
        ("policy", "another policy"),
        ("turn", "another turn"),
        ("schema", "unsupported schema"),
        ("history-shape", "invalid history-access evidence"),
        ("history-causality", "causally inconsistent"),
        ("action", "does not match its clamped action"),
        ("parameters", "do not match the provider request"),
    ],
)
def test_trace_evidence_rejects_malformed_or_misaligned_payloads(
    case: str,
    message: str,
) -> None:
    plan = _phase3_plan()
    turn, _ = _stored_turn(plan)
    assert turn.policy_trace_json is not None
    payload = json.loads(turn.policy_trace_json)
    trace_json: str | None
    if case == "missing":
        trace_json = None
    elif case == "invalid-json":
        trace_json = "{"
    elif case == "not-object":
        trace_json = canonical_json([])
    elif case == "shape":
        trace_json = canonical_json({"unexpected": True})
    elif case == "noncanonical":
        trace_json = json.dumps(payload, indent=2, sort_keys=True)
    else:
        if case == "policy":
            payload["policy_id"] = "other-policy"
        elif case == "turn":
            payload["turn_index"] = 1
        elif case == "schema":
            payload["trace_schema_version"] = "unknown"
        elif case == "history-shape":
            payload["history_access"] = "all_history"
        elif case == "history-causality":
            payload["history_access"] = "own_previous_response"
            payload["observation_has_previous_response"] = True
        elif case == "action":
            payload["action"]["temperature_delta"] = 0.05
        elif case == "parameters":
            payload["action_application"]["final_decoding_parameters"]["temperature"] = 0.9
        trace_json = canonical_json(payload)

    with pytest.raises(StoreInvariantError, match=message):
        _trace_evidence(replace(turn, policy_trace_json=trace_json), plan.action_bounds)


def test_trace_evidence_reports_clamping_bounds_and_saturation() -> None:
    plan = _phase3_plan()
    turn, _ = _stored_turn(
        plan,
        raw_action=ControllerAction(
            temperature_delta=0.5,
            top_p_delta=0.0,
            top_k_delta=0,
            presence_penalty_delta=0.0,
        ),
    )

    magnitude, within_bounds, saturated, observed_history = _trace_evidence(
        turn, plan.action_bounds
    )

    assert magnitude == pytest.approx(0.5)
    assert within_bounds is False
    assert saturated is True
    assert observed_history is False


def test_analysis_public_helpers_reject_wrong_types_and_phase2_plans(tmp_path: Path) -> None:
    plan = _phase3_plan()
    phase2 = plan.model_copy(
        update={
            "decision_rule_version": PHASE2_DECISION_RULE_VERSION,
            "evaluation": None,
            "evaluation_spec_sha256": None,
            "static_selection_record": None,
            "static_selection_result_sha256": None,
            "matched_units": (),
        }
    )
    selection = _selection_record()
    with SQLiteRunStore(tmp_path / "empty.sqlite3") as store:
        with pytest.raises(TypeError, match="ExperimentPlan"):
            build_evaluation_design(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ExperimentPlan"):
            evaluation_records_from_store(object(), store)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SQLiteRunStore"):
            evaluation_records_from_store(plan, object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Phase 3 plan"):
            evaluation_records_from_store(phase2, store)
    with pytest.raises(ValueError, match="typed Phase 3 plan"):
        build_evaluation_design(phase2)
    with pytest.raises(TypeError, match="ExperimentPlan"):
        analyze_closed_run(object(), selection, tmp_path / "run.sqlite3")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StaticSelectionRecord"):
        analyze_closed_run(plan, object(), tmp_path / "run.sqlite3")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pathlib.Path"):
        analyze_closed_run(plan, selection, "run.sqlite3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Phase 3 EvaluationSpec"):
        analyze_closed_run(phase2, selection, tmp_path / "run.sqlite3")


def test_build_design_rejects_noncontiguous_turn_indexes() -> None:
    plan = _phase3_plan()
    planned = plan.turns[0]
    shifted_condition = planned.condition.model_copy(update={"turn_index": 1})
    shifted = planned.model_copy(update={"condition": shifted_condition})
    noncontiguous = plan.model_copy(update={"turns": (shifted,)})

    with pytest.raises(ValueError, match="contiguous from zero"):
        build_evaluation_design(noncontiguous)


def test_manifest_plan_match_is_exact() -> None:
    plan = _phase3_plan()
    manifest = _manifest_matching_phase3_plan(plan)

    _validate_plan_manifest(plan, manifest)
    with pytest.raises(StoreInvariantError, match="does not exactly match"):
        _validate_plan_manifest(
            plan,
            manifest.model_copy(update={"experiment_config_hash": "f" * 64}),
        )


def test_phase3_analysis_rejects_a_nonempty_matched_history_map() -> None:
    plan = _phase3_plan()
    manifest = _manifest_matching_phase3_plan(plan).model_copy(
        update={
            "matched_history_policy_sources": {
                "neural_matched_history_state_reset": "neural_persistent"
            }
        }
    )

    with pytest.raises(StoreInvariantError, match="does not exactly match"):
        _validate_plan_manifest(plan, manifest)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("turn-set", "conditions do not exactly match"),
        ("input-set", "lack exact prompt-side"),
        ("uncommitted", "committed metric evidence"),
        ("plan", "differs from its frozen plan"),
        ("input", "prompt-side evidence differs"),
        ("history", "history presence disagrees"),
    ],
)
def test_evaluation_record_reconstruction_fails_closed(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _phase3_plan()
    turn, evidence = _stored_turn(plan)
    turns: tuple[StoredTurn, ...] = (turn,)
    inputs: tuple[TurnInputEvidence, ...] = (evidence,)
    if case == "turn-set":
        turns = ()
    elif case == "input-set":
        inputs = ()
    elif case == "uncommitted":
        turns = (replace(turn, state=TurnState.PREPARED, metrics=None),)
    elif case == "plan":
        turns = (
            replace(
                turn,
                request=turn.request.model_copy(update={"prompt": "different prompt"}),
            ),
        )
    elif case == "input":
        inputs = (evidence.model_copy(update={"prompt_family": "other-family"}),)
    elif case == "history":
        turns = (
            replace(
                turn,
                history=HistoryBinding(
                    previous_condition_id="a" * 64,
                    previous_history_commitment_sha256="b" * 64,
                ),
            ),
        )

    monkeypatch.setattr(SQLiteRunStore, "list_turns", lambda _store: turns)
    monkeypatch.setattr(SQLiteRunStore, "list_turn_inputs", lambda _store: inputs)
    with SQLiteRunStore(tmp_path / f"{case}.sqlite3") as store:
        with pytest.raises(StoreInvariantError, match=message):
            evaluation_records_from_store(plan, store)


def test_evaluation_record_preserves_explicit_metric_unavailability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _phase3_plan()
    turn, evidence = _stored_turn(plan)
    assert turn.metrics is not None
    unavailable = UnitIntervalMetricValue(
        value=None,
        availability=False,
        metric_version="test-v1",
        input_hash=turn.metrics.task_score.input_hash,
    )
    turn = replace(
        turn,
        metrics=turn.metrics.model_copy(update={"task_score": unavailable}),
    )
    monkeypatch.setattr(SQLiteRunStore, "list_turns", lambda _store: (turn,))
    monkeypatch.setattr(SQLiteRunStore, "list_turn_inputs", lambda _store: (evidence,))
    with SQLiteRunStore(tmp_path / "unavailable.sqlite3") as store:
        records = evaluation_records_from_store(plan, store)

    assert records[0].task_score is None
    assert records[0].required_metrics_available is False


def test_analysis_manifest_enforces_dataset_and_evidence_identities() -> None:
    manifest = _synthetic_manifest()
    with pytest.raises(ValueError, match="development data"):
        _validate_dataset_boundary(
            manifest.model_copy(update={"dataset_purpose": DatasetPurpose.DEVELOPMENT})
        )
    with pytest.raises(ValueError, match="requires a dataset seal"):
        _validate_dataset_boundary(
            manifest.model_copy(update={"dataset_purpose": DatasetPurpose.EVALUATION})
        )
    with pytest.raises(ValueError, match="must not claim"):
        _validate_dataset_boundary(manifest.model_copy(update={"dataset_seal_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="evaluation spec hash"):
        _validate_dataset_boundary(manifest.model_copy(update={"evaluation_spec_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="static selection hash"):
        _validate_dataset_boundary(
            manifest.model_copy(update={"static_selection_result_sha256": "a" * 64})
        )
    with pytest.raises(ValueError, match="evaluation design disagrees"):
        _validate_dataset_boundary(manifest.model_copy(update={"dataset_sha256": "a" * 64}))

    sealed_design = manifest.evaluation_design.model_copy(
        update={
            "dataset_purpose": DatasetPurpose.EVALUATION,
            "dataset_seal_sha256": "b" * 64,
        }
    )
    sealed = manifest.model_copy(
        update={
            "dataset_purpose": DatasetPurpose.EVALUATION,
            "dataset_seal_sha256": "b" * 64,
            "evaluation_design": sealed_design,
        }
    )
    assert _validate_dataset_boundary(sealed) == sealed


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"comparison_result_sha256s": ("b" * 64, "a" * 64)},
            "sorted and unique",
        ),
        ({"comparison_count": 1}, "comparison count"),
        ({"guardrail_count": 2}, "guardrail count"),
    ],
)
def test_analysis_finalization_validates_sorted_hashes_and_counts(
    updates: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "analysis_manifest_sha256": "1" * 64,
        "evaluation_result_sha256": "2" * 64,
        "decision_sha256": "3" * 64,
        "comparison_result_sha256s": (),
        "guardrail_result_sha256s": ("4" * 64,),
        "comparison_count": 0,
        "guardrail_count": 1,
    }
    payload.update(updates)
    with pytest.raises(ValidationError, match=message):
        AnalysisFinalization.model_validate(payload)

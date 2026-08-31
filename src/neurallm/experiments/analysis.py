"""Offline Phase 3 analysis over a closed canonical run store."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from neurallm.control.action_space import (
    ActionApplication,
    normalized_action_magnitude,
)
from neurallm.domain.models import ActionBounds, ControllerAction, RunManifest
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import (
    ExpectedEvaluationDesign,
    Phase3EvaluationResult,
    SequenceExpectation,
    StaticSelectionRecord,
    TurnEvaluationRecord,
    evaluate_phase3,
)
from neurallm.evaluation.contract import phase3_analysis_contract_sha256
from neurallm.experiments.plan import PHASE3_DECISION_RULE_VERSION, ExperimentPlan
from neurallm.storage import (
    AnalysisManifest,
    SQLiteRunStore,
    StoredAnalysis,
    StoredTurn,
    StoreInvariantError,
    TurnState,
)


def _normalized_action_magnitude(action: ControllerAction, bounds: ActionBounds) -> float:
    """Return RMS action magnitude after normalizing each declared control dimension."""

    try:
        return normalized_action_magnitude(action, bounds)
    except ValueError as exc:
        raise StoreInvariantError(str(exc)) from exc


def _trace_evidence(
    turn: StoredTurn,
    bounds: ActionBounds,
) -> tuple[float, bool, bool, bool]:
    if turn.policy_trace_json is None:
        raise StoreInvariantError("committed Phase 3 turn is missing its policy trace")
    try:
        raw_payload: object = json.loads(turn.policy_trace_json)
    except json.JSONDecodeError as exc:
        raise StoreInvariantError("committed policy trace is not valid JSON") from exc
    if not isinstance(raw_payload, dict) or not all(isinstance(key, str) for key in raw_payload):
        raise StoreInvariantError("committed policy trace is not a JSON object")
    payload: Mapping[str, object] = raw_payload
    if set(payload) != {
        "action",
        "action_application",
        "history_access",
        "observation_has_previous_response",
        "policy_id",
        "policy_trace",
        "trace_schema_version",
        "turn_index",
    }:
        raise StoreInvariantError("Phase 3 policy trace has an unexpected evidence shape")
    if canonical_json(payload) != turn.policy_trace_json:
        raise StoreInvariantError("Phase 3 policy trace is not canonical JSON")
    if payload["policy_id"] != turn.condition.policy_id:
        raise StoreInvariantError("Phase 3 policy trace targets another policy")
    if payload["turn_index"] != turn.condition.turn_index:
        raise StoreInvariantError("Phase 3 policy trace targets another turn")
    if payload["trace_schema_version"] != "phase3-applied-policy-trace-v1":
        raise StoreInvariantError("Phase 3 policy trace has an unsupported schema")
    history_access = payload["history_access"]
    observation_has_previous_response = payload["observation_has_previous_response"]
    if history_access not in {"none", "own_previous_response"} or not isinstance(
        observation_has_previous_response,
        bool,
    ):
        raise StoreInvariantError("Phase 3 policy trace has invalid history-access evidence")
    expected_history = history_access == "own_previous_response" and turn.condition.turn_index > 0
    if observation_has_previous_response != expected_history:
        raise StoreInvariantError("Phase 3 policy trace history access is causally inconsistent")
    action = ControllerAction.model_validate(payload["action"])
    application = ActionApplication.model_validate(payload["action_application"])
    if action != application.step_clamped_action:
        raise StoreInvariantError("applied trace action does not match its clamped action")
    if application.final_decoding_parameters != turn.request.decoding_parameters:
        raise StoreInvariantError("applied trace parameters do not match the provider request")
    return (
        _normalized_action_magnitude(action, bounds),
        bounds.contains(application.raw_action),
        application.saturation.any_saturation,
        observation_has_previous_response,
    )


def build_evaluation_design(plan: ExperimentPlan) -> ExpectedEvaluationDesign:
    """Construct the exact evaluator grid declared by a Phase 3 plan."""

    if not isinstance(plan, ExperimentPlan):
        raise TypeError("plan must be an ExperimentPlan")
    if plan.evaluation is None or plan.dataset_purpose is None:
        raise ValueError("evaluation design requires a typed Phase 3 plan")
    sequence_indexes: dict[str, set[int]] = {}
    for turn in plan.turns:
        sequence_indexes.setdefault(turn.condition.prompt_sequence_id, set()).add(
            turn.condition.turn_index
        )
    sequences: list[SequenceExpectation] = []
    for sequence_id, indexes in sorted(sequence_indexes.items()):
        expected_indexes = set(range(len(indexes)))
        if indexes != expected_indexes:
            raise ValueError("Phase 3 sequence turn indexes must be contiguous from zero")
        sequences.append(
            SequenceExpectation(
                prompt_sequence_id=sequence_id,
                turn_count=len(indexes),
            )
        )
    dataset_seal_sha256 = None if plan.dataset_seal is None else plan.dataset_seal.seal_sha256
    return ExpectedEvaluationDesign(
        dataset_purpose=plan.dataset_purpose,
        dataset_sha256=plan.dataset_hash,
        dataset_seal_sha256=dataset_seal_sha256,
        provider_identity_id=plan.provider_identity.identity_id,
        sequences=tuple(sequences),
        model_seeds=tuple(sorted({turn.condition.model_seed for turn in plan.turns})),
        controller_seeds=tuple(sorted({turn.condition.controller_seed for turn in plan.turns})),
        policy_ids=tuple(sorted({turn.condition.policy_id for turn in plan.turns})),
    )


def build_phase3_analysis_contract_sha256(plan: ExperimentPlan) -> str:
    """Return the evaluator provenance digest frozen into the run manifest."""

    if not isinstance(plan, ExperimentPlan):
        raise TypeError("plan must be an ExperimentPlan")
    if (
        plan.evaluation is None
        or plan.evaluation_spec_sha256 is None
        or plan.static_selection_record is None
        or plan.static_selection_result_sha256 is None
        or plan.dataset_purpose is None
    ):
        raise ValueError("analysis contract requires a complete Phase 3 plan")
    design = build_evaluation_design(plan)
    return phase3_analysis_contract_sha256(
        experiment_plan_sha256=plan.scientific_identity_sha256,
        evaluation_spec=plan.evaluation,
        evaluation_spec_sha256=plan.evaluation_spec_sha256,
        static_selection_record=plan.static_selection_record,
        static_selection_result_sha256=plan.static_selection_result_sha256,
        evaluation_design=design,
        dataset_sha256=plan.dataset_hash,
        dataset_purpose=plan.dataset_purpose,
        dataset_seal_sha256=design.dataset_seal_sha256,
    )


def _validate_plan_manifest(plan: ExperimentPlan, manifest: RunManifest) -> None:
    expected_policy_ids = {turn.condition.policy_id for turn in plan.turns}
    expected_model_seeds = tuple(sorted({turn.condition.model_seed for turn in plan.turns}))
    expected_controller_seeds = tuple(
        sorted({turn.condition.controller_seed for turn in plan.turns})
    )
    if (
        manifest.experiment_config_hash != plan.experiment_config_hash
        or manifest.dataset_hash != plan.dataset_hash
        or manifest.provider_identity != plan.provider_identity
        or manifest.provider_effective_configuration_json
        != plan.provider_effective_configuration_json
        or manifest.action_bounds != plan.action_bounds
        or manifest.decoding_bounds != plan.decoding_bounds
        or dict(manifest.metric_versions) != dict(plan.metric_versions)
        or manifest.decision_rule_version != PHASE3_DECISION_RULE_VERSION
        or manifest.database_schema_version != plan.database_schema_version
        or set(manifest.policy_config_hashes) != expected_policy_ids
        or bool(manifest.matched_history_policy_sources)
        or manifest.seed_schedule.model_seeds != expected_model_seeds
        or manifest.seed_schedule.controller_seeds != expected_controller_seeds
        or manifest.phase3_analysis_contract_sha256 != build_phase3_analysis_contract_sha256(plan)
    ):
        raise StoreInvariantError("closed run manifest does not exactly match the Phase 3 plan")


def evaluation_records_from_store(
    plan: ExperimentPlan,
    store: SQLiteRunStore,
) -> tuple[TurnEvaluationRecord, ...]:
    """Reconstruct typed evaluator records from committed store evidence only."""

    if not isinstance(plan, ExperimentPlan):
        raise TypeError("plan must be an ExperimentPlan")
    if not isinstance(store, SQLiteRunStore):
        raise TypeError("store must be a SQLiteRunStore")
    if plan.evaluation is None:
        raise ValueError("evaluation records require a Phase 3 plan")
    stored_turns = {turn.condition_id: turn for turn in store.list_turns()}
    planned_ids = {turn.condition.condition_id for turn in plan.turns}
    if set(stored_turns) != planned_ids:
        raise StoreInvariantError("closed run conditions do not exactly match the Phase 3 plan")
    input_evidence = {item.condition_id: item for item in store.list_turn_inputs()}
    if set(input_evidence) != planned_ids:
        raise StoreInvariantError("Phase 3 turns lack exact prompt-side input evidence")

    records: list[TurnEvaluationRecord] = []
    for planned in plan.turns:
        condition_id = planned.condition.condition_id
        turn = stored_turns[condition_id]
        evidence = input_evidence[condition_id]
        if turn.state is not TurnState.COMMITTED or turn.metrics is None:
            raise StoreInvariantError("Phase 3 evaluation requires committed metric evidence")
        if turn.condition != planned.condition or turn.request.prompt != planned.prompt:
            raise StoreInvariantError("stored Phase 3 turn differs from its frozen plan")
        if (
            evidence.prompt_case_id != planned.prompt_case_id
            or evidence.prompt_family != planned.prompt_family
            or evidence.prompt_features != planned.prompt_features
            or evidence.validator != planned.validator
        ):
            raise StoreInvariantError("stored prompt-side evidence differs from its frozen plan")
        (
            action_magnitude,
            action_within_bounds,
            action_saturated,
            observation_has_previous_response,
        ) = _trace_evidence(turn, plan.action_bounds)
        stored_history_present = turn.history is not None
        previous_commitment = (
            None
            if not observation_has_previous_response or turn.history is None
            else turn.history.previous_history_commitment_sha256
        )
        if stored_history_present != (turn.condition.turn_index > 0):
            raise StoreInvariantError("stored history presence disagrees with the logical turn")
        metrics = turn.metrics
        records.append(
            TurnEvaluationRecord(
                dataset_sha256=plan.dataset_hash,
                prompt_sequence_id=turn.condition.prompt_sequence_id,
                turn_index=turn.condition.turn_index,
                policy_id=turn.condition.policy_id,
                model_seed=turn.condition.model_seed,
                controller_seed=turn.condition.controller_seed,
                provider_identity_id=turn.condition.provider_identity_id,
                has_previous_response=observation_has_previous_response,
                previous_history_commitment_sha256=previous_commitment,
                task_score=metrics.task_score.value,
                instruction_adherence=metrics.instruction_adherence.value,
                response_length_tokens=metrics.response_length_tokens.value,
                repetition_ratio=metrics.repetition_ratio.value,
                action_magnitude=action_magnitude,
                action_within_bounds=action_within_bounds,
                action_saturated=action_saturated,
            )
        )
    return tuple(records)


def analyze_closed_run(
    plan: ExperimentPlan,
    static_selection_record: StaticSelectionRecord,
    database_path: Path,
) -> StoredAnalysis:
    """Evaluate, persist, verify, and return one closed Phase 3 run analysis."""

    if not isinstance(plan, ExperimentPlan):
        raise TypeError("plan must be an ExperimentPlan")
    if not isinstance(static_selection_record, StaticSelectionRecord):
        raise TypeError("static_selection_record must be a StaticSelectionRecord")
    if not isinstance(database_path, Path):
        raise TypeError("database_path must be a pathlib.Path")
    if plan.evaluation is None or plan.evaluation_spec_sha256 is None:
        raise ValueError("analysis requires a Phase 3 EvaluationSpec")
    if (
        plan.static_selection_record is None
        or plan.static_selection_result_sha256 is None
        or static_selection_record != plan.static_selection_record
        or static_selection_record.selection_result_sha256 != plan.static_selection_result_sha256
    ):
        raise ValueError("analysis static-selection evidence does not match the frozen plan")
    design = build_evaluation_design(plan)
    with SQLiteRunStore(database_path) as store:
        store.verify_integrity()
        run_manifest = store.get_manifest()
        run_finalization = store.get_finalization()
        if run_manifest is None or run_finalization is None:
            raise StoreInvariantError("analysis requires a manifest-bound finalized run")
        _validate_plan_manifest(plan, run_manifest)
        expected_condition_ids = tuple(sorted(turn.condition.condition_id for turn in plan.turns))
        if run_finalization.expected_condition_ids != expected_condition_ids:
            raise StoreInvariantError("run finalization does not close the Phase 3 plan")
        records = evaluation_records_from_store(plan, store)
        result: Phase3EvaluationResult = evaluate_phase3(
            records,
            design=design,
            spec=plan.evaluation,
        )
        analysis_manifest = AnalysisManifest(
            run_manifest_sha256=canonical_sha256(run_manifest),
            run_finalization_sha256=canonical_sha256(run_finalization),
            scientific_result_sha256=run_finalization.scientific_result_sha256,
            experiment_plan_sha256=plan.scientific_identity_sha256,
            evaluation_spec=plan.evaluation,
            evaluation_spec_sha256=plan.evaluation_spec_sha256,
            static_selection_record=static_selection_record,
            static_selection_result_sha256=(static_selection_record.selection_result_sha256),
            evaluation_design=design,
            dataset_sha256=plan.dataset_hash,
            dataset_purpose=design.dataset_purpose,
            dataset_seal_sha256=design.dataset_seal_sha256,
            evaluation_input_sha256=result.input_sha256,
        )
        store.persist_analysis(analysis_manifest, result)
        stored = store.get_analysis()
        if stored is None:
            raise StoreInvariantError("analysis finalization was not durably persisted")
        store.verify_integrity()
        store.compact()
    return stored


__all__ = [
    "analyze_closed_run",
    "build_evaluation_design",
    "build_phase3_analysis_contract_sha256",
    "evaluation_records_from_store",
]

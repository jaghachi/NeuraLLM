"""Export compact deterministic views of canonical Phase 2 through Phase 5 stores."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from neurallm.control.action_space import apply_action, normalized_action_magnitude
from neurallm.control.neural import ActionDecoder, NeuralSubstrate, ObservationEncoder
from neurallm.domain.models import (
    ControllerAction,
    ControllerObservation,
    MetricValue,
    ResponseMetrics,
    RunManifest,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.scientific import NegativeSideEvidence, ScientificGuardrailResult
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    RunFinalization,
    SQLiteRunStore,
    StoredAnalysis,
    StoredScientificAnalysis,
    StoredTurn,
    TurnInputEvidence,
    TurnState,
    scientific_result_sha256,
)

CLOSED_RUN_ARTIFACTS = frozenset(
    {
        "run.sqlite3",
        "manifest.json",
        "results.csv",
        "comparisons.csv",
        "decision.json",
        "report.md",
    }
)
SQLITE_RECOVERY_SIDECARS = frozenset({"run.sqlite3-journal", "run.sqlite3-shm", "run.sqlite3-wal"})

_RESULT_FIELDS = (
    "condition_id",
    "request_sha256",
    "history_commitment_sha256",
    "experiment_id",
    "dataset_version",
    "prompt_sequence_id",
    "turn_index",
    "policy_id",
    "model_seed",
    "controller_seed",
    "provider_identity_id",
    "base_decoding_profile_id",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "max_tokens",
    "response_text",
    "task_score",
    "instruction_adherence",
    "response_length_tokens",
    "repetition_ratio",
    "repeated_3_gram_ratio",
    "repeated_4_gram_ratio",
    "distinct_2",
    "distinct_3",
    "late_window_repetition_ratio",
    "format_validity",
    "semantic_similarity",
    "semantic_similarity_available",
)

_PHASE2_COMPARISON_FIELDS = (
    "comparison_id",
    "focal_policy_id",
    "comparator_policy_id",
    "estimate",
    "status",
)

_PHASE3_COMPARISON_FIELDS = (
    "comparison_id",
    "focal_policy_id",
    "comparator_policy_id",
    "serious_comparator",
    "unit_count",
    "estimate",
    "bootstrap_lower",
    "bootstrap_upper",
    "bootstrap_resamples",
    "bootstrap_seed",
    "permutation_p_value",
    "permutation_exact",
    "permutation_count",
    "permutation_seed",
    "holm_adjusted_p_value",
    "behavioral_alias",
    "guardrail_statuses",
    "status",
)

_PHASE5_COMPARISON_FIELDS = (
    "comparison_id",
    "comparison_kind",
    "focal_policy_id",
    "comparator_policy_id",
    "comparator_role",
    "attribution_only",
    "included_in_efficacy",
    "included_in_holm_family",
    "primary_metric",
    "unit_count",
    "mean_difference",
    "bootstrap_lower",
    "bootstrap_upper",
    "bootstrap_resamples",
    "bootstrap_seed",
    "negative_multiplicity_method",
    "negative_familywise_alpha",
    "negative_family_size",
    "negative_confidence_level",
    "negative_bootstrap_lower",
    "negative_bootstrap_upper",
    "negative_bootstrap_resamples",
    "negative_bootstrap_seed",
    "negative_decisive",
    "permutation_p_value",
    "permutation_exact",
    "permutation_count",
    "permutation_seed",
    "holm_adjusted_p_value",
    "practical_effect_threshold",
    "behavioral_alias",
    "guardrail_statuses",
    "status",
    "detail",
)

_PHASE2_DECISION_RULE_VERSION = "phase2-no-scientific-decision-v1"
_PHASE3_DECISION_RULE_VERSION = "phase3-baseline-evaluator-v1"
_PHASE4_DECISION_RULE_VERSION = "phase4-neural-mechanism-only-v1"
_CONFIRMATORY_DECISION_RULE_VERSION = "confirmatory-scientific-decision-v2"
_MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS = {
    "engineering-smoke-no-scientific-decision-v1": "engineering_smoke",
    "development-pilot-no-scientific-decision-v1": "development_pilot",
}
_MODEL_BACKED_RULE_TIERS = {
    **_MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS,
    _CONFIRMATORY_DECISION_RULE_VERSION: "confirmatory",
}
_MODEL_BACKED_POLICY_IDS = {
    "best_static",
    "heuristic_adaptive",
    "neural_matched_history_state_reset",
    "neural_persistent",
    "random_matched",
}
_PHASE4_MATCHED_HISTORY_POLICY_SOURCES = {
    "neural_matched_history_state_reset": "neural_persistent",
}


@dataclass(frozen=True, slots=True)
class ArtifactExportSummary:
    """Stable identities and counts for one completed export."""

    output_directory: Path
    manifest_sha256: str
    scientific_result_sha256: str
    committed_turns: int
    artifact_names: tuple[str, ...]
    implementation_phase: int
    phase3_baseline_evaluator_verdict: str | None
    scientific_decision: str | None = None


def _metric_value(metric: MetricValue[int] | MetricValue[float]) -> int | float | str:
    return "" if metric.value is None else metric.value


def _result_row(turn: StoredTurn) -> dict[str, object]:
    if turn.response is None or turn.metrics is None or turn.history_commitment_sha256 is None:
        raise ValueError("committed turn is missing response, metric, or history evidence")
    condition = turn.condition
    parameters = turn.request.decoding_parameters
    metrics: ResponseMetrics = turn.metrics
    return {
        "condition_id": turn.condition_id,
        "request_sha256": turn.request_sha256,
        "history_commitment_sha256": turn.history_commitment_sha256,
        "experiment_id": condition.experiment_id,
        "dataset_version": condition.dataset_version,
        "prompt_sequence_id": condition.prompt_sequence_id,
        "turn_index": condition.turn_index,
        "policy_id": condition.policy_id,
        "model_seed": condition.model_seed,
        "controller_seed": condition.controller_seed,
        "provider_identity_id": condition.provider_identity_id,
        "base_decoding_profile_id": condition.base_decoding_profile_id,
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
        "top_k": parameters.top_k,
        "presence_penalty": parameters.presence_penalty,
        "max_tokens": parameters.max_tokens,
        "response_text": turn.response.text,
        "task_score": _metric_value(metrics.task_score),
        "instruction_adherence": _metric_value(metrics.instruction_adherence),
        "response_length_tokens": _metric_value(metrics.response_length_tokens),
        "repetition_ratio": _metric_value(metrics.repetition_ratio),
        "repeated_3_gram_ratio": _metric_value(metrics.repeated_3_gram_ratio),
        "repeated_4_gram_ratio": _metric_value(metrics.repeated_4_gram_ratio),
        "distinct_2": _metric_value(metrics.distinct_2),
        "distinct_3": _metric_value(metrics.distinct_3),
        "late_window_repetition_ratio": _metric_value(metrics.late_window_repetition_ratio),
        "format_validity": _metric_value(metrics.format_validity),
        "semantic_similarity": _metric_value(metrics.semantic_similarity),
        "semantic_similarity_available": metrics.semantic_similarity.availability,
    }


def _csv_text(fieldnames: tuple[str, ...], rows: tuple[dict[str, object], ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_atomic(path: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _reject_unexpected_files(
    output_directory: Path,
    *,
    allow_sqlite_recovery_sidecars: bool = False,
) -> None:
    allowed = set(CLOSED_RUN_ARTIFACTS)
    if allow_sqlite_recovery_sidecars:
        allowed.update(SQLITE_RECOVERY_SIDECARS)
    unexpected = sorted(
        item.name for item in output_directory.iterdir() if item.name not in allowed
    )
    if unexpected:
        raise ValueError(f"run directory contains unexpected artifacts: {unexpected!r}")


def _decision_payload(manifest: RunManifest, turns: tuple[StoredTurn, ...]) -> dict[str, object]:
    result_sha256 = scientific_result_sha256(turns)
    return {
        "schema_version": 1,
        "implementation_phase": 2,
        "claim_scope": "engineering_validation_only",
        "scientific_decision": None,
        "comparison_status": "not_available_until_phase_3",
        "decision_rule_version": manifest.decision_rule_version,
        "manifest_sha256": canonical_sha256(manifest),
        "scientific_result_sha256": result_sha256,
        "provider_type": manifest.provider_identity.provider_type,
        "committed_turns": len(turns),
        "database_integrity_verified": True,
        "rationale": (
            "Phase 2 validates the provider-to-artifact engineering path only; "
            "it does not estimate policy efficacy or select a scientific outcome."
        ),
    }


def _phase4_decision_payload(
    manifest: RunManifest,
    turns: tuple[StoredTurn, ...],
) -> dict[str, object]:
    """Return a mechanism-only Phase 4 result without an efficacy verdict."""

    return {
        "schema_version": 1,
        "implementation_phase": 4,
        "claim_scope": "deterministic_mechanism_validation_only",
        "scientific_decision": None,
        "comparison_status": "matched_history_attribution_mechanism_only",
        "decision_rule_version": manifest.decision_rule_version,
        "manifest_sha256": canonical_sha256(manifest),
        "scientific_result_sha256": scientific_result_sha256(turns),
        "provider_type": manifest.provider_identity.provider_type,
        "committed_turns": len(turns),
        "matched_history_policy_sources": dict(manifest.matched_history_policy_sources),
        "database_integrity_verified": True,
        "rationale": (
            "Phase 4 records deterministic controller activity and matched-history "
            "substrate-reset isolation only; it does not estimate neural efficacy, a "
            "beneficial model-backed persistent-state effect, or a scientific outcome."
        ),
    }


def _model_backed_nonscientific_decision_payload(
    manifest: RunManifest,
    finalization: RunFinalization,
    turns: tuple[StoredTurn, ...],
) -> dict[str, object]:
    """Return an explicitly non-scientific smoke or pilot closeout."""

    tier = _MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS[manifest.decision_rule_version]
    accounting = finalization.execution_accounting
    if accounting is None:
        raise ValueError("model-backed closeout lacks durable execution accounting")
    return {
        "schema_version": 1,
        "implementation_phase": 5,
        "run_tier": tier,
        "claim_scope": (
            "engineering_validation_only"
            if tier == "engineering_smoke"
            else "development_calibration_only"
        ),
        "scientific_decision": None,
        "comparison_status": "not_eligible_for_confirmatory_inference",
        "decision_rule_version": manifest.decision_rule_version,
        "manifest_sha256": canonical_sha256(manifest),
        "scientific_result_sha256": scientific_result_sha256(turns),
        "scientific_identity_sha256": manifest.scientific_identity_sha256,
        "provider_type": manifest.provider_identity.provider_type,
        "committed_turns": len(turns),
        "matched_history_policy_sources": dict(manifest.matched_history_policy_sources),
        "execution_accounting": accounting.model_dump(mode="json"),
        "database_integrity_verified": True,
        "rationale": (
            "This tier verifies the complete five-arm engineering and causal-evidence path. "
            "It is not eligible to emit a confirmatory scientific decision."
        ),
    }


def _phase4_trajectory_key(turn: StoredTurn, policy_id: str) -> tuple[object, ...]:
    condition = turn.condition
    return (
        condition.experiment_id,
        condition.dataset_version,
        condition.prompt_sequence_id,
        policy_id,
        condition.model_seed,
        condition.controller_seed,
        condition.provider_identity_id,
        condition.base_decoding_profile_id,
        condition.turn_index,
    )


def _phase4_base_trajectory_key(turn: StoredTurn) -> tuple[object, ...]:
    condition = turn.condition
    return (
        condition.experiment_id,
        condition.dataset_version,
        condition.prompt_sequence_id,
        condition.model_seed,
        condition.controller_seed,
        condition.provider_identity_id,
        condition.base_decoding_profile_id,
    )


def _validate_phase4_mechanism_evidence(
    manifest: RunManifest,
    turns: tuple[StoredTurn, ...],
    turn_inputs: tuple[TurnInputEvidence, ...],
) -> None:
    """Fail closed unless a Phase 4 store contains complete causal neural evidence."""

    from neurallm.control.neural import NeuralPolicyState
    from neurallm.experiments.runner import CausalAppliedPolicyTrace

    persistent_id = "neural_persistent"
    reset_id = "neural_matched_history_state_reset"
    neural_policy_ids = {persistent_id, reset_id}
    expected_policy_ids = (
        _MODEL_BACKED_POLICY_IDS
        if manifest.decision_rule_version in _MODEL_BACKED_RULE_TIERS
        else neural_policy_ids
    )
    if manifest.database_schema_version != CURRENT_SCHEMA_VERSION:
        raise ValueError("Phase 4 export requires the current database schema")
    if set(manifest.policy_config_hashes) != expected_policy_ids:
        raise ValueError("Phase 4 export requires exactly the persistent and reset policies")
    if {turn.condition.policy_id for turn in turns} != expected_policy_ids:
        raise ValueError("Phase 4 store must contain exactly the two neural policies")

    input_by_condition_id = {evidence.condition_id: evidence for evidence in turn_inputs}
    turn_condition_ids = {turn.condition_id for turn in turns}
    if (
        len(input_by_condition_id) != len(turn_inputs)
        or set(input_by_condition_id) != turn_condition_ids
    ):
        raise ValueError("Phase 4 export requires exact prompt-side evidence coverage")

    neural_turns = tuple(turn for turn in turns if turn.condition.policy_id in neural_policy_ids)
    by_condition_id = {turn.condition_id: turn for turn in turns}
    by_coordinate = {_phase4_trajectory_key(turn, turn.condition.policy_id): turn for turn in turns}
    base_parameters_by_trajectory = {
        _phase4_base_trajectory_key(turn): turn.request.decoding_parameters
        for turn in turns
        if turn.condition.policy_id == persistent_id and turn.condition.turn_index == 0
    }
    traces: dict[str, CausalAppliedPolicyTrace] = {}
    later_reset_count = 0
    any_later_activity = False
    for turn in neural_turns:
        if (
            turn.state is not TurnState.COMMITTED
            or turn.response is None
            or turn.metrics is None
            or turn.policy_state_json is None
            or turn.policy_trace_json is None
            or turn.history_commitment_sha256 is None
        ):
            raise ValueError("Phase 4 turn lacks complete committed mechanism evidence")
        trace = CausalAppliedPolicyTrace.model_validate_json(turn.policy_trace_json)
        traces[turn.condition_id] = trace
        input_evidence = input_by_condition_id[turn.condition_id]
        if (
            trace.policy_id != turn.condition.policy_id
            or trace.turn_index != turn.condition.turn_index
        ):
            raise ValueError("Phase 4 trace identity does not match its stored condition")
        if (
            trace.policy_trace.policy_id != turn.condition.policy_id
            or trace.policy_trace.turn_index != turn.condition.turn_index
        ):
            raise ValueError("Phase 4 nested neural trace has the wrong identity")
        if trace.action_application.final_decoding_parameters != turn.request.decoding_parameters:
            raise ValueError("Phase 4 trace decoding parameters do not match the stored request")
        if trace.action != trace.action_application.step_clamped_action:
            raise ValueError("Phase 4 applied trace does not bind its step-clamped action")
        if trace.policy_trace.action != trace.action_application.raw_action:
            raise ValueError("Phase 4 neural trace does not bind its raw controller action")
        expected_access = (
            "matched_focal_previous_response"
            if turn.condition.policy_id == reset_id
            else "own_previous_response"
        )
        if trace.history_access != expected_access:
            raise ValueError("Phase 4 trace declares the wrong history-access mode")
        state = NeuralPolicyState.model_validate_json(turn.policy_state_json)
        if state.next_turn_index != turn.condition.turn_index + 1:
            raise ValueError("Phase 4 stored neural state has the wrong logical turn")
        if state.controller_seed != turn.condition.controller_seed:
            raise ValueError("Phase 4 stored neural state has the wrong controller seed")
        if state.action_bounds != manifest.action_bounds:
            raise ValueError("Phase 4 stored neural state has the wrong action bounds")
        if state.substrate != trace.policy_trace.substrate_transition.state_after:
            raise ValueError("Phase 4 stored neural state does not match the traced transition")
        expected_transition = NeuralSubstrate().step(
            trace.policy_trace.effective_substrate_state,
            trace.policy_trace.observation_encoding,
            state.controller_seed,
        )
        if trace.policy_trace.substrate_transition != expected_transition:
            raise ValueError("Phase 4 trace does not reproduce the neural substrate equations")
        expected_magnitude = normalized_action_magnitude(
            trace.policy_trace.action,
            manifest.action_bounds,
        )
        if trace.policy_trace.action_magnitude != expected_magnitude:
            raise ValueError("Phase 4 trace reports the wrong normalized action magnitude")
        decoded = ActionDecoder().decode(
            trace.policy_trace.substrate_transition.state_after,
            manifest.action_bounds,
            action_enabled=turn.condition.turn_index > 0,
        )
        if (
            trace.policy_trace.decoder_activation != decoded.activation
            or trace.policy_trace.action != decoded.action
            or trace.policy_trace.action_magnitude != decoded.action_magnitude
        ):
            raise ValueError("Phase 4 trace does not reproduce the declared action decoder")
        base_parameters = base_parameters_by_trajectory.get(_phase4_base_trajectory_key(turn))
        if base_parameters is None:
            raise ValueError("Phase 4 trajectory lacks its persistent turn-zero base")
        expected_application = apply_action(
            base_parameters,
            decoded.action,
            manifest.action_bounds,
            manifest.decoding_bounds,
        )
        if trace.action_application != expected_application:
            raise ValueError("Phase 4 trace does not reproduce the declared action application")
        if turn.condition.turn_index == 0:
            if turn.history is not None or trace.observation_has_previous_response:
                raise ValueError("Phase 4 turn zero must carry explicit null history")
            expected_initial_state = NeuralSubstrate().initial_state(turn.condition.controller_seed)
            if (
                trace.policy_trace.state_reset_applied
                or trace.policy_trace.stored_substrate_state != expected_initial_state
                or trace.policy_trace.effective_substrate_state
                != trace.policy_trace.stored_substrate_state
            ):
                raise ValueError("Phase 4 turn zero does not use the declared initial state")
            expected_encoding = ObservationEncoder().encode(
                ControllerObservation(
                    turn_index=0,
                    prompt_family=input_evidence.prompt_family,
                    current_prompt_features=input_evidence.prompt_features,
                    previous_response_metrics=None,
                    has_previous_response=False,
                )
            )
            if trace.policy_trace.observation_encoding != expected_encoding:
                raise ValueError("Phase 4 trace does not match its prompt-side evidence")
            continue

        any_later_activity |= trace.policy_trace.action_magnitude > 0.0
        if turn.condition.policy_id == reset_id:
            later_reset_count += 1
        if turn.history is None:
            raise ValueError("Phase 4 later turn lacks its focal history binding")
        previous = by_condition_id.get(turn.history.previous_condition_id)
        if (
            previous is None
            or previous.metrics is None
            or previous.policy_state_json is None
            or previous.history_commitment_sha256 is None
        ):
            raise ValueError("Phase 4 history source lacks complete committed evidence")
        if (
            previous.condition.policy_id != persistent_id
            or previous.condition.turn_index != turn.condition.turn_index - 1
        ):
            raise ValueError("Phase 4 history does not bind the focal prior turn")
        if (
            trace.history_source_policy_id != persistent_id
            or trace.history_source_condition_id != previous.condition_id
            or trace.history_commitment_sha256 != previous.history_commitment_sha256
            or trace.observation_metrics_sha256 != canonical_sha256(previous.metrics)
        ):
            raise ValueError("Phase 4 causal trace does not match its committed focal history")
        previous_state = NeuralPolicyState.model_validate_json(previous.policy_state_json)
        if trace.policy_trace.stored_substrate_state != previous_state.substrate:
            raise ValueError("Phase 4 trace did not load the committed focal substrate")
        expected_encoding = ObservationEncoder().encode(
            ControllerObservation(
                turn_index=turn.condition.turn_index,
                prompt_family=input_evidence.prompt_family,
                current_prompt_features=input_evidence.prompt_features,
                previous_response_metrics=previous.metrics,
                has_previous_response=True,
            )
        )
        if trace.policy_trace.observation_encoding != expected_encoding:
            raise ValueError("Phase 4 trace does not match its prompt-side evidence")

    if later_reset_count == 0 or not any_later_activity:
        raise ValueError("Phase 4 export requires later reset evidence and controller activity")

    persistent_coordinates = {
        _phase4_trajectory_key(turn, persistent_id)
        for turn in turns
        if turn.condition.policy_id == persistent_id
    }
    reset_coordinates = {
        _phase4_trajectory_key(turn, persistent_id)
        for turn in turns
        if turn.condition.policy_id == reset_id
    }
    if persistent_coordinates != reset_coordinates:
        raise ValueError("Phase 4 attribution arms lack exact paired coverage")

    substrate = NeuralSubstrate()
    any_later_mechanism_divergence = False
    for reset_turn in (turn for turn in turns if turn.condition.policy_id == reset_id):
        persistent_turn = by_coordinate.get(_phase4_trajectory_key(reset_turn, persistent_id))
        if persistent_turn is None:
            raise ValueError("Phase 4 reset turn lacks its paired focal current turn")
        persistent_response = persistent_turn.response
        reset_response = reset_turn.response
        if persistent_response is None or reset_response is None:
            raise ValueError("Phase 4 attribution pair lacks committed responses")
        persistent_trace = traces[persistent_turn.condition_id]
        reset_trace = traces[reset_turn.condition_id]
        if persistent_turn.request.prompt != reset_turn.request.prompt:
            raise ValueError("Phase 4 attribution pair has mismatched current prompts")
        persistent_input = input_by_condition_id[persistent_turn.condition_id]
        reset_input = input_by_condition_id[reset_turn.condition_id]
        if (
            persistent_input.prompt_case_id != reset_input.prompt_case_id
            or persistent_input.prompt_family != reset_input.prompt_family
            or persistent_input.prompt_features != reset_input.prompt_features
            or persistent_input.validator != reset_input.validator
        ):
            raise ValueError("Phase 4 attribution pair has mismatched prompt-side evidence")
        if reset_turn.condition.turn_index == 0:
            if (
                persistent_turn.request.decoding_parameters
                != reset_turn.request.decoding_parameters
                or persistent_response.text != reset_response.text
                or persistent_response.effective_parameters != reset_response.effective_parameters
                or persistent_response.provider_identity != reset_response.provider_identity
                or persistent_turn.policy_state_json != reset_turn.policy_state_json
                or persistent_trace.policy_trace.mechanism_sha256
                != reset_trace.policy_trace.mechanism_sha256
            ):
                raise ValueError("Phase 4 attribution arms are not equivalent at turn zero")
            continue
        if (
            persistent_turn.history != reset_turn.history
            or persistent_trace.policy_trace.observation_encoding
            != reset_trace.policy_trace.observation_encoding
            or persistent_trace.policy_trace.stored_substrate_state
            != reset_trace.policy_trace.stored_substrate_state
        ):
            raise ValueError("Phase 4 attribution arms do not share exact focal inputs")
        if (
            persistent_trace.policy_trace.state_reset_applied
            or persistent_trace.policy_trace.effective_substrate_state
            != persistent_trace.policy_trace.stored_substrate_state
        ):
            raise ValueError("Phase 4 persistent arm contains an undeclared intervention")
        if (
            not reset_trace.policy_trace.state_reset_applied
            or reset_trace.policy_trace.effective_substrate_state
            != substrate.initial_state(reset_turn.condition.controller_seed)
        ):
            raise ValueError("Phase 4 reset arm did not apply the declared substrate reset")
        any_later_mechanism_divergence |= (
            persistent_trace.policy_trace.effective_substrate_state
            != reset_trace.policy_trace.effective_substrate_state
            or persistent_trace.policy_trace.action != reset_trace.policy_trace.action
            or persistent_turn.request.decoding_parameters != reset_turn.request.decoding_parameters
        )

    if not any_later_mechanism_divergence:
        raise ValueError("Phase 4 export requires a later paired mechanism divergence")


def _phase3_comparison_rows(analysis: StoredAnalysis) -> tuple[dict[str, object], ...]:
    focal_policy_id = analysis.manifest.evaluation_spec.focal_policy_id
    return tuple(
        {
            "comparison_id": canonical_sha256(comparison),
            "focal_policy_id": focal_policy_id,
            "comparator_policy_id": comparison.comparator_policy_id,
            "serious_comparator": comparison.serious_comparator,
            "unit_count": comparison.unit_count,
            "estimate": comparison.mean_difference,
            "bootstrap_lower": comparison.bootstrap.lower,
            "bootstrap_upper": comparison.bootstrap.upper,
            "bootstrap_resamples": comparison.bootstrap.resamples,
            "bootstrap_seed": comparison.bootstrap.seed,
            "permutation_p_value": comparison.permutation.p_value,
            "permutation_exact": comparison.permutation.exact,
            "permutation_count": comparison.permutation.performed_permutations,
            "permutation_seed": comparison.permutation.seed,
            "holm_adjusted_p_value": (
                "" if comparison.holm is None else comparison.holm.adjusted_p_value
            ),
            "behavioral_alias": comparison.behavioral_alias,
            "guardrail_statuses": ";".join(
                f"{guardrail.name.value}={guardrail.status.value}"
                for guardrail in comparison.guardrails
            ),
            "status": comparison.verdict.value,
        }
        for comparison in analysis.result.comparisons
    )


def _phase3_decision_payload(
    manifest: RunManifest,
    turns: tuple[StoredTurn, ...],
    analysis: StoredAnalysis,
) -> dict[str, object]:
    result = analysis.result
    return {
        "schema_version": 1,
        "implementation_phase": 3,
        "claim_scope": result.claim_scope,
        "scientific_decision": None,
        "phase3_baseline_evaluator_verdict": result.verdict.value,
        "comparison_status": "available" if result.comparisons else "invalid_or_unavailable",
        "decision_rule_version": manifest.decision_rule_version,
        "manifest_sha256": canonical_sha256(manifest),
        "scientific_result_sha256": scientific_result_sha256(turns),
        "analysis_manifest_sha256": canonical_sha256(analysis.manifest),
        "analysis_finalization_sha256": canonical_sha256(analysis.finalization),
        "evaluation_input_sha256": result.input_sha256,
        "evaluation_result_sha256": result.result_sha256,
        "evaluation_spec": analysis.manifest.evaluation_spec.model_dump(mode="json"),
        "evaluation_spec_sha256": analysis.manifest.evaluation_spec_sha256,
        "action_magnitude_version": analysis.manifest.action_magnitude_version,
        "static_selection_result_sha256": (analysis.manifest.static_selection_result_sha256),
        "dataset_purpose": analysis.manifest.dataset_purpose.value,
        "dataset_seal_sha256": analysis.manifest.dataset_seal_sha256,
        "provider_type": manifest.provider_identity.provider_type,
        "committed_turns": len(turns),
        "comparison_count": len(result.comparisons),
        "coverage": result.coverage.model_dump(mode="json"),
        "global_guardrails": tuple(
            guardrail.model_dump(mode="json") for guardrail in result.global_guardrails
        ),
        "statistics_computed": result.statistics_computed,
        "statistics_call_count": result.statistics_call_count,
        "database_integrity_verified": True,
        "rationale": (
            "This is a Phase 3 baseline-evaluator result within the declared synthetic or "
            "sealed-data protocol. It validates comparison behavior only and does not make "
            "a Phase 5 end-to-end efficacy or persistent-state attribution decision."
        ),
    }


def _scientific_guardrail_statuses(
    guardrails: tuple[ScientificGuardrailResult, ...],
) -> str:
    """Render typed scientific guardrails in stable scope/name order."""

    ordered = sorted(
        guardrails,
        key=lambda guardrail: (
            guardrail.scope,
            guardrail.name,
        ),
    )
    return ";".join(
        f"{guardrail.scope}:{guardrail.name}={guardrail.status.value}" for guardrail in ordered
    )


def _phase5_comparison_rows(
    analysis: StoredScientificAnalysis,
) -> tuple[dict[str, object], ...]:
    """Return the exact three efficacy rows plus one attribution-only row."""

    rows: list[dict[str, object]] = []
    for comparison in analysis.result.efficacy_comparisons:
        bootstrap = comparison.bootstrap
        negative = comparison.negative_side_evidence
        permutation = comparison.permutation
        rows.append(
            {
                "comparison_id": canonical_sha256(comparison),
                "comparison_kind": comparison.comparison_kind,
                "focal_policy_id": comparison.focal_policy_id,
                "comparator_policy_id": comparison.comparator_policy_id,
                "comparator_role": comparison.comparator_role.value,
                "attribution_only": False,
                "included_in_efficacy": True,
                "included_in_holm_family": comparison.included_in_holm_family,
                "primary_metric": comparison.primary_metric,
                "unit_count": comparison.unit_count,
                "mean_difference": (
                    "" if comparison.mean_difference is None else comparison.mean_difference
                ),
                "bootstrap_lower": "" if bootstrap is None else bootstrap.lower,
                "bootstrap_upper": "" if bootstrap is None else bootstrap.upper,
                "bootstrap_resamples": "" if bootstrap is None else bootstrap.resamples,
                "bootstrap_seed": "" if bootstrap is None else bootstrap.seed,
                **_negative_side_fields(negative),
                "permutation_p_value": "" if permutation is None else permutation.p_value,
                "permutation_exact": "" if permutation is None else permutation.exact,
                "permutation_count": (
                    "" if permutation is None else permutation.performed_permutations
                ),
                "permutation_seed": "" if permutation is None else permutation.seed,
                "holm_adjusted_p_value": (
                    "" if comparison.holm is None else comparison.holm.adjusted_p_value
                ),
                "practical_effect_threshold": comparison.practical_effect_threshold,
                "behavioral_alias": comparison.behavioral_alias,
                "guardrail_statuses": _scientific_guardrail_statuses(tuple(comparison.guardrails)),
                "status": comparison.status.value,
                "detail": comparison.detail,
            }
        )

    attribution = analysis.result.attribution
    bootstrap = attribution.bootstrap
    negative = attribution.negative_side_evidence
    permutation = attribution.permutation
    rows.append(
        {
            "comparison_id": canonical_sha256(attribution),
            "comparison_kind": attribution.comparison_kind,
            "focal_policy_id": attribution.focal_policy_id,
            "comparator_policy_id": attribution.comparator_policy_id,
            "comparator_role": "attribution_only",
            "attribution_only": attribution.attribution_only,
            "included_in_efficacy": attribution.included_in_efficacy,
            "included_in_holm_family": attribution.included_in_holm_family,
            "primary_metric": attribution.primary_metric,
            "unit_count": attribution.unit_count,
            "mean_difference": (
                "" if attribution.mean_difference is None else attribution.mean_difference
            ),
            "bootstrap_lower": "" if bootstrap is None else bootstrap.lower,
            "bootstrap_upper": "" if bootstrap is None else bootstrap.upper,
            "bootstrap_resamples": "" if bootstrap is None else bootstrap.resamples,
            "bootstrap_seed": "" if bootstrap is None else bootstrap.seed,
            **_negative_side_fields(negative),
            "permutation_p_value": "" if permutation is None else permutation.p_value,
            "permutation_exact": "" if permutation is None else permutation.exact,
            "permutation_count": (
                "" if permutation is None else permutation.performed_permutations
            ),
            "permutation_seed": "" if permutation is None else permutation.seed,
            "holm_adjusted_p_value": "",
            "practical_effect_threshold": attribution.practical_effect_threshold,
            "behavioral_alias": attribution.behavioral_alias,
            "guardrail_statuses": _scientific_guardrail_statuses(
                tuple(attribution.causal_guardrails)
            ),
            "status": attribution.status.value,
            "detail": attribution.detail,
        }
    )
    if len(rows) != 4:
        raise ValueError("confirmatory export requires three efficacy rows plus attribution")
    return tuple(rows)


def _negative_side_fields(evidence: NegativeSideEvidence | None) -> dict[str, object]:
    if evidence is None:
        return {
            "negative_multiplicity_method": "",
            "negative_familywise_alpha": "",
            "negative_family_size": "",
            "negative_confidence_level": "",
            "negative_bootstrap_lower": "",
            "negative_bootstrap_upper": "",
            "negative_bootstrap_resamples": "",
            "negative_bootstrap_seed": "",
            "negative_decisive": "",
        }
    return {
        "negative_multiplicity_method": evidence.method_version,
        "negative_familywise_alpha": evidence.familywise_alpha,
        "negative_family_size": evidence.family_size,
        "negative_confidence_level": evidence.adjusted_two_sided_confidence_level,
        "negative_bootstrap_lower": evidence.bootstrap.lower,
        "negative_bootstrap_upper": evidence.bootstrap.upper,
        "negative_bootstrap_resamples": evidence.bootstrap.resamples,
        "negative_bootstrap_seed": evidence.bootstrap.seed,
        "negative_decisive": evidence.decisive_negative,
    }


def _phase5_decision_payload(
    manifest: RunManifest,
    finalization: RunFinalization,
    turns: tuple[StoredTurn, ...],
    analysis: StoredScientificAnalysis,
) -> dict[str, object]:
    """Return the complete compact confirmatory decision identity and evidence."""

    accounting = finalization.execution_accounting
    if accounting is None:
        raise ValueError("confirmatory closeout lacks durable execution accounting")
    result = analysis.result
    return {
        "schema_version": 2,
        "implementation_phase": 5,
        "run_tier": "confirmatory",
        "claim_scope": result.claim_scope,
        "scientific_decision": result.decision.decision.value,
        "reason_codes": tuple(reason.value for reason in result.decision.reason_codes),
        "decision_rule_version": manifest.decision_rule_version,
        "manifest_sha256": canonical_sha256(manifest),
        "scientific_result_sha256": scientific_result_sha256(turns),
        "analysis_manifest_sha256": canonical_sha256(analysis.manifest),
        "analysis_finalization_sha256": canonical_sha256(analysis.finalization),
        "confirmatory_analysis_contract_sha256": (manifest.confirmatory_analysis_contract_sha256),
        "confirmatory_analysis_spec": result.confirmatory_analysis_spec.model_dump(mode="json"),
        "confirmatory_analysis_spec_sha256": result.confirmatory_analysis_spec_sha256,
        "prompt_family_by_sequence": dict(result.prompt_family_by_sequence),
        "prompt_family_design_sha256": result.prompt_family_design_sha256,
        "validated_negative_multiplicity_sha256": (result.validated_negative_multiplicity_sha256),
        "scientific_identity_sha256": manifest.scientific_identity_sha256,
        "preregistration_sha256": manifest.preregistration_sha256,
        "evaluation_input_sha256": result.input_sha256,
        "evaluation_result_sha256": result.result_sha256,
        "decision_input_sha256": result.decision.decision_input_sha256,
        "provider_type": manifest.provider_identity.provider_type,
        "provider_identity_id": manifest.provider_identity.identity_id,
        "committed_turns": len(turns),
        "execution_accounting": accounting.model_dump(mode="json"),
        "coverage": result.coverage.model_dump(mode="json"),
        "efficacy_comparisons": tuple(
            comparison.model_dump(mode="json") for comparison in result.efficacy_comparisons
        ),
        "recovery": result.recovery.model_dump(mode="json"),
        "persistent_state_attribution": result.attribution.model_dump(mode="json"),
        "subgroup_effects": tuple(
            effect.model_dump(mode="json") for effect in result.subgroup_effects
        ),
        "guardrails": tuple(guardrail.model_dump(mode="json") for guardrail in result.guardrails),
        "limitations": tuple(
            limitation.model_dump(mode="json") for limitation in result.limitations
        ),
        "statistics_call_count": result.statistics_call_count,
        "database_integrity_verified": True,
    }


def _report_text(manifest: RunManifest, turns: tuple[StoredTurn, ...]) -> str:
    return (
        "# NeuraLLM Phase 2 Engineering Report\n\n"
        "This closed run validates the deterministic provider-to-artifact execution path. "
        "It does not establish policy efficacy, comparator advantage, neural activity, or a "
        "scientific decision.\n\n"
        f"- Manifest SHA-256: `{canonical_sha256(manifest)}`\n"
        f"- Scientific result SHA-256: `{scientific_result_sha256(turns)}`\n"
        f"- Provider type: `{manifest.provider_identity.provider_type}`\n"
        f"- Provider identity: `{manifest.provider_identity.identity_id}`\n"
        f"- Committed turns: `{len(turns)}`\n"
        f"- Database schema version: `{manifest.database_schema_version}`\n"
        f"- Decision rule: `{manifest.decision_rule_version}`\n\n"
        "`comparisons.csv` is intentionally empty because serious comparators and statistical "
        "evaluation begin in Phase 3. The canonical response and metric evidence remains in "
        "`run.sqlite3`; the other files are deterministic derived views.\n"
    )


def _phase4_report_text(manifest: RunManifest, turns: tuple[StoredTurn, ...]) -> str:
    source_policy = _PHASE4_MATCHED_HISTORY_POLICY_SOURCES["neural_matched_history_state_reset"]
    provider_phrase = (
        "under the deterministic fake provider"
        if manifest.provider_identity.provider_type == "fake"
        else f"under the declared `{manifest.provider_identity.provider_type}` provider"
    )
    return (
        "# NeuraLLM Phase 4 Deterministic Mechanism Report\n\n"
        "This closed run records neural-controller activity and a causally matched "
        f"substrate-reset intervention {provider_phrase}. It does not establish neural "
        "efficacy, a beneficial model-backed persistent-state effect, or a scientific "
        "decision.\n\n"
        f"- Manifest SHA-256: `{canonical_sha256(manifest)}`\n"
        f"- Scientific result SHA-256: `{scientific_result_sha256(turns)}`\n"
        f"- Provider type: `{manifest.provider_identity.provider_type}`\n"
        f"- Provider identity: `{manifest.provider_identity.identity_id}`\n"
        f"- Committed turns: `{len(turns)}`\n"
        f"- Database schema version: `{manifest.database_schema_version}`\n"
        f"- Decision rule: `{manifest.decision_rule_version}`\n"
        "- Matched-history edge: "
        f"`neural_matched_history_state_reset -> {source_policy}`\n\n"
        "`comparisons.csv` is intentionally header-only because this is a mechanism-level "
        "attribution control, not an efficacy or statistical comparison. Exact requests, "
        "responses, metrics, serialized neural states, causal traces, and commitment hashes "
        "remain in `run.sqlite3`; the other files are deterministic derived views.\n"
    )


def _model_backed_activity_lines(
    manifest: RunManifest,
    turns: tuple[StoredTurn, ...],
) -> str:
    magnitudes: dict[str, list[float]] = {}
    for turn in turns:
        if turn.policy_trace_json is None:
            raise ValueError("model-backed activity evidence lacks a policy trace")
        payload = json.loads(turn.policy_trace_json)
        if not isinstance(payload, dict) or "action" not in payload:
            raise ValueError("model-backed policy trace lacks its applied action")
        action = ControllerAction.model_validate(payload["action"])
        magnitudes.setdefault(turn.condition.policy_id, []).append(
            normalized_action_magnitude(action, manifest.action_bounds)
        )
    return "\n".join(
        f"- `{policy_id}` mean normalized action magnitude: "
        f"`{sum(values) / len(values):.6f}`; nonzero turns: "
        f"`{sum(value > 0.0 for value in values)}/{len(values)}`"
        for policy_id, values in sorted(magnitudes.items())
    )


def _model_backed_nonscientific_report_text(
    manifest: RunManifest,
    finalization: RunFinalization,
    turns: tuple[StoredTurn, ...],
) -> str:
    tier = _MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS[manifest.decision_rule_version]
    accounting = finalization.execution_accounting
    if accounting is None:
        raise ValueError("model-backed closeout lacks durable execution accounting")
    tier_label = "Engineering smoke" if tier == "engineering_smoke" else "Development pilot"
    return (
        f"# NeuraLLM Model-Backed {tier_label} Report\n\n"
        "## Engineering validity\n\n"
        f"- Run tier: `{tier}`\n"
        f"- Manifest SHA-256: `{canonical_sha256(manifest)}`\n"
        f"- Scientific result SHA-256: `{scientific_result_sha256(turns)}`\n"
        f"- Scientific identity SHA-256: `{manifest.scientific_identity_sha256}`\n"
        f"- Provider type: `{manifest.provider_identity.provider_type}`\n"
        f"- Planned logical generations: `{accounting.planned_logical_generations}`\n"
        f"- Dispatched logical generations: `{accounting.dispatched_logical_generations}`\n"
        f"- Successful responses: `{accounting.successful_responses}`\n"
        f"- Uncertain dispatches: `{accounting.uncertain_dispatches}`\n"
        f"- Committed logical generations: `{accounting.committed_logical_generations}`\n\n"
        "## Controller activity\n\n"
        f"{_model_backed_activity_lines(manifest, turns)}\n\n"
        "## End-to-end efficacy\n\n"
        "Not estimated. Engineering smoke and development pilot evidence cannot be promoted "
        "to a confirmatory efficacy claim.\n\n"
        "## Persistent-state attribution\n\n"
        "The matched-history persistent/reset mechanism and exact causal pairing passed "
        "structural reconstruction. A beneficial model-output attribution effect is not "
        "estimated at this tier.\n\n"
        "## Guardrail outcomes\n\n"
        "- Exact five-arm schedule coverage: `pass`\n"
        "- Provider identity stability in committed responses: `pass`\n"
        "- Action-bound and causal-history reconstruction: `pass`\n"
        "- Durable logical-generation accounting: `pass`\n\n"
        "## Limitations\n\n"
        f"This run used provider type `{manifest.provider_identity.provider_type}`. "
        "No smoke or pilot result can select a final scientific state.\n\n"
        "## Final decision\n\n"
        "`scientific_decision` is `null`; this run is intentionally ineligible for the "
        "confirmatory decision vocabulary.\n"
    )


def _phase5_effect_text(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
) -> str:
    if estimate is None or lower is None or upper is None:
        return "inferential statistics unavailable"
    return f"mean difference `{estimate:.6f}`, CI `[{lower:.6f}, {upper:.6f}]`"


def _phase5_negative_evidence_text(evidence: NegativeSideEvidence | None) -> str:
    """Render the separately adjusted evidence used by VALIDATED_NEGATIVE gates."""

    if evidence is None:
        return "adjusted negative-side evidence unavailable"
    decisive = str(evidence.decisive_negative).lower()
    return (
        f"adjusted negative-side evidence `{evidence.gate_id}` via "
        f"`{evidence.method_version}`: familywise alpha "
        f"`{evidence.familywise_alpha:.6f}` across `{evidence.family_size}` gates, "
        "adjusted two-sided confidence "
        f"`{evidence.adjusted_two_sided_confidence_level:.6f}`, simultaneous CI "
        f"`[{evidence.bootstrap.lower:.6f}, {evidence.bootstrap.upper:.6f}]`, "
        f"practical threshold `{evidence.practical_effect_threshold:.6f}`, "
        f"decisive negative `{decisive}`"
    )


def _phase5_confirmatory_report_text(
    manifest: RunManifest,
    finalization: RunFinalization,
    turns: tuple[StoredTurn, ...],
    analysis: StoredScientificAnalysis,
) -> str:
    """Render the seven exact scientific closeout sections."""

    accounting = finalization.execution_accounting
    if accounting is None:
        raise ValueError("confirmatory report lacks durable execution accounting")
    result = analysis.result
    efficacy_lines = []
    for comparison in result.efficacy_comparisons:
        bootstrap = comparison.bootstrap
        efficacy_lines.append(
            f"- `{comparison.focal_policy_id}` vs `{comparison.comparator_policy_id}` "
            f"({comparison.comparator_role.value}): `{comparison.status.value}`; "
            + _phase5_effect_text(
                comparison.mean_difference,
                None if bootstrap is None else bootstrap.lower,
                None if bootstrap is None else bootstrap.upper,
            )
            + "; "
            + _phase5_negative_evidence_text(comparison.negative_side_evidence)
            + "."
        )
    recovery_lines = [
        f"- Recovery family: `{result.recovery.status.value}`; {result.recovery.detail}",
        f"- Right-censored recovery units: focal "
        f"`{result.recovery.right_censored_focal_units}`, serious comparators "
        f"`{result.recovery.right_censored_comparator_units}`.",
    ]
    recovery_lines.extend(
        f"- `{metric.metric_name.value}`: `{metric.status.value}`; "
        f"{_phase5_effect_text(metric.estimate, metric.bootstrap.lower, metric.bootstrap.upper)}; "
        f"{_phase5_negative_evidence_text(metric.negative_side_evidence)}."
        for metric in result.recovery.metric_results
    )
    attribution = result.attribution
    attribution_bootstrap = attribution.bootstrap
    attribution_effect_text = _phase5_effect_text(
        attribution.mean_difference,
        None if attribution_bootstrap is None else attribution_bootstrap.lower,
        None if attribution_bootstrap is None else attribution_bootstrap.upper,
    )
    attribution_lines = (
        f"- `{attribution.focal_policy_id}` vs `{attribution.comparator_policy_id}`: "
        f"`{attribution.status.value}`; "
        f"{attribution_effect_text}; "
        f"{_phase5_negative_evidence_text(attribution.negative_side_evidence)}.\n"
        "- This matched-history reset comparison is attribution-only, excludes turn zero, "
        "and is not an efficacy baseline or Holm-family member."
    )
    guardrail_lines = "\n".join(
        f"- `{guardrail.name}` (`{guardrail.scope}`): `{guardrail.status.value}` — "
        f"{guardrail.detail}"
        for guardrail in sorted(
            result.guardrails,
            key=lambda guardrail: (guardrail.name, guardrail.scope),
        )
    )
    limitation_lines = (
        "\n".join(
            f"- `{limitation.code}`: `{limitation.disposition.value}` — {limitation.detail}"
            for limitation in result.limitations
        )
        if result.limitations
        else "- No preregistered limitations were triggered."
    )
    efficacy_text = "\n".join(efficacy_lines)
    recovery_text = "\n".join(recovery_lines)
    subgroup_lines = []
    for effect in result.subgroup_effects:
        effect_text = _phase5_effect_text(
            effect.bootstrap.estimate,
            effect.bootstrap.lower,
            effect.bootstrap.upper,
        )
        subgroup_lines.append(
            f"- Subgroup `{effect.field_name}={effect.field_value}` vs "
            f"`{effect.comparator_policy_id}`: `{effect.direction}`; {effect_text}."
        )
    subgroup_text = (
        "\n".join(subgroup_lines)
        if subgroup_lines
        else "- No multi-level preregistered subgroup analysis was required."
    )
    reason_codes = ", ".join(f"`{reason.value}`" for reason in result.decision.reason_codes)
    return (
        "# NeuraLLM Phase 5 Confirmatory Scientific Report\n\n"
        "## Engineering validity\n\n"
        f"- Manifest SHA-256: `{canonical_sha256(manifest)}`\n"
        f"- Scientific result SHA-256: `{scientific_result_sha256(turns)}`\n"
        f"- Evaluation result SHA-256: `{result.result_sha256}`\n"
        f"- Scientific identity SHA-256: `{manifest.scientific_identity_sha256}`\n"
        f"- Preregistration SHA-256: `{manifest.preregistration_sha256}`\n"
        "- Confirmatory analysis contract SHA-256: "
        f"`{manifest.confirmatory_analysis_contract_sha256}`\n"
        f"- Provider identity: `{manifest.provider_identity.identity_id}`\n"
        f"- Exact matched coverage: `{result.coverage.exact}` "
        f"(`{result.coverage.observed_count}/{result.coverage.expected_count}`)\n"
        f"- Logical generations planned/dispatched/successful/committed: "
        f"`{accounting.planned_logical_generations}/"
        f"{accounting.dispatched_logical_generations}/"
        f"{accounting.successful_responses}/"
        f"{accounting.committed_logical_generations}`; uncertain: "
        f"`{accounting.uncertain_dispatches}`\n"
        f"- Persisted statistical computations: `{result.statistics_call_count}`\n\n"
        "## Controller activity\n\n"
        f"{_model_backed_activity_lines(manifest, turns)}\n\n"
        "## End-to-end efficacy\n\n"
        f"{efficacy_text}\n"
        f"{recovery_text}\n"
        f"{subgroup_text}\n\n"
        "## Persistent-state attribution\n\n"
        f"{attribution_lines}\n\n"
        "## Guardrail outcomes\n\n"
        f"{guardrail_lines}\n\n"
        "## Limitations\n\n"
        f"{limitation_lines}\n\n"
        "## Final decision\n\n"
        f"`{result.decision.decision.value}`. Reason codes: {reason_codes}.\n"
    )


def _phase3_report_text(
    manifest: RunManifest,
    turns: tuple[StoredTurn, ...],
    analysis: StoredAnalysis,
) -> str:
    result = analysis.result
    policy_activity: dict[str, list[float]] = {}
    for outcome in result.outcomes:
        policy_activity.setdefault(outcome.policy_id, []).append(outcome.action_magnitude)
    activity_lines = (
        "\n".join(
            f"- `{policy_id}` mean normalized action magnitude: `{sum(values) / len(values):.6f}`"
            for policy_id, values in sorted(policy_activity.items())
        )
        if policy_activity
        else "- No controller-activity summary is available because evaluation was invalidated."
    )
    comparison_lines = (
        "\n".join(
            f"- `{analysis.manifest.evaluation_spec.focal_policy_id}` vs "
            f"`{comparison.comparator_policy_id}`: `{comparison.verdict.value}`, "
            f"mean difference `{comparison.mean_difference:.6f}`, "
            f"{analysis.manifest.evaluation_spec.confidence_level * 100:.1f}% configured "
            f"CI `[ {comparison.bootstrap.lower:.6f}, "
            f"{comparison.bootstrap.upper:.6f} ]`."
            for comparison in result.comparisons
        )
        if result.comparisons
        else "- No pairwise comparisons were computed; inspect the invalid guardrails below."
    )
    guardrails = tuple(result.global_guardrails) + tuple(
        guardrail for comparison in result.comparisons for guardrail in comparison.guardrails
    )
    guardrail_lines = "\n".join(
        f"- `{guardrail.name.value}`"
        f"{f' (`{guardrail.policy_id}`)' if guardrail.policy_id else ''}: "
        f"`{guardrail.status.value}` — {guardrail.detail}"
        for guardrail in guardrails
    )
    return (
        "# NeuraLLM Phase 3 Baseline Evaluator Report\n\n"
        "## Scope\n\n"
        "This report covers the Phase 3 baseline and statistical-evaluator protocol only. "
        "It does not establish neural-controller efficacy or persistent-state attribution.\n\n"
        "## Engineering validity\n\n"
        f"- Manifest SHA-256: `{canonical_sha256(manifest)}`\n"
        f"- Scientific result SHA-256: `{scientific_result_sha256(turns)}`\n"
        f"- Evaluation result SHA-256: `{result.result_sha256}`\n"
        f"- Provider identity: `{manifest.provider_identity.identity_id}`\n"
        f"- Committed turns: `{len(turns)}`\n"
        f"- Exact matched coverage: `{result.coverage.exact}`\n"
        f"- Statistics computed: `{result.statistics_computed}` "
        f"(`{result.statistics_call_count}` calls)\n\n"
        "## Baseline evaluator validation\n\n"
        f"{comparison_lines}\n\n"
        "## Controller activity\n\n"
        f"{activity_lines}\n\n"
        "## Guardrail outcomes\n\n"
        f"{guardrail_lines}\n\n"
        "## End-to-end efficacy\n\n"
        "Not assessed in Phase 3. No final scientific decision is emitted.\n\n"
        "## Persistent-state attribution\n\n"
        "Not assessed; the matched-history state-reset comparator begins in Phase 4.\n\n"
        "## Limitations\n\n"
        "A Phase 3 verdict describes behavior under this frozen baseline-evaluator protocol. "
        "It cannot be promoted to a neural-efficacy, biological-substrate, or Phase 5 claim.\n\n"
        "## Phase 3 result\n\n"
        f"Baseline evaluator verdict: `{result.verdict.value}`. "
        "`scientific_decision` remains `null`.\n"
    )


def export_closed_run(output_directory: Path) -> ArtifactExportSummary:
    """Verify and export exactly the compact artifact set for a closed run."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path")
    output_directory = output_directory.expanduser().resolve(strict=True)
    if not output_directory.is_dir():
        raise ValueError("output_directory must be a directory")
    _reject_unexpected_files(output_directory, allow_sqlite_recovery_sidecars=True)
    database_path = output_directory / "run.sqlite3"
    if not database_path.is_file():
        raise FileNotFoundError("run directory does not contain run.sqlite3")

    with SQLiteRunStore(database_path) as store:
        store.verify_integrity()
        manifest = store.get_manifest()
        if manifest is None:
            raise ValueError("run store does not contain a manifest")
        finalization = store.get_finalization()
        if finalization is None:
            raise ValueError("run store is not finalized")
        turns = store.list_turns()
        if not turns:
            raise ValueError("closed run must contain at least one turn")
        incomplete = tuple(
            turn.condition_id for turn in turns if turn.state is not TurnState.COMMITTED
        )
        if incomplete:
            raise ValueError(f"run contains non-committed turns: {incomplete!r}")
        result_sha256 = scientific_result_sha256(turns)
        if result_sha256 != finalization.scientific_result_sha256:
            raise ValueError(
                "finalized scientific result hash does not match the recomputed output"
            )
        analysis = store.get_analysis()
        scientific_analysis = store.get_scientific_analysis()
        if manifest.decision_rule_version == _PHASE3_DECISION_RULE_VERSION:
            if analysis is None or scientific_analysis is not None:
                raise ValueError("Phase 3 run is missing finalized analysis evidence")
            if manifest.matched_history_policy_sources:
                raise ValueError("Phase 3 run cannot declare matched-history policy sources")
        elif manifest.decision_rule_version == _PHASE4_DECISION_RULE_VERSION:
            if analysis is not None or scientific_analysis is not None:
                raise ValueError("Phase 4 mechanism run cannot contain scientific analysis")
            if (
                dict(manifest.matched_history_policy_sources)
                != _PHASE4_MATCHED_HISTORY_POLICY_SOURCES
            ):
                raise ValueError("Phase 4 run lacks its exact matched-history policy edge")
            _validate_phase4_mechanism_evidence(
                manifest,
                turns,
                store.list_turn_inputs(),
            )
        elif manifest.decision_rule_version in _MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS:
            if analysis is not None or scientific_analysis is not None:
                raise ValueError("smoke and pilot runs cannot contain scientific analysis")
            if (
                dict(manifest.matched_history_policy_sources)
                != _PHASE4_MATCHED_HISTORY_POLICY_SOURCES
            ):
                raise ValueError("model-backed run lacks its exact attribution edge")
            expected_tier = _MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS[manifest.decision_rule_version]
            if manifest.run_tier != expected_tier:
                raise ValueError("model-backed manifest has the wrong run tier")
            if finalization.execution_accounting is None:
                raise ValueError("model-backed run lacks durable execution accounting")
            _validate_phase4_mechanism_evidence(
                manifest,
                turns,
                store.list_turn_inputs(),
            )
        elif manifest.decision_rule_version == _CONFIRMATORY_DECISION_RULE_VERSION:
            if analysis is not None or scientific_analysis is None:
                raise ValueError("confirmatory run lacks its finalized scientific analysis")
            if (
                manifest.run_tier != "confirmatory"
                or manifest.provider_identity.provider_type != "llama_cpp"
                or not manifest.working_tree_clean
                or dict(manifest.matched_history_policy_sources)
                != _PHASE4_MATCHED_HISTORY_POLICY_SOURCES
                or finalization.execution_accounting is None
            ):
                raise ValueError("confirmatory manifest lacks its claim-eligible identity")
            _validate_phase4_mechanism_evidence(
                manifest,
                turns,
                store.list_turn_inputs(),
            )
        elif manifest.decision_rule_version == _PHASE2_DECISION_RULE_VERSION:
            if analysis is not None or scientific_analysis is not None:
                raise ValueError("Phase 2 run unexpectedly contains analysis evidence")
            if manifest.matched_history_policy_sources:
                raise ValueError("Phase 2 run unexpectedly declares matched-history sources")
        else:
            raise ValueError(f"unknown decision rule version: {manifest.decision_rule_version!r}")
        store.compact()

    result_rows = tuple(_result_row(turn) for turn in turns)
    comparison_fields: tuple[str, ...]
    comparison_rows: tuple[dict[str, object], ...]
    implementation_phase: int
    if manifest.decision_rule_version == _PHASE4_DECISION_RULE_VERSION:
        comparison_fields = _PHASE2_COMPARISON_FIELDS
        comparison_rows = ()
        decision = _phase4_decision_payload(manifest, turns)
        report = _phase4_report_text(manifest, turns)
        implementation_phase = 4
    elif manifest.decision_rule_version in _MODEL_BACKED_NONSCIENTIFIC_RULE_TIERS:
        comparison_fields = _PHASE2_COMPARISON_FIELDS
        comparison_rows = ()
        decision = _model_backed_nonscientific_decision_payload(
            manifest,
            finalization,
            turns,
        )
        report = _model_backed_nonscientific_report_text(
            manifest,
            finalization,
            turns,
        )
        implementation_phase = 5
    elif manifest.decision_rule_version == _CONFIRMATORY_DECISION_RULE_VERSION:
        if scientific_analysis is None:
            raise ValueError("confirmatory export lacks scientific analysis evidence")
        comparison_fields = _PHASE5_COMPARISON_FIELDS
        comparison_rows = _phase5_comparison_rows(scientific_analysis)
        decision = _phase5_decision_payload(
            manifest,
            finalization,
            turns,
            scientific_analysis,
        )
        report = _phase5_confirmatory_report_text(
            manifest,
            finalization,
            turns,
            scientific_analysis,
        )
        implementation_phase = 5
    elif manifest.decision_rule_version == _PHASE2_DECISION_RULE_VERSION:
        comparison_fields = _PHASE2_COMPARISON_FIELDS
        comparison_rows = ()
        decision = _decision_payload(manifest, turns)
        report = _report_text(manifest, turns)
        implementation_phase = 2
    elif manifest.decision_rule_version == _PHASE3_DECISION_RULE_VERSION:
        if analysis is None:
            raise ValueError("Phase 3 export lacks analysis evidence")
        comparison_fields = _PHASE3_COMPARISON_FIELDS
        comparison_rows = _phase3_comparison_rows(analysis)
        decision = _phase3_decision_payload(manifest, turns, analysis)
        report = _phase3_report_text(manifest, turns, analysis)
        implementation_phase = 3
    else:
        raise ValueError(f"unknown decision rule version: {manifest.decision_rule_version!r}")
    _write_atomic(output_directory / "manifest.json", canonical_json(manifest) + "\n")
    _write_atomic(output_directory / "results.csv", _csv_text(_RESULT_FIELDS, result_rows))
    _write_atomic(
        output_directory / "comparisons.csv",
        _csv_text(comparison_fields, comparison_rows),
    )
    _write_atomic(output_directory / "decision.json", canonical_json(decision) + "\n")
    _write_atomic(output_directory / "report.md", report)
    _reject_unexpected_files(output_directory)
    return ArtifactExportSummary(
        output_directory=output_directory,
        manifest_sha256=canonical_sha256(manifest),
        scientific_result_sha256=result_sha256,
        committed_turns=len(turns),
        artifact_names=tuple(sorted(CLOSED_RUN_ARTIFACTS)),
        implementation_phase=implementation_phase,
        phase3_baseline_evaluator_verdict=(
            None if analysis is None else analysis.result.verdict.value
        ),
        scientific_decision=(
            None
            if scientific_analysis is None
            else scientific_analysis.result.decision.decision.value
        ),
    )


__all__ = [
    "CLOSED_RUN_ARTIFACTS",
    "SQLITE_RECOVERY_SIDECARS",
    "ArtifactExportSummary",
    "export_closed_run",
    "scientific_result_sha256",
]

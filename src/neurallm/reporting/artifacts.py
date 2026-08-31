"""Export compact deterministic views of canonical Phase 2 through Phase 4 stores."""

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
    ControllerObservation,
    MetricValue,
    ResponseMetrics,
    RunManifest,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    SQLiteRunStore,
    StoredAnalysis,
    StoredTurn,
    TurnInputEvidence,
    TurnState,
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

_PHASE3_DECISION_RULE_VERSION = "phase3-baseline-evaluator-v1"
_PHASE4_DECISION_RULE_VERSION = "phase4-neural-mechanism-only-v1"
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
    expected_policy_ids = {persistent_id, reset_id}
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
    for turn in turns:
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


def scientific_result_sha256(turns: tuple[StoredTurn, ...]) -> str:
    """Hash only canonical committed scientific results, excluding run location/source state."""

    if not turns:
        raise ValueError("scientific result requires at least one committed turn")
    evidence: list[dict[str, object]] = []
    for turn in turns:
        if (
            turn.state is not TurnState.COMMITTED
            or turn.response is None
            or turn.metrics is None
            or turn.policy_state_json is None
            or turn.policy_trace_json is None
            or turn.history_commitment_sha256 is None
        ):
            raise ValueError("scientific result contains incomplete turn evidence")
        evidence.append(
            {
                "condition_id": turn.condition_id,
                "request": turn.request,
                "history_binding": turn.history,
                "response": turn.response,
                "metrics": turn.metrics,
                "policy_state": json.loads(turn.policy_state_json),
                "policy_trace": json.loads(turn.policy_trace_json),
                "history_commitment_sha256": turn.history_commitment_sha256,
            }
        )
    return canonical_sha256({"schema_version": 1, "turns": evidence})


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
        if manifest.decision_rule_version == _PHASE3_DECISION_RULE_VERSION:
            if analysis is None:
                raise ValueError("Phase 3 run is missing finalized analysis evidence")
            if manifest.matched_history_policy_sources:
                raise ValueError("Phase 3 run cannot declare matched-history policy sources")
        elif manifest.decision_rule_version == _PHASE4_DECISION_RULE_VERSION:
            if analysis is not None:
                raise ValueError("Phase 4 mechanism run cannot contain Phase 3 analysis")
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
        elif analysis is not None:
            raise ValueError("pre-Phase 3 run unexpectedly contains analysis evidence")
        elif manifest.matched_history_policy_sources:
            raise ValueError("pre-Phase 4 run unexpectedly declares matched-history policy sources")
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
    elif analysis is None:
        comparison_fields = _PHASE2_COMPARISON_FIELDS
        comparison_rows = ()
        decision = _decision_payload(manifest, turns)
        report = _report_text(manifest, turns)
        implementation_phase = 2
    else:
        comparison_fields = _PHASE3_COMPARISON_FIELDS
        comparison_rows = _phase3_comparison_rows(analysis)
        decision = _phase3_decision_payload(manifest, turns, analysis)
        report = _phase3_report_text(manifest, turns, analysis)
        implementation_phase = 3
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
    )


__all__ = [
    "CLOSED_RUN_ARTIFACTS",
    "SQLITE_RECOVERY_SIDECARS",
    "ArtifactExportSummary",
    "export_closed_run",
    "scientific_result_sha256",
]

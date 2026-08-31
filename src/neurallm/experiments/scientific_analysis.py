"""Read-only confirmatory scientific orchestration over a closed run store."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from neurallm.domain.models import RunManifest, Sha256Hex
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.aggregation import (
    aggregate_matched_units,
    record_sort_key,
    validate_exact_coverage,
)
from neurallm.evaluation.attribution import (
    PersistentStateAttributionResult,
    evaluate_persistent_state_attribution,
)
from neurallm.evaluation.confirmatory import (
    AttributionUnitOutcome,
    ConfirmatoryAnalysisSpec,
    ConfirmatoryEvaluationResult,
    RecoveryUnitOutcome,
    ScientificUnitOutcome,
    SubgroupEffectResult,
    confirmatory_result_sha256,
    confirmatory_statistics_call_count,
)
from neurallm.evaluation.guardrails import (
    action_saturation_guardrail,
    integrity_guardrails,
    pairwise_guardrails,
)
from neurallm.evaluation.models import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    GuardrailName,
    GuardrailResult,
    GuardrailStatus,
    MatchedUnitKey,
    SequencePolicyOutcome,
    TurnEvaluationRecord,
)
from neurallm.evaluation.recovery import (
    RecoveryEvaluationResult,
    RecoveryMetricName,
    evaluate_recovery,
    post_stressor_repetition_change,
    post_stressor_task_score_change,
    time_to_return_to_target_band,
)
from neurallm.evaluation.scientific import (
    ATTRIBUTION_COMPARATOR_ID,
    EFFICACY_COMPARATOR_IDS,
    FOCAL_POLICY_ID,
    SERIOUS_COMPARATOR_IDS,
    ExperimentTier,
    GuardrailCleanTaskScore,
    LimitationDisposition,
    LimitationKind,
    ScientificDecisionInput,
    ScientificEvidenceStatus,
    ScientificFrozenModel,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
    ScientificLimitation,
    decide_scientific_outcome,
    evaluate_efficacy_comparisons,
)
from neurallm.evaluation.statistics import paired_bootstrap_ci
from neurallm.experiments.analysis import (
    build_evaluation_design,
    evaluation_records_from_store,
)
from neurallm.experiments.plan import ExperimentPlan
from neurallm.experiments.protocol import (
    ATTRIBUTION_HISTORY_SOURCE_POLICY_ID,
    ATTRIBUTION_POLICY_ID,
    CONFIRMATORY_DECISION_RULE_VERSION,
    EFFICACY_POLICY_IDS,
    MODEL_BACKED_POLICY_IDS,
    RunTier,
)
from neurallm.providers.llama_cpp import require_llama_cpp_provider_binding
from neurallm.storage import (
    RunFinalization,
    SQLiteRunStore,
    StoredTurn,
    StoreInvariantError,
    TurnInputEvidence,
    scientific_result_sha256,
)


class ConfirmatoryAnalysisContext(ScientificFrozenModel):
    """Small provenance boundary distinguishing real claims from test assembly."""

    analysis_contract_sha256: Sha256Hex
    evaluation_input_sha256: Sha256Hex
    causal_mechanism_validated: Literal[True] = True
    claim_eligible: bool
    run_manifest_sha256: Sha256Hex | None = None
    run_finalization_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def validate_claim_binding(self) -> Self:
        has_closed_run_bindings = (
            self.run_manifest_sha256 is not None and self.run_finalization_sha256 is not None
        )
        if self.claim_eligible != has_closed_run_bindings:
            raise ValueError(
                "claim eligibility requires both closed-run manifest and finalization bindings"
            )
        return self


def _require_confirmatory_plan(plan: ExperimentPlan) -> ConfirmatoryAnalysisSpec:
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("plan must be an ExperimentPlan")
    if (
        plan.protocol is None
        or plan.protocol.run_tier is not RunTier.CONFIRMATORY
        or plan.protocol.policy_ids != MODEL_BACKED_POLICY_IDS
        or plan.protocol.efficacy_policy_ids != EFFICACY_POLICY_IDS
        or plan.protocol.attribution.policy_id != ATTRIBUTION_POLICY_ID
        or plan.protocol.attribution.history_source_policy_id
        != ATTRIBUTION_HISTORY_SOURCE_POLICY_ID
    ):
        raise ValueError("scientific analysis requires the exact confirmatory five-arm protocol")
    if (
        plan.provider_identity.provider_type != "llama_cpp"
        or plan.provider_identity.model_sha256 is None
    ):
        raise ValueError(
            "scientific analysis requires llama_cpp provider identity with a model-artifact SHA-256"
        )
    try:
        require_llama_cpp_provider_binding(
            plan.provider_identity,
            plan.provider_effective_configuration_json,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scientific analysis requires internally consistent llama_cpp identity evidence"
        ) from exc
    if (
        plan.confirmatory_analysis is None
        or plan.evaluation is None
        or plan.evaluation_spec_sha256 != canonical_sha256(plan.evaluation)
        or plan.static_selection_record is None
        or plan.static_selection_result_sha256
        != plan.static_selection_record.selection_result_sha256
    ):
        raise ValueError("confirmatory analysis plan lacks its frozen analysis evidence")
    if (
        plan.dataset_purpose is not DatasetPurpose.EVALUATION
        or plan.dataset_seal is None
        or plan.dataset_seal.dataset_sha256 != plan.dataset_hash
    ):
        raise ValueError("confirmatory analysis requires its sealed evaluation dataset")
    if (
        plan.preregistration is None
        or plan.preregistration.experiment_id != plan.experiment_id
        or plan.preregistration.run_tier != RunTier.CONFIRMATORY.value
        or plan.preregistration.scientific_identity_sha256 != plan.scientific_identity_sha256
    ):
        raise ValueError("confirmatory analysis requires its exact published preregistration")
    if {turn.condition.policy_id for turn in plan.turns} != set(MODEL_BACKED_POLICY_IDS):
        raise ValueError("confirmatory analysis plan lacks the exact five policy arms")
    if plan.confirmatory_analysis.subgroup_fields != ("prompt_family",):
        raise ValueError(
            "confirmatory orchestration supports only the frozen prompt_family subgroup"
        )
    return plan.confirmatory_analysis


def confirmatory_analysis_contract_sha256(
    *,
    scientific_identity_sha256: str,
    preregistration_sha256: str,
    confirmatory_analysis_spec: ConfirmatoryAnalysisSpec,
    confirmatory_analysis_spec_sha256: str,
    evaluation_spec: EvaluationSpec,
    evaluation_spec_sha256: str,
    turn_input_evidence_sha256: str,
    prompt_family_by_sequence: Mapping[str, str],
    prompt_family_design_sha256: str,
    dataset_sha256: str,
    dataset_purpose: DatasetPurpose,
    dataset_seal_sha256: str,
) -> str:
    """Hash the pre-execution contract from persistable manifest evidence.

    ``scientific_identity_sha256`` already commits to the complete plan with
    its preregistration field removed: protocol, evaluation/static-selection
    evidence, matched coverage, dataset identity, and the finalized schedule.
    The remaining fields are repeated deliberately so storage can independently
    prove the submitted analysis spec and published preregistration are exactly
    the pre-execution ones without retaining an ``ExperimentPlan`` object.
    """

    if confirmatory_analysis_spec_sha256 != canonical_sha256(confirmatory_analysis_spec):
        raise ValueError("confirmatory analysis spec hash does not match its canonical evidence")
    if evaluation_spec_sha256 != canonical_sha256(evaluation_spec):
        raise ValueError("evaluation spec hash does not match its canonical evidence")
    if not prompt_family_by_sequence or prompt_family_design_sha256 != canonical_sha256(
        prompt_family_by_sequence
    ):
        raise ValueError("prompt-family design hash does not match its canonical evidence")
    if dataset_purpose is not DatasetPurpose.EVALUATION:
        raise ValueError("confirmatory analysis requires evaluation-purpose data")
    return canonical_sha256(
        {
            "schema_version": 2,
            "implementation_version": "confirmatory-analysis-contract-v2",
            "scientific_identity_sha256": scientific_identity_sha256,
            "preregistration_sha256": preregistration_sha256,
            "confirmatory_analysis_spec": confirmatory_analysis_spec,
            "confirmatory_analysis_spec_sha256": confirmatory_analysis_spec_sha256,
            "evaluation_spec": evaluation_spec,
            "evaluation_spec_sha256": evaluation_spec_sha256,
            "turn_input_evidence_sha256": turn_input_evidence_sha256,
            "prompt_family_by_sequence": dict(sorted(prompt_family_by_sequence.items())),
            "prompt_family_design_sha256": prompt_family_design_sha256,
            "dataset_sha256": dataset_sha256,
            "dataset_purpose": dataset_purpose,
            "dataset_seal_sha256": dataset_seal_sha256,
        }
    )


def build_confirmatory_analysis_contract_sha256(plan: ExperimentPlan) -> str:
    """Build the persistable confirmatory contract identity from a frozen plan."""

    spec = _require_confirmatory_plan(plan)
    assert plan.dataset_seal is not None
    assert plan.preregistration is not None
    assert plan.dataset_purpose is not None
    assert plan.evaluation is not None
    assert plan.evaluation_spec_sha256 is not None
    prompt_family_by_sequence = _prompt_family_by_sequence(plan)
    return confirmatory_analysis_contract_sha256(
        scientific_identity_sha256=plan.scientific_identity_sha256,
        preregistration_sha256=plan.preregistration.seal_sha256,
        confirmatory_analysis_spec=spec,
        confirmatory_analysis_spec_sha256=canonical_sha256(spec),
        evaluation_spec=plan.evaluation,
        evaluation_spec_sha256=plan.evaluation_spec_sha256,
        turn_input_evidence_sha256=build_confirmatory_turn_input_evidence_sha256(plan),
        prompt_family_by_sequence=prompt_family_by_sequence,
        prompt_family_design_sha256=canonical_sha256(prompt_family_by_sequence),
        dataset_sha256=plan.dataset_hash,
        dataset_purpose=plan.dataset_purpose,
        dataset_seal_sha256=plan.dataset_seal.seal_sha256,
    )


def _validate_manifest(plan: ExperimentPlan, manifest: RunManifest) -> None:
    expected_model_seeds = tuple(sorted({turn.condition.model_seed for turn in plan.turns}))
    expected_controller_seeds = tuple(
        sorted({turn.condition.controller_seed for turn in plan.turns})
    )
    assert plan.preregistration is not None
    if (
        not manifest.working_tree_clean
        or manifest.experiment_config_hash != plan.experiment_config_hash
        or manifest.dataset_hash != plan.dataset_hash
        or manifest.provider_identity != plan.provider_identity
        or manifest.provider_identity.provider_type != "llama_cpp"
        or manifest.provider_identity.model_sha256 is None
        or manifest.provider_effective_configuration_json
        != plan.provider_effective_configuration_json
        or manifest.action_bounds != plan.action_bounds
        or manifest.decoding_bounds != plan.decoding_bounds
        or dict(manifest.metric_versions) != dict(plan.metric_versions)
        or manifest.decision_rule_version != CONFIRMATORY_DECISION_RULE_VERSION
        or manifest.database_schema_version != plan.database_schema_version
        or set(manifest.policy_config_hashes) != set(MODEL_BACKED_POLICY_IDS)
        or dict(manifest.matched_history_policy_sources)
        != {ATTRIBUTION_POLICY_ID: ATTRIBUTION_HISTORY_SOURCE_POLICY_ID}
        or manifest.seed_schedule.model_seeds != expected_model_seeds
        or manifest.seed_schedule.controller_seeds != expected_controller_seeds
        or manifest.run_tier != RunTier.CONFIRMATORY.value
        or manifest.scientific_identity_sha256 != plan.scientific_identity_sha256
        or manifest.preregistration_sha256 != plan.preregistration.seal_sha256
        or manifest.turn_input_evidence_sha256
        != build_confirmatory_turn_input_evidence_sha256(plan)
        or manifest.confirmatory_analysis_contract_sha256
        != build_confirmatory_analysis_contract_sha256(plan)
        or manifest.phase3_analysis_contract_sha256 is not None
    ):
        raise StoreInvariantError(
            "closed run manifest does not exactly match the confirmatory scientific plan"
        )


def _validate_finalization(
    plan: ExperimentPlan,
    manifest: RunManifest,
    finalization: RunFinalization,
) -> None:
    expected_condition_ids = tuple(sorted(turn.condition.condition_id for turn in plan.turns))
    accounting = finalization.execution_accounting
    if (
        finalization.manifest_sha256 != canonical_sha256(manifest)
        or finalization.expected_condition_ids != expected_condition_ids
        or finalization.expected_condition_count != len(expected_condition_ids)
        or accounting is None
        or accounting.planned_logical_generations != len(expected_condition_ids)
        or accounting.committed_logical_generations != len(expected_condition_ids)
        or accounting.successful_responses != len(expected_condition_ids)
        or accounting.uncertain_dispatches != 0
    ):
        raise StoreInvariantError("run finalization does not close the exact confirmatory schedule")


def _scientific_status(status: GuardrailStatus) -> ScientificGuardrailStatus:
    return {
        GuardrailStatus.PASS: ScientificGuardrailStatus.PASS,
        GuardrailStatus.FAIL: ScientificGuardrailStatus.FAIL,
        GuardrailStatus.INVALID: ScientificGuardrailStatus.INVALID,
    }[status]


def _scientific_guardrail(
    guardrail: GuardrailResult,
    *,
    scope_prefix: str,
) -> ScientificGuardrailResult:
    if guardrail.comparator_policy_id is not None:
        scope = f"{scope_prefix}:pair:{guardrail.policy_id}:{guardrail.comparator_policy_id}"
    elif guardrail.policy_id is not None:
        scope = f"{scope_prefix}:policy:{guardrail.policy_id}"
    else:
        scope = f"{scope_prefix}:global"
    return ScientificGuardrailResult(
        name=guardrail.name.value,
        status=_scientific_status(guardrail.status),
        scope=scope,
        detail=guardrail.detail,
        observed_value=guardrail.observed_value,
        threshold=guardrail.threshold,
    )


def _scientific_guardrails_from_records(
    records: tuple[TurnEvaluationRecord, ...],
    design: ExpectedEvaluationDesign,
    evaluation_spec: EvaluationSpec,
) -> tuple[ScientificGuardrailResult, ...]:
    """Recompute every confirmatory guardrail from frozen source evidence."""

    coverage = validate_exact_coverage(records, design)
    global_guardrails = tuple(
        _scientific_guardrail(guardrail, scope_prefix="efficacy")
        for guardrail in integrity_guardrails(records, design, coverage)
    )
    efficacy_records = tuple(
        record for record in records if record.policy_id in EFFICACY_POLICY_IDS
    )
    efficacy_by_policy = _outcomes_by_policy(aggregate_matched_units(efficacy_records))
    if set(efficacy_by_policy) != set(EFFICACY_POLICY_IDS):
        raise ValueError("efficacy aggregation must contain exactly four independent arms")
    focal = tuple(efficacy_by_policy[FOCAL_POLICY_ID].values())
    saturation = _scientific_guardrail(
        action_saturation_guardrail(
            efficacy_records,
            focal_policy_id=FOCAL_POLICY_ID,
            maximum_rate=evaluation_spec.maximum_action_saturation_rate,
        ),
        scope_prefix="efficacy",
    )
    efficacy_pairs = tuple(
        _scientific_guardrail(guardrail, scope_prefix="efficacy")
        for comparator_id in EFFICACY_COMPARATOR_IDS
        for guardrail in pairwise_guardrails(
            focal,
            tuple(efficacy_by_policy[comparator_id].values()),
            focal_policy_id=FOCAL_POLICY_ID,
            comparator_policy_id=comparator_id,
            maximum_adherence_regression=evaluation_spec.maximum_adherence_regression,
            maximum_length_reduction_ratio=evaluation_spec.maximum_length_reduction_ratio,
            behavioral_alias_tolerance=evaluation_spec.behavioral_alias_tolerance,
        )
    )
    attribution_records = tuple(
        record
        for record in records
        if record.turn_index > 0
        and record.policy_id in {FOCAL_POLICY_ID, ATTRIBUTION_COMPARATOR_ID}
    )
    attribution_by_policy = _outcomes_by_policy(aggregate_matched_units(attribution_records))
    if set(attribution_by_policy) != {FOCAL_POLICY_ID, ATTRIBUTION_COMPARATOR_ID}:
        raise ValueError("attribution requires exact persistent/reset intervention units")
    attribution_pairs = tuple(
        _scientific_guardrail(guardrail, scope_prefix="attribution")
        for guardrail in pairwise_guardrails(
            tuple(attribution_by_policy[FOCAL_POLICY_ID].values()),
            tuple(attribution_by_policy[ATTRIBUTION_COMPARATOR_ID].values()),
            focal_policy_id=FOCAL_POLICY_ID,
            comparator_policy_id=ATTRIBUTION_COMPARATOR_ID,
            maximum_adherence_regression=evaluation_spec.maximum_adherence_regression,
            maximum_length_reduction_ratio=evaluation_spec.maximum_length_reduction_ratio,
            behavioral_alias_tolerance=evaluation_spec.behavioral_alias_tolerance,
        )
    )
    intervention_count = len(attribution_records)
    causal_guardrails = (
        ScientificGuardrailResult(
            name="causal_mechanism_validation",
            status=ScientificGuardrailStatus.PASS,
            scope="attribution:causal",
            detail="stored persistent/reset mechanism evidence passed the causal validator",
        ),
        ScientificGuardrailResult(
            name="intervention_turn_only_attribution",
            status=ScientificGuardrailStatus.PASS,
            scope="attribution:causal",
            detail="turn zero is excluded and every attribution observation has turn_index > 0",
            observed_value=float(intervention_count),
            threshold=float(intervention_count),
        ),
    )
    return (
        *global_guardrails,
        saturation,
        *efficacy_pairs,
        *causal_guardrails,
        *attribution_pairs,
    )


def _outcomes_by_policy(
    outcomes: Sequence[SequencePolicyOutcome],
) -> dict[str, dict[tuple[str, int], SequencePolicyOutcome]]:
    grouped: dict[str, dict[tuple[str, int], SequencePolicyOutcome]] = defaultdict(dict)
    for outcome in outcomes:
        key = (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed)
        if key in grouped[outcome.policy_id]:
            raise ValueError("aggregated scientific outcomes contain a duplicate matched unit")
        grouped[outcome.policy_id][key] = outcome
    return dict(grouped)


def _paired_task_differences(
    outcomes: Mapping[str, Mapping[tuple[str, int], SequencePolicyOutcome]],
    comparator_policy_id: str,
) -> tuple[float, ...]:
    focal = outcomes[FOCAL_POLICY_ID]
    comparator = outcomes[comparator_policy_id]
    if focal.keys() != comparator.keys():
        raise ValueError("scientific efficacy outcomes lack exact matched-unit keys")
    return tuple(focal[key].task_score - comparator[key].task_score for key in sorted(focal))


def _gate_status(
    guardrails: Sequence[ScientificGuardrailResult],
) -> ScientificGuardrailStatus:
    if any(guardrail.status is ScientificGuardrailStatus.INVALID for guardrail in guardrails):
        return ScientificGuardrailStatus.INVALID
    if any(guardrail.status is ScientificGuardrailStatus.FAIL for guardrail in guardrails):
        return ScientificGuardrailStatus.FAIL
    return ScientificGuardrailStatus.PASS


def _scientific_unit_outcomes(
    outcomes: Sequence[SequencePolicyOutcome],
    *,
    prompt_family_by_sequence: Mapping[str, str],
    global_guardrails: tuple[ScientificGuardrailResult, ...],
    saturation_guardrail: ScientificGuardrailResult,
    pair_guardrails: Mapping[str, tuple[ScientificGuardrailResult, ...]],
) -> tuple[ScientificUnitOutcome, ...]:
    scientific: list[ScientificUnitOutcome] = []
    for outcome in outcomes:
        applicable = list(global_guardrails)
        if outcome.policy_id == FOCAL_POLICY_ID:
            applicable.append(saturation_guardrail)
            for comparator_policy_id in EFFICACY_COMPARATOR_IDS:
                applicable.extend(pair_guardrails[comparator_policy_id])
        elif outcome.policy_id in pair_guardrails:
            applicable.extend(pair_guardrails[outcome.policy_id])
        gate_names = tuple(sorted({guardrail.name for guardrail in applicable}))
        scientific.append(
            ScientificUnitOutcome.model_validate(
                {
                    "unit_key": outcome.unit_key,
                    "prompt_family": prompt_family_by_sequence[outcome.unit_key.prompt_sequence_id],
                    "policy_id": outcome.policy_id,
                    "guardrail_clean_task_score": GuardrailCleanTaskScore(
                        raw_task_score=outcome.task_score,
                        gate_status=_gate_status(applicable),
                        gate_names=gate_names,
                    ),
                    "instruction_adherence": outcome.instruction_adherence,
                    "repetition_ratio": outcome.repetition_ratio,
                    "response_length_tokens": outcome.response_length_tokens,
                }
            )
        )
    return tuple(
        sorted(
            scientific,
            key=lambda item: (
                item.unit_key.prompt_sequence_id,
                item.unit_key.model_seed,
                item.policy_id,
            ),
        )
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("scientific aggregation cannot average an empty group")
    return sum(values) / len(values)


def _turn_metric_means(
    records: Sequence[TurnEvaluationRecord],
) -> dict[tuple[str, int, str, int], tuple[float, float]]:
    groups: dict[tuple[str, int, str, int], list[TurnEvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[
            (
                record.prompt_sequence_id,
                record.model_seed,
                record.policy_id,
                record.turn_index,
            )
        ].append(record)
    means: dict[tuple[str, int, str, int], tuple[float, float]] = {}
    for key, group in sorted(groups.items()):
        task_scores: list[float] = []
        repetition_ratios: list[float] = []
        for record in group:
            if record.task_score is None or record.repetition_ratio is None:
                raise ValueError("recovery requires complete task and repetition metrics")
            task_scores.append(record.task_score)
            repetition_ratios.append(record.repetition_ratio)
        means[key] = (
            _mean(task_scores),
            _mean(repetition_ratios),
        )
    return means


def _recovery_evidence(
    records: Sequence[TurnEvaluationRecord],
    spec: ConfirmatoryAnalysisSpec,
    *,
    perform_statistics: bool = True,
) -> tuple[RecoveryEvaluationResult, tuple[RecoveryUnitOutcome, ...], int, int]:
    means = _turn_metric_means(records)
    model_seeds = tuple(sorted({record.model_seed for record in records}))
    task_differences: list[float] = []
    repetition_differences: list[float] = []
    return_time_differences: list[float] = []
    unit_outcomes: list[RecoveryUnitOutcome] = []
    focal_censored = 0
    comparator_censored = 0
    for event in spec.recovery_events:
        for model_seed in model_seeds:
            unit_comparator_censored = 0
            focal_stressor = means[
                (
                    event.prompt_sequence_id,
                    model_seed,
                    FOCAL_POLICY_ID,
                    event.stressor_turn_index,
                )
            ]
            focal_recovery = tuple(
                means[(event.prompt_sequence_id, model_seed, FOCAL_POLICY_ID, turn_index)]
                for turn_index in event.recovery_turn_indexes
            )
            focal_task_change = post_stressor_task_score_change(
                focal_stressor[0],
                _mean(tuple(value[0] for value in focal_recovery)),
            )
            focal_repetition_change = post_stressor_repetition_change(
                focal_stressor[1],
                _mean(tuple(value[1] for value in focal_recovery)),
            )
            comparator_task_changes: list[float] = []
            comparator_repetition_changes: list[float] = []
            comparator_return_times: list[float] = []
            for comparator_policy_id in spec.recovery.serious_comparator_ids:
                stressor = means[
                    (
                        event.prompt_sequence_id,
                        model_seed,
                        comparator_policy_id,
                        event.stressor_turn_index,
                    )
                ]
                recovery = tuple(
                    means[
                        (
                            event.prompt_sequence_id,
                            model_seed,
                            comparator_policy_id,
                            turn_index,
                        )
                    ]
                    for turn_index in event.recovery_turn_indexes
                )
                comparator_task_changes.append(
                    post_stressor_task_score_change(
                        stressor[0],
                        _mean(tuple(value[0] for value in recovery)),
                    )
                )
                comparator_repetition_changes.append(
                    post_stressor_repetition_change(
                        stressor[1],
                        _mean(tuple(value[1] for value in recovery)),
                    )
                )
                return_time = time_to_return_to_target_band(
                    tuple(value[0] for value in recovery),
                    tuple(value[1] for value in recovery),
                    minimum_task_score=event.minimum_task_score_target,
                    maximum_repetition_ratio=event.maximum_repetition_ratio_target,
                )
                if return_time is None:
                    comparator_censored += 1
                    unit_comparator_censored += 1
                    return_time = len(event.recovery_turn_indexes) + 1
                comparator_return_times.append(float(return_time))
            focal_return_time = time_to_return_to_target_band(
                tuple(value[0] for value in focal_recovery),
                tuple(value[1] for value in focal_recovery),
                minimum_task_score=event.minimum_task_score_target,
                maximum_repetition_ratio=event.maximum_repetition_ratio_target,
            )
            if focal_return_time is None:
                focal_censored += 1
                focal_return_time = len(event.recovery_turn_indexes) + 1
                focal_unit_censored = True
            else:
                focal_unit_censored = False
            task_difference = min(
                focal_task_change - comparator_change
                for comparator_change in comparator_task_changes
            )
            repetition_difference = min(
                focal_repetition_change - comparator_change
                for comparator_change in comparator_repetition_changes
            )
            return_time_difference = min(
                comparator_time - float(focal_return_time)
                for comparator_time in comparator_return_times
            )
            task_differences.append(task_difference)
            repetition_differences.append(repetition_difference)
            return_time_differences.append(return_time_difference)
            unit_outcomes.append(
                RecoveryUnitOutcome(
                    unit_key=MatchedUnitKey(
                        prompt_sequence_id=event.prompt_sequence_id,
                        model_seed=model_seed,
                    ),
                    post_stressor_task_score_change=task_difference,
                    post_stressor_repetition_change=repetition_difference,
                    time_to_return_to_target_band=return_time_difference,
                    focal_right_censored=focal_unit_censored,
                    comparator_right_censored_count=unit_comparator_censored,
                )
            )
    if perform_statistics:
        result = evaluate_recovery(
            {
                RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: tuple(task_differences),
                RecoveryMetricName.POST_STRESSOR_REPETITION_CHANGE: tuple(repetition_differences),
                RecoveryMetricName.TIME_TO_RETURN_TO_TARGET_BAND: tuple(return_time_differences),
            },
            spec=spec.recovery,
            negative_multiplicity=spec.validated_negative_multiplicity,
            right_censored_focal_units=focal_censored,
            right_censored_comparator_units=comparator_censored,
        )
    else:
        result = RecoveryEvaluationResult(
            status=ScientificEvidenceStatus.INVALID,
            detail="global integrity evidence failed before recovery statistics",
            right_censored_focal_units=focal_censored,
            right_censored_comparator_units=comparator_censored,
        )
    return result, tuple(unit_outcomes), focal_censored, comparator_censored


def _prompt_family_by_sequence(plan: ExperimentPlan) -> dict[str, str]:
    families: dict[str, set[str]] = defaultdict(set)
    for turn in plan.turns:
        families[turn.condition.prompt_sequence_id].add(turn.prompt_family)
    if any(len(values) != 1 for values in families.values()):
        raise ValueError("prompt_family subgroup requires one family per sequence unit")
    return {sequence_id: next(iter(values)) for sequence_id, values in families.items()}


def build_confirmatory_turn_input_evidence_sha256(plan: ExperimentPlan) -> str:
    """Hash every frozen prompt-side input for independent durable reconstruction."""

    _require_confirmatory_plan(plan)
    evidence = tuple(
        sorted(
            (
                TurnInputEvidence(
                    condition_id=turn.condition.condition_id,
                    prompt_case_id=turn.prompt_case_id,
                    prompt_family=turn.prompt_family,
                    prompt_features=turn.prompt_features,
                    validator=turn.validator,
                )
                for turn in plan.turns
            ),
            key=lambda item: item.condition_id,
        )
    )
    return canonical_sha256(evidence)


def _subgroup_limitations(
    plan: ExperimentPlan,
    outcomes: Mapping[str, Mapping[tuple[str, int], SequencePolicyOutcome]],
) -> tuple[tuple[ScientificLimitation, ...], tuple[SubgroupEffectResult, ...]]:
    assert plan.confirmatory_analysis is not None
    family_by_sequence = _prompt_family_by_sequence(plan)
    spec = plan.confirmatory_analysis.efficacy
    limitations: list[ScientificLimitation] = []
    effects: list[SubgroupEffectResult] = []
    for comparator_policy_id in SERIOUS_COMPARATOR_IDS:
        focal = outcomes[FOCAL_POLICY_ID]
        comparator = outcomes[comparator_policy_id]
        grouped: dict[str, list[float]] = defaultdict(list)
        for key in sorted(focal):
            grouped[family_by_sequence[key[0]]].append(
                focal[key].task_score - comparator[key].task_score
            )
        if len(grouped) < 2:
            continue
        directions: set[str] = set()
        for field_value, differences in sorted(grouped.items()):
            bootstrap = paired_bootstrap_ci(
                tuple(differences),
                resamples=spec.bootstrap_resamples,
                confidence_level=spec.confidence_level,
                seed=spec.bootstrap_seed,
            )
            direction: Literal["beneficial", "harmful", "unresolved"]
            if bootstrap.estimate >= spec.practical_effect_threshold and bootstrap.lower > 0.0:
                direction = "beneficial"
            elif bootstrap.upper < 0.0:
                direction = "harmful"
            else:
                direction = "unresolved"
            effects.append(
                SubgroupEffectResult(
                    field_value=field_value,
                    comparator_policy_id=comparator_policy_id,
                    unit_count=len(differences),
                    bootstrap=bootstrap,
                    practical_effect_threshold=spec.practical_effect_threshold,
                    direction=direction,
                )
            )
            if direction != "unresolved":
                directions.add(direction)
        if directions == {"beneficial", "harmful"}:
            limitations.append(
                ScientificLimitation(
                    kind=LimitationKind.SUBGROUP_CONFLICT,
                    code=f"prompt_family_conflict_{comparator_policy_id}",
                    detail=(
                        "prompt_family subgroups contain resolved beneficial and harmful "
                        f"effects against {comparator_policy_id}"
                    ),
                    disposition=LimitationDisposition.INCONCLUSIVE,
                )
            )
    return tuple(limitations), tuple(effects)


def _optional_metric_limitations(
    spec: ConfirmatoryAnalysisSpec,
    availability: Mapping[str, tuple[int, int]],
) -> tuple[ScientificLimitation, ...]:
    if set(availability) != set(spec.optional_metric_dispositions):
        raise ValueError("optional metric availability must match the frozen disposition set")
    limitations: list[ScientificLimitation] = []
    for metric_name, disposition in spec.optional_metric_dispositions.items():
        available, total = availability[metric_name]
        if (
            not isinstance(available, int)
            or isinstance(available, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total < 1
            or not 0 <= available <= total
        ):
            raise ValueError("optional metric availability counts are invalid")
        if available == total:
            continue
        limitations.append(
            ScientificLimitation(
                kind=LimitationKind.OPTIONAL_METRIC_UNAVAILABLE,
                code=f"optional_metric_unavailable_{metric_name}",
                detail=f"{metric_name} available for {available} of {total} committed turns",
                disposition=disposition,
            )
        )
    return tuple(limitations)


def _optional_metric_availability_from_turns(
    turns: Sequence[StoredTurn],
    spec: ConfirmatoryAnalysisSpec,
) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    total = len(turns)
    for metric_name in spec.optional_metric_dispositions:
        available = 0
        for turn in turns:
            metric = None if turn.metrics is None else getattr(turn.metrics, metric_name, None)
            available += bool(metric is not None and metric.availability)
        counts[metric_name] = available, total
    return counts


def _analyze_records(
    plan: ExperimentPlan,
    records: tuple[TurnEvaluationRecord, ...],
    *,
    optional_metric_availability: Mapping[str, tuple[int, int]],
    causal_mechanism_validated: bool,
    claim_eligible: bool,
    run_manifest_sha256: str | None,
    run_finalization_sha256: str | None,
) -> tuple[ConfirmatoryEvaluationResult, ConfirmatoryAnalysisContext]:
    spec = _require_confirmatory_plan(plan)
    if not causal_mechanism_validated:
        raise ValueError("persistent-state attribution requires validated causal evidence")
    design = build_evaluation_design(plan)
    coverage = validate_exact_coverage(records, design)
    if not coverage.exact:
        raise ValueError("confirmatory analysis requires exact full five-arm coverage")

    efficacy_records = tuple(
        record for record in records if record.policy_id in EFFICACY_POLICY_IDS
    )
    efficacy_outcomes = aggregate_matched_units(efficacy_records)
    outcomes_by_policy = _outcomes_by_policy(efficacy_outcomes)
    if set(outcomes_by_policy) != set(EFFICACY_POLICY_IDS):
        raise ValueError("efficacy aggregation must contain exactly four independent arms")
    assert plan.evaluation is not None
    final_guardrails = _scientific_guardrails_from_records(records, design, plan.evaluation)
    scientific_global = tuple(
        guardrail for guardrail in final_guardrails if guardrail.scope == "efficacy:global"
    )
    scientific_saturation = next(
        guardrail
        for guardrail in final_guardrails
        if guardrail.name == GuardrailName.ACTION_SATURATION_RATE.value
        and guardrail.scope == f"efficacy:policy:{FOCAL_POLICY_ID}"
    )
    scientific_pairs: dict[str, tuple[ScientificGuardrailResult, ...]] = {
        comparator_policy_id: tuple(
            guardrail
            for guardrail in final_guardrails
            if guardrail.scope == f"efficacy:pair:{FOCAL_POLICY_ID}:{comparator_policy_id}"
        )
        for comparator_policy_id in EFFICACY_COMPARATOR_IDS
    }
    integrity_invalid = any(
        guardrail.status is ScientificGuardrailStatus.INVALID for guardrail in scientific_global
    )
    prompt_family_by_sequence = _prompt_family_by_sequence(plan)
    unit_outcomes = _scientific_unit_outcomes(
        efficacy_outcomes,
        prompt_family_by_sequence=prompt_family_by_sequence,
        global_guardrails=scientific_global,
        saturation_guardrail=scientific_saturation,
        pair_guardrails=scientific_pairs,
    )
    comparison_guardrails: dict[str, tuple[ScientificGuardrailResult, ...]] = {
        comparator_policy_id: (
            *scientific_global,
            scientific_saturation,
            *scientific_pairs[comparator_policy_id],
        )
        for comparator_policy_id in EFFICACY_COMPARATOR_IDS
    }
    aliases: dict[str, bool] = {
        comparator_policy_id: any(
            guardrail.name == GuardrailName.BEHAVIORAL_ALIAS_DETECTION.value
            and guardrail.status is ScientificGuardrailStatus.FAIL
            for guardrail in scientific_pairs[comparator_policy_id]
        )
        for comparator_policy_id in EFFICACY_COMPARATOR_IDS
    }
    efficacy = evaluate_efficacy_comparisons(
        {
            comparator_policy_id: _paired_task_differences(
                outcomes_by_policy,
                comparator_policy_id,
            )
            for comparator_policy_id in EFFICACY_COMPARATOR_IDS
        },
        spec=spec.efficacy,
        negative_multiplicity=spec.validated_negative_multiplicity,
        guardrails_by_comparator=comparison_guardrails,
        behavioral_alias_by_comparator=aliases,
    )

    (
        recovery,
        recovery_unit_outcomes,
        focal_censored,
        comparator_censored,
    ) = _recovery_evidence(
        records,
        spec,
        perform_statistics=not integrity_invalid,
    )
    attribution_records = tuple(
        record
        for record in records
        if record.turn_index > 0
        and record.policy_id in {FOCAL_POLICY_ID, ATTRIBUTION_COMPARATOR_ID}
    )
    attribution_outcomes = aggregate_matched_units(attribution_records)
    attribution_by_policy = _outcomes_by_policy(attribution_outcomes)
    if set(attribution_by_policy) != {FOCAL_POLICY_ID, ATTRIBUTION_COMPARATOR_ID}:
        raise ValueError("attribution requires exact persistent/reset intervention units")
    causal_guardrails = tuple(
        guardrail for guardrail in final_guardrails if guardrail.scope.startswith("attribution:")
    )
    attribution_focal = attribution_by_policy[FOCAL_POLICY_ID]
    attribution_comparator = attribution_by_policy[ATTRIBUTION_COMPARATOR_ID]
    if attribution_focal.keys() != attribution_comparator.keys():
        raise ValueError("attribution outcomes lack exact persistent/reset matched-unit keys")
    attribution_unit_outcomes = tuple(
        AttributionUnitOutcome(
            unit_key=MatchedUnitKey(
                prompt_sequence_id=key[0],
                model_seed=key[1],
            ),
            persistent_minus_reset_task_score=(
                attribution_focal[key].task_score - attribution_comparator[key].task_score
            ),
        )
        for key in sorted(attribution_focal)
    )
    attribution_differences = tuple(
        outcome.persistent_minus_reset_task_score for outcome in attribution_unit_outcomes
    )
    if integrity_invalid:
        attribution = PersistentStateAttributionResult(
            unit_count=len(attribution_differences),
            practical_effect_threshold=spec.attribution.practical_effect_threshold,
            causal_guardrails=causal_guardrails,
            status=ScientificEvidenceStatus.INVALID,
            detail="global integrity evidence failed before attribution statistics",
        )
    else:
        attribution = evaluate_persistent_state_attribution(
            attribution_differences,
            spec=spec.attribution,
            negative_multiplicity=spec.validated_negative_multiplicity,
            causal_guardrails=causal_guardrails,
            behavioral_alias=(
                max(abs(difference) for difference in attribution_differences)
                <= plan.evaluation.behavioral_alias_tolerance
            ),
        )

    limitations = list(_optional_metric_limitations(spec, optional_metric_availability))
    if focal_censored or comparator_censored:
        limitations.append(
            ScientificLimitation(
                kind=LimitationKind.OTHER,
                code="recovery_right_censoring",
                detail=(
                    f"right-censored window+1 units: focal={focal_censored}, "
                    f"serious_comparator={comparator_censored}"
                ),
                disposition=LimitationDisposition.DISCLOSURE_ONLY,
            )
        )
    if integrity_invalid:
        subgroup_limitations: tuple[ScientificLimitation, ...] = ()
        subgroup_effects: tuple[SubgroupEffectResult, ...] = ()
    else:
        subgroup_limitations, subgroup_effects = _subgroup_limitations(
            plan,
            outcomes_by_policy,
        )
    limitations.extend(subgroup_limitations)
    decision_input = ScientificDecisionInput(
        tier=ExperimentTier.CONFIRMATORY,
        efficacy_comparisons=efficacy,
        recovery=recovery.decision_gate,
        attribution=attribution.decision_gate,
        guardrails=final_guardrails,
        limitations=tuple(limitations),
    )
    decision = decide_scientific_outcome(decision_input)
    analysis_contract_sha256 = build_confirmatory_analysis_contract_sha256(plan)
    input_sha256 = canonical_sha256(
        {
            "schema_version": 2,
            "implementation_version": "confirmatory-analysis-input-v2",
            "analysis_contract_sha256": analysis_contract_sha256,
            "records": tuple(sorted(records, key=record_sort_key)),
            "optional_metric_availability": dict(sorted(optional_metric_availability.items())),
            "causal_mechanism_validated": causal_mechanism_validated,
            "claim_eligible": claim_eligible,
            "run_manifest_sha256": run_manifest_sha256,
            "run_finalization_sha256": run_finalization_sha256,
        }
    )
    statistics_call_count = confirmatory_statistics_call_count(
        efficacy,
        recovery,
        attribution,
        subgroup_effects,
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "implementation_version": "confirmatory-evaluation-v2",
        "claim_scope": "confirmatory-model-backed-scientific-decision",
        "analysis_contract_sha256": analysis_contract_sha256,
        "confirmatory_analysis_spec": spec,
        "confirmatory_analysis_spec_sha256": canonical_sha256(spec),
        "prompt_family_by_sequence": prompt_family_by_sequence,
        "prompt_family_design_sha256": canonical_sha256(prompt_family_by_sequence),
        "validated_negative_multiplicity_sha256": canonical_sha256(
            spec.validated_negative_multiplicity
        ),
        "causal_mechanism_validated": causal_mechanism_validated,
        "claim_eligible": claim_eligible,
        "run_manifest_sha256": run_manifest_sha256,
        "run_finalization_sha256": run_finalization_sha256,
        "input_sha256": input_sha256,
        "coverage": coverage,
        "optional_metric_availability": dict(sorted(optional_metric_availability.items())),
        "unit_outcomes": unit_outcomes,
        "recovery_unit_outcomes": recovery_unit_outcomes,
        "attribution_unit_outcomes": attribution_unit_outcomes,
        "efficacy_comparisons": efficacy,
        "recovery": recovery,
        "attribution": attribution,
        "subgroup_effects": subgroup_effects,
        "guardrails": final_guardrails,
        "limitations": tuple(limitations),
        "decision": decision,
        "statistics_call_count": statistics_call_count,
    }
    result = ConfirmatoryEvaluationResult.model_validate(
        {
            **payload,
            "result_sha256": confirmatory_result_sha256(payload),
        }
    )
    context = ConfirmatoryAnalysisContext(
        analysis_contract_sha256=analysis_contract_sha256,
        evaluation_input_sha256=input_sha256,
        causal_mechanism_validated=causal_mechanism_validated,
        claim_eligible=claim_eligible,
        run_manifest_sha256=run_manifest_sha256,
        run_finalization_sha256=run_finalization_sha256,
    )
    return result, context


def analyze_confirmatory_records(
    plan: ExperimentPlan,
    records: Sequence[TurnEvaluationRecord],
    *,
    optional_metric_availability: Mapping[str, tuple[int, int]],
    causal_mechanism_validated: bool,
) -> tuple[ConfirmatoryEvaluationResult, ConfirmatoryAnalysisContext]:
    """Assemble provider-free structural evidence that is never claim-eligible."""

    return _analyze_records(
        plan,
        tuple(records),
        optional_metric_availability=optional_metric_availability,
        causal_mechanism_validated=causal_mechanism_validated,
        claim_eligible=False,
        run_manifest_sha256=None,
        run_finalization_sha256=None,
    )


def analyze_closed_confirmatory_run(
    plan: ExperimentPlan,
    database_path: Path,
) -> tuple[ConfirmatoryEvaluationResult, ConfirmatoryAnalysisContext]:
    """Read, validate, and analyze one real finalized llama.cpp confirmatory store."""

    spec = _require_confirmatory_plan(plan)
    if not isinstance(database_path, Path):
        raise TypeError("database_path must be a pathlib.Path")
    if plan.provider_identity.provider_type != "llama_cpp":
        raise ValueError("real confirmatory scientific analysis requires llama_cpp evidence")
    with SQLiteRunStore(database_path) as store:
        store.verify_integrity()
        manifest = store.get_manifest()
        finalization = store.get_finalization()
        if manifest is None or finalization is None:
            raise StoreInvariantError("scientific analysis requires a manifest-bound closed run")
        _validate_manifest(plan, manifest)
        _validate_finalization(plan, manifest, finalization)
        turns = store.list_turns()
        try:
            recomputed_scientific_result_sha256 = scientific_result_sha256(turns)
        except ValueError as exc:
            raise StoreInvariantError(
                "confirmatory run contains incomplete aggregate scientific evidence"
            ) from exc
        if recomputed_scientific_result_sha256 != finalization.scientific_result_sha256:
            raise StoreInvariantError(
                "confirmatory finalization does not match the committed scientific result"
            )
        turn_inputs = store.list_turn_inputs()
        from neurallm.reporting.artifacts import _validate_phase4_mechanism_evidence

        try:
            _validate_phase4_mechanism_evidence(manifest, turns, turn_inputs)
        except ValueError as exc:
            raise StoreInvariantError(
                "confirmatory persistent/reset causal mechanism evidence is invalid"
            ) from exc
        records = evaluation_records_from_store(plan, store)
        optional_availability = _optional_metric_availability_from_turns(turns, spec)
        store.verify_integrity()
        manifest_sha256 = canonical_sha256(manifest)
        finalization_sha256 = canonical_sha256(finalization)
    return _analyze_records(
        plan,
        records,
        optional_metric_availability=optional_availability,
        causal_mechanism_validated=True,
        claim_eligible=True,
        run_manifest_sha256=manifest_sha256,
        run_finalization_sha256=finalization_sha256,
    )


__all__ = [
    "ConfirmatoryAnalysisContext",
    "analyze_closed_confirmatory_run",
    "analyze_confirmatory_records",
    "build_confirmatory_analysis_contract_sha256",
    "build_confirmatory_turn_input_evidence_sha256",
    "confirmatory_analysis_contract_sha256",
]

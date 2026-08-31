"""Explicit integrity and substantive guardrails for Phase 3."""

from __future__ import annotations

from collections.abc import Sequence

from neurallm.evaluation.models import (
    CoverageResult,
    ExpectedEvaluationDesign,
    GuardrailName,
    GuardrailResult,
    GuardrailStatus,
    SequencePolicyOutcome,
    TurnEvaluationRecord,
)


def integrity_guardrails(
    records: Sequence[TurnEvaluationRecord],
    design: ExpectedEvaluationDesign,
    coverage: CoverageResult,
) -> tuple[GuardrailResult, ...]:
    """Evaluate fail-closed conditions that must pass before statistics."""

    dataset_identity_stable = all(
        record.dataset_sha256 == design.dataset_sha256 for record in records
    )
    matched_coverage_valid = coverage.exact and dataset_identity_stable
    coverage_result = GuardrailResult(
        name=GuardrailName.MATCHED_CONDITION_COVERAGE,
        status=(GuardrailStatus.PASS if matched_coverage_valid else GuardrailStatus.INVALID),
        observed_value=(coverage.observed_count / coverage.expected_count),
        threshold=1.0,
        detail=(
            "observed keys exactly match the frozen condition grid"
            if matched_coverage_valid
            else (
                f"expected={coverage.expected_count}, observed={coverage.observed_count}, "
                f"missing={len(coverage.missing_keys)}, "
                f"unexpected={len(coverage.unexpected_keys)}, "
                f"duplicates={len(coverage.duplicate_keys)}, "
                f"dataset_identity_stable={dataset_identity_stable}"
            )
        ),
    )

    provider_ids = {record.provider_identity_id for record in records}
    provider_stable = provider_ids == {design.provider_identity_id}
    provider_result = GuardrailResult(
        name=GuardrailName.PROVIDER_IDENTITY_STABILITY,
        status=GuardrailStatus.PASS if provider_stable else GuardrailStatus.INVALID,
        observed_value=float(len(provider_ids)),
        threshold=1.0,
        detail=(
            "every record matches the frozen provider identity"
            if provider_stable
            else "record provider identities do not exactly match the frozen identity"
        ),
    )

    turn_zero_records = tuple(record for record in records if record.turn_index == 0)
    turn_zero_valid = bool(turn_zero_records) and all(
        not record.has_previous_response and record.previous_history_commitment_sha256 is None
        for record in turn_zero_records
    )
    turn_zero_result = GuardrailResult(
        name=GuardrailName.TURN_ZERO_EQUIVALENCE,
        status=GuardrailStatus.PASS if turn_zero_valid else GuardrailStatus.INVALID,
        observed_value=float(
            sum(
                not record.has_previous_response
                and record.previous_history_commitment_sha256 is None
                for record in turn_zero_records
            )
        ),
        threshold=float(len(turn_zero_records)),
        detail=(
            "all turn-zero conditions have explicit null/false history"
            if turn_zero_valid
            else "turn-zero history semantics are absent or non-equivalent"
        ),
    )

    bounds_valid = all(record.action_within_bounds for record in records)
    bounds_result = GuardrailResult(
        name=GuardrailName.ACTION_BOUND_COMPLIANCE,
        status=GuardrailStatus.PASS if bounds_valid else GuardrailStatus.INVALID,
        observed_value=float(sum(record.action_within_bounds for record in records)),
        threshold=float(len(records)),
        detail=(
            "every recorded action is within the frozen bounds"
            if bounds_valid
            else "one or more recorded actions exceed the frozen bounds"
        ),
    )

    available_count = sum(record.required_metrics_available for record in records)
    metrics_valid = available_count == len(records)
    metrics_result = GuardrailResult(
        name=GuardrailName.METRIC_AVAILABILITY,
        status=GuardrailStatus.PASS if metrics_valid else GuardrailStatus.INVALID,
        observed_value=float(available_count),
        threshold=float(len(records)),
        detail=(
            "all required evaluator metrics are available"
            if metrics_valid
            else "one or more required evaluator metrics are unavailable"
        ),
    )

    return (
        coverage_result,
        provider_result,
        turn_zero_result,
        bounds_result,
        metrics_result,
    )


def action_saturation_guardrail(
    records: Sequence[TurnEvaluationRecord],
    *,
    focal_policy_id: str,
    maximum_rate: float,
) -> GuardrailResult:
    """Gate excessive focal-policy action saturation."""

    focal_records = tuple(record for record in records if record.policy_id == focal_policy_id)
    if not focal_records:
        return GuardrailResult(
            name=GuardrailName.ACTION_SATURATION_RATE,
            status=GuardrailStatus.INVALID,
            policy_id=focal_policy_id,
            detail="the focal policy has no records",
        )
    saturation_rate = sum(record.action_saturated for record in focal_records) / len(focal_records)
    passed = saturation_rate <= maximum_rate
    return GuardrailResult(
        name=GuardrailName.ACTION_SATURATION_RATE,
        status=GuardrailStatus.PASS if passed else GuardrailStatus.FAIL,
        policy_id=focal_policy_id,
        observed_value=saturation_rate,
        threshold=maximum_rate,
        detail=(
            "focal action saturation is within the preregistered limit"
            if passed
            else "focal action saturation exceeds the preregistered limit"
        ),
    )


def _ordered_pairs(
    focal: Sequence[SequencePolicyOutcome],
    comparator: Sequence[SequencePolicyOutcome],
) -> tuple[tuple[SequencePolicyOutcome, SequencePolicyOutcome], ...]:
    focal_by_key = {
        (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed): outcome
        for outcome in focal
    }
    comparator_by_key = {
        (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed): outcome
        for outcome in comparator
    }
    if focal_by_key.keys() != comparator_by_key.keys():
        raise ValueError("pairwise guardrails require identical matched-unit keys")
    return tuple((focal_by_key[key], comparator_by_key[key]) for key in sorted(focal_by_key))


def pairwise_guardrails(
    focal: Sequence[SequencePolicyOutcome],
    comparator: Sequence[SequencePolicyOutcome],
    *,
    focal_policy_id: str,
    comparator_policy_id: str,
    maximum_adherence_regression: float,
    maximum_length_reduction_ratio: float,
    behavioral_alias_tolerance: float,
) -> tuple[GuardrailResult, ...]:
    """Evaluate matched adherence, length-confound, and alias checks."""

    pairs = _ordered_pairs(focal, comparator)
    adherence_difference = sum(
        focal_outcome.instruction_adherence - comparator_outcome.instruction_adherence
        for focal_outcome, comparator_outcome in pairs
    ) / len(pairs)
    adherence_passed = adherence_difference >= -maximum_adherence_regression
    adherence = GuardrailResult(
        name=GuardrailName.INSTRUCTION_ADHERENCE_NON_REGRESSION,
        status=GuardrailStatus.PASS if adherence_passed else GuardrailStatus.FAIL,
        policy_id=focal_policy_id,
        comparator_policy_id=comparator_policy_id,
        observed_value=adherence_difference,
        threshold=-maximum_adherence_regression,
        detail=(
            "matched adherence is within the non-regression margin"
            if adherence_passed
            else "matched adherence regresses beyond the preregistered margin"
        ),
    )

    improving_length_reductions: list[float] = []
    for focal_outcome, comparator_outcome in pairs:
        repetition_improvement = (
            comparator_outcome.repetition_ratio - focal_outcome.repetition_ratio
        )
        if repetition_improvement <= 0.0:
            continue
        comparator_length = comparator_outcome.response_length_tokens
        length_reduction_ratio = (
            max(
                0.0,
                (comparator_length - focal_outcome.response_length_tokens) / comparator_length,
            )
            if comparator_length > 0.0
            else 0.0
        )
        improving_length_reductions.append(length_reduction_ratio)
    maximum_improving_unit_length_reduction = max(improving_length_reductions, default=0.0)
    length_confounded = maximum_improving_unit_length_reduction > maximum_length_reduction_ratio
    length = GuardrailResult(
        name=GuardrailName.RESPONSE_LENGTH_CONFOUND,
        status=GuardrailStatus.FAIL if length_confounded else GuardrailStatus.PASS,
        policy_id=focal_policy_id,
        comparator_policy_id=comparator_policy_id,
        observed_value=maximum_improving_unit_length_reduction,
        threshold=maximum_length_reduction_ratio,
        detail=(
            "a matched unit with repetition improvement has excessive output shortening"
            if length_confounded
            else "no repetition-improving matched unit exceeds the shortening limit"
        ),
    )

    maximum_behavior_difference = max(
        max(
            abs(focal_outcome.task_score - comparator_outcome.task_score),
            abs(focal_outcome.instruction_adherence - comparator_outcome.instruction_adherence),
            abs(focal_outcome.response_length_tokens - comparator_outcome.response_length_tokens),
            abs(focal_outcome.repetition_ratio - comparator_outcome.repetition_ratio),
            abs(focal_outcome.action_magnitude - comparator_outcome.action_magnitude),
            abs(focal_outcome.action_saturation_rate - comparator_outcome.action_saturation_rate),
        )
        for focal_outcome, comparator_outcome in pairs
    )
    alias = maximum_behavior_difference <= behavioral_alias_tolerance
    alias_result = GuardrailResult(
        name=GuardrailName.BEHAVIORAL_ALIAS_DETECTION,
        status=GuardrailStatus.FAIL if alias else GuardrailStatus.PASS,
        policy_id=focal_policy_id,
        comparator_policy_id=comparator_policy_id,
        observed_value=maximum_behavior_difference,
        threshold=behavioral_alias_tolerance,
        detail=(
            "focal and comparator behavior are aliased within tolerance"
            if alias
            else "focal and comparator behavior are distinguishable"
        ),
    )
    return adherence, length, alias_result


def has_invalid_guardrail(results: Sequence[GuardrailResult]) -> bool:
    """Return whether any integrity guardrail invalidates evaluation."""

    return any(result.status is GuardrailStatus.INVALID for result in results)


__all__ = [
    "action_saturation_guardrail",
    "has_invalid_guardrail",
    "integrity_guardrails",
    "pairwise_guardrails",
]

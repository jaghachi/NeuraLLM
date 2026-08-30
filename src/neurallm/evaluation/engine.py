"""Pure Phase 3 evaluator: integrity, aggregation, paired statistics, decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from neurallm.evaluation.aggregation import (
    aggregate_matched_units,
    evaluation_input_sha256,
    validate_exact_coverage,
)
from neurallm.evaluation.guardrails import (
    action_saturation_guardrail,
    has_invalid_guardrail,
    integrity_guardrails,
    pairwise_guardrails,
)
from neurallm.evaluation.models import (
    BootstrapResult,
    CoverageResult,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    GuardrailName,
    GuardrailResult,
    GuardrailStatus,
    HolmAdjustedPValue,
    PairwiseComparisonResult,
    PermutationTestResult,
    Phase3EvaluationResult,
    Phase3Verdict,
    SequencePolicyOutcome,
    TurnEvaluationRecord,
    phase3_result_sha256,
)
from neurallm.evaluation.statistics import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_permutation_test,
)


@dataclass(frozen=True)
class _PairDraft:
    comparator_policy_id: str
    serious_comparator: bool
    differences: tuple[float, ...]
    bootstrap: BootstrapResult
    permutation: PermutationTestResult
    behavioral_alias: bool
    guardrails: tuple[GuardrailResult, ...]


def _build_result(
    *,
    verdict: Phase3Verdict,
    input_sha256: str,
    coverage: CoverageResult,
    outcomes: tuple[SequencePolicyOutcome, ...],
    global_guardrails: tuple[GuardrailResult, ...],
    comparisons: tuple[PairwiseComparisonResult, ...],
    statistics_call_count: int,
) -> Phase3EvaluationResult:
    statistics_computed = statistics_call_count > 0
    payload: dict[str, object] = {
        "schema_version": 1,
        "implementation_version": "phase3-evaluator-v1",
        "claim_scope": "phase-3-statistical-behavior-only",
        "verdict": verdict,
        "input_sha256": input_sha256,
        "coverage": coverage,
        "outcomes": outcomes,
        "global_guardrails": global_guardrails,
        "comparisons": comparisons,
        "statistics_computed": statistics_computed,
        "statistics_call_count": statistics_call_count,
    }
    return Phase3EvaluationResult(
        verdict=verdict,
        input_sha256=input_sha256,
        result_sha256=phase3_result_sha256(payload),
        coverage=coverage,
        outcomes=outcomes,
        global_guardrails=global_guardrails,
        comparisons=comparisons,
        statistics_computed=statistics_computed,
        statistics_call_count=statistics_call_count,
    )


def _policy_set_guardrails(
    design: ExpectedEvaluationDesign,
    spec: EvaluationSpec,
    guardrails: tuple[GuardrailResult, ...],
) -> tuple[GuardrailResult, ...]:
    expected_policy_ids = {
        spec.focal_policy_id,
        *spec.required_serious_comparator_ids,
        *spec.negative_control_policy_ids,
    }
    if set(design.policy_ids) == expected_policy_ids:
        return guardrails
    replacement = GuardrailResult(
        name=GuardrailName.MATCHED_CONDITION_COVERAGE,
        status=GuardrailStatus.INVALID,
        observed_value=float(len(design.policy_ids)),
        threshold=float(len(expected_policy_ids)),
        detail="frozen design policy set does not exactly match the evaluation specification",
    )
    return (replacement, *guardrails[1:])


def _outcomes_for_policy(
    outcomes: Sequence[SequencePolicyOutcome],
    policy_id: str,
) -> tuple[SequencePolicyOutcome, ...]:
    return tuple(outcome for outcome in outcomes if outcome.policy_id == policy_id)


def _paired_differences(
    focal: Sequence[SequencePolicyOutcome],
    comparator: Sequence[SequencePolicyOutcome],
) -> tuple[float, ...]:
    focal_by_key = {
        (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed): outcome
        for outcome in focal
    }
    comparator_by_key = {
        (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed): outcome
        for outcome in comparator
    }
    if focal_by_key.keys() != comparator_by_key.keys() or not focal_by_key:
        raise ValueError("comparison policies do not have identical matched units")
    return tuple(
        focal_by_key[key].task_score - comparator_by_key[key].task_score
        for key in sorted(focal_by_key)
    )


def _pair_verdict(
    draft: _PairDraft,
    holm: HolmAdjustedPValue | None,
    spec: EvaluationSpec,
    *,
    focal_saturation_failed: bool,
) -> Phase3Verdict:
    substantive_failure = focal_saturation_failed or any(
        result.status is GuardrailStatus.FAIL
        and result.name is not GuardrailName.BEHAVIORAL_ALIAS_DETECTION
        for result in draft.guardrails
    )
    if substantive_failure:
        return Phase3Verdict.INFERIOR

    bootstrap = draft.bootstrap
    if draft.behavioral_alias or (
        bootstrap.lower >= -spec.equivalence_margin and bootstrap.upper <= spec.equivalence_margin
    ):
        return Phase3Verdict.EQUIVALENT

    p_value = holm.adjusted_p_value if holm is not None else draft.permutation.p_value
    alpha = 1.0 - spec.confidence_level
    if (
        bootstrap.estimate >= spec.practical_effect_threshold
        and bootstrap.lower > 0.0
        and p_value <= alpha
    ):
        return Phase3Verdict.SUPERIOR
    if (
        bootstrap.estimate <= -spec.practical_effect_threshold
        and bootstrap.upper < 0.0
        and p_value <= alpha
    ):
        return Phase3Verdict.INFERIOR
    return Phase3Verdict.INCONCLUSIVE


def _overall_verdict(
    comparisons: Sequence[PairwiseComparisonResult],
    *,
    focal_saturation_failed: bool,
) -> Phase3Verdict:
    serious = tuple(comparison for comparison in comparisons if comparison.serious_comparator)
    if focal_saturation_failed or any(
        comparison.verdict is Phase3Verdict.INFERIOR for comparison in serious
    ):
        return Phase3Verdict.INFERIOR
    if serious and all(comparison.verdict is Phase3Verdict.SUPERIOR for comparison in serious):
        return Phase3Verdict.SUPERIOR
    if serious and all(comparison.verdict is Phase3Verdict.EQUIVALENT for comparison in serious):
        return Phase3Verdict.EQUIVALENT
    return Phase3Verdict.INCONCLUSIVE


def evaluate_phase3(
    records: Sequence[TurnEvaluationRecord],
    *,
    design: ExpectedEvaluationDesign,
    spec: EvaluationSpec,
) -> Phase3EvaluationResult:
    """Evaluate immutable synthetic/run evidence under the Phase 3 contract.

    Integrity and exact coverage are resolved before either statistical routine
    is called.  An invalid result therefore records zero statistical calls.
    """

    input_sha256 = evaluation_input_sha256(records, design, spec)
    coverage = validate_exact_coverage(records, design)
    global_guardrails = _policy_set_guardrails(
        design,
        spec,
        integrity_guardrails(records, design, coverage),
    )
    if has_invalid_guardrail(global_guardrails):
        return _build_result(
            verdict=Phase3Verdict.INVALID,
            input_sha256=input_sha256,
            coverage=coverage,
            outcomes=(),
            global_guardrails=global_guardrails,
            comparisons=(),
            statistics_call_count=0,
        )

    outcomes = aggregate_matched_units(records)
    focal_outcomes = _outcomes_for_policy(outcomes, spec.focal_policy_id)
    saturation = action_saturation_guardrail(
        records,
        focal_policy_id=spec.focal_policy_id,
        maximum_rate=spec.maximum_action_saturation_rate,
    )
    global_guardrails = (*global_guardrails, saturation)
    if saturation.status is GuardrailStatus.INVALID:
        return _build_result(
            verdict=Phase3Verdict.INVALID,
            input_sha256=input_sha256,
            coverage=coverage,
            outcomes=outcomes,
            global_guardrails=global_guardrails,
            comparisons=(),
            statistics_call_count=0,
        )

    serious_ids = set(spec.required_serious_comparator_ids)
    comparator_ids = tuple(sorted(serious_ids | set(spec.negative_control_policy_ids)))
    drafts: list[_PairDraft] = []
    statistics_call_count = 0
    for comparator_id in comparator_ids:
        comparator_outcomes = _outcomes_for_policy(outcomes, comparator_id)
        differences = _paired_differences(focal_outcomes, comparator_outcomes)
        bootstrap = paired_bootstrap_ci(
            differences,
            resamples=spec.bootstrap_resamples,
            confidence_level=spec.confidence_level,
            seed=spec.bootstrap_seed,
        )
        statistics_call_count += 1
        permutation = paired_sign_flip_permutation_test(
            differences,
            resamples=spec.permutation_resamples,
            seed=spec.permutation_seed,
        )
        statistics_call_count += 1
        guardrails = pairwise_guardrails(
            focal_outcomes,
            comparator_outcomes,
            focal_policy_id=spec.focal_policy_id,
            comparator_policy_id=comparator_id,
            maximum_adherence_regression=spec.maximum_adherence_regression,
            maximum_length_reduction_ratio=spec.maximum_length_reduction_ratio,
            behavioral_alias_tolerance=spec.behavioral_alias_tolerance,
        )
        alias = (
            next(
                result
                for result in guardrails
                if result.name is GuardrailName.BEHAVIORAL_ALIAS_DETECTION
            ).status
            is GuardrailStatus.FAIL
        )
        drafts.append(
            _PairDraft(
                comparator_policy_id=comparator_id,
                serious_comparator=comparator_id in serious_ids,
                differences=differences,
                bootstrap=bootstrap,
                permutation=permutation,
                behavioral_alias=alias,
                guardrails=guardrails,
            )
        )

    holm_by_comparator = {
        result.comparator_policy_id: result
        for result in holm_adjust(
            {
                draft.comparator_policy_id: draft.permutation.p_value
                for draft in drafts
                if draft.serious_comparator
            }
        )
    }
    focal_saturation_failed = saturation.status is GuardrailStatus.FAIL
    comparisons = tuple(
        PairwiseComparisonResult(
            comparator_policy_id=draft.comparator_policy_id,
            serious_comparator=draft.serious_comparator,
            unit_count=len(draft.differences),
            mean_difference=sum(draft.differences) / len(draft.differences),
            bootstrap=draft.bootstrap,
            permutation=draft.permutation,
            holm=holm_by_comparator.get(draft.comparator_policy_id),
            behavioral_alias=draft.behavioral_alias,
            guardrails=draft.guardrails,
            verdict=_pair_verdict(
                draft,
                holm_by_comparator.get(draft.comparator_policy_id),
                spec,
                focal_saturation_failed=focal_saturation_failed,
            ),
        )
        for draft in drafts
    )
    verdict = _overall_verdict(
        comparisons,
        focal_saturation_failed=focal_saturation_failed,
    )
    return _build_result(
        verdict=verdict,
        input_sha256=input_sha256,
        coverage=coverage,
        outcomes=outcomes,
        global_guardrails=global_guardrails,
        comparisons=comparisons,
        statistics_call_count=statistics_call_count,
    )


__all__ = ["evaluate_phase3"]

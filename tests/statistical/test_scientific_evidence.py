"""Golden boundaries for Phase 5 efficacy, recovery, and attribution evidence."""

from __future__ import annotations

from collections.abc import Sequence
from operator import setitem

import pytest
from pydantic import ValidationError

from neurallm.evaluation.attribution import (
    AttributionAnalysisSpec,
    PersistentStateAttributionResult,
    evaluate_persistent_state_attribution,
)
from neurallm.evaluation.recovery import (
    RecoveryAnalysisSpec,
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
    EfficacyAnalysisSpec,
    ScientificEvidenceStatus,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
    evaluate_efficacy_comparisons,
)


def _efficacy_spec() -> EfficacyAnalysisSpec:
    return EfficacyAnalysisSpec(
        practical_effect_threshold=0.02,
        bootstrap_resamples=256,
        bootstrap_seed=101,
        permutation_resamples=512,
        permutation_seed=202,
    )


def _recovery_spec() -> RecoveryAnalysisSpec:
    return RecoveryAnalysisSpec(
        practical_thresholds={metric: 0.02 for metric in RecoveryMetricName},
        bootstrap_resamples=256,
        bootstrap_seed=303,
    )


def _causal_guardrail(
    status: ScientificGuardrailStatus = ScientificGuardrailStatus.PASS,
) -> ScientificGuardrailResult:
    return ScientificGuardrailResult(
        name="matched_history_state_reset_isolation",
        status=status,
        scope="persistent_minus_reset",
        detail=f"matched-history isolation is {status.value}",
    )


def test_efficacy_uses_exact_roles_and_holm_only_for_serious_comparators() -> None:
    differences = (0.1,) * 21
    results = evaluate_efficacy_comparisons(
        {comparator: differences for comparator in EFFICACY_COMPARATOR_IDS},
        spec=_efficacy_spec(),
    )

    assert tuple(result.comparator_policy_id for result in results) == EFFICACY_COMPARATOR_IDS
    assert all(result.primary_metric == "guardrail_clean_task_score" for result in results)
    assert all(result.status is ScientificEvidenceStatus.PASS for result in results)
    assert all(result.holm is not None for result in results[:2])
    assert results[2].holm is None
    assert results[2].included_in_holm_family is False
    assert all(result.holm is None or result.holm.family_size == 2 for result in results)

    with pytest.raises(ValueError, match="exactly the efficacy comparators"):
        evaluate_efficacy_comparisons(
            {
                **{comparator: differences for comparator in EFFICACY_COMPARATOR_IDS},
                ATTRIBUTION_COMPARATOR_ID: differences,
            },
            spec=_efficacy_spec(),
        )


def test_invalid_efficacy_guardrail_prevents_all_inferential_statistics() -> None:
    differences = (0.1,) * 21
    results = evaluate_efficacy_comparisons(
        {comparator: differences for comparator in EFFICACY_COMPARATOR_IDS},
        spec=_efficacy_spec(),
        guardrails_by_comparator={
            "best_static": (_causal_guardrail(ScientificGuardrailStatus.INVALID),)
        },
    )

    assert all(result.status is ScientificEvidenceStatus.INVALID for result in results)
    assert all(result.bootstrap is None for result in results)
    assert all(result.permutation is None for result in results)
    assert all(result.holm is None for result in results)

    unequal_coverage = evaluate_efficacy_comparisons(
        {
            "best_static": differences[:-1],
            "heuristic_adaptive": differences,
            "random_matched": differences,
        },
        spec=_efficacy_spec(),
    )
    nonfinite = evaluate_efficacy_comparisons(
        {
            "best_static": (float("nan"),) * 21,
            "heuristic_adaptive": differences,
            "random_matched": differences,
        },
        spec=_efficacy_spec(),
    )
    for invalid_family in (unequal_coverage, nonfinite):
        assert all(result.status is ScientificEvidenceStatus.INVALID for result in invalid_family)
        assert all(result.bootstrap is None for result in invalid_family)


def test_required_serious_efficacy_has_distinct_negative_and_inconclusive_boundaries() -> None:
    positive = (0.1,) * 21
    decisively_below_threshold = (0.0,) * 21
    boundary_crossing = (-0.2, 0.24) * 10 + (0.02,)

    negative = evaluate_efficacy_comparisons(
        {
            "best_static": decisively_below_threshold,
            "heuristic_adaptive": positive,
            "random_matched": positive,
        },
        spec=_efficacy_spec(),
    )
    inconclusive = evaluate_efficacy_comparisons(
        {
            "best_static": boundary_crossing,
            "heuristic_adaptive": positive,
            "random_matched": positive,
        },
        spec=_efficacy_spec(),
    )

    assert negative[0].status is ScientificEvidenceStatus.DECISIVE_NEGATIVE
    assert inconclusive[0].status is ScientificEvidenceStatus.INCONCLUSIVE


def test_recovery_endpoints_are_positive_favors_neural_and_fail_closed() -> None:
    assert post_stressor_task_score_change(0.3, 0.7) == pytest.approx(0.4)
    assert post_stressor_repetition_change(0.6, 0.2) == pytest.approx(0.4)
    assert (
        time_to_return_to_target_band(
            (0.4, 0.7),
            (0.5, 0.2),
            minimum_task_score=0.6,
            maximum_repetition_ratio=0.3,
        )
        == 2
    )
    assert (
        time_to_return_to_target_band(
            (0.4, 0.5),
            (0.5, 0.4),
            minimum_task_score=0.6,
            maximum_repetition_ratio=0.3,
        )
        is None
    )

    values: dict[RecoveryMetricName | str, Sequence[float]] = {
        metric: (0.1,) * 12 for metric in RecoveryMetricName
    }
    passed = evaluate_recovery(values, spec=_recovery_spec())
    assert passed.status is ScientificEvidenceStatus.PASS
    assert tuple(result.metric_name for result in passed.metric_results) == tuple(
        RecoveryMetricName
    )

    negative_values = {**values, RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: (0.0,) * 12}
    negative = evaluate_recovery(negative_values, spec=_recovery_spec())
    assert negative.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE

    missing = evaluate_recovery(
        {RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: (0.1,) * 12},
        spec=_recovery_spec(),
    )
    assert missing.status is ScientificEvidenceStatus.INVALID
    assert missing.metric_results == ()


def test_recovery_spec_is_deeply_frozen_and_json_round_trippable() -> None:
    spec = _recovery_spec()
    with pytest.raises(TypeError):
        setitem(
            spec.practical_thresholds,
            RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE,
            0.5,
        )  # type: ignore[call-overload]
    assert RecoveryAnalysisSpec.model_validate(spec.model_dump(mode="python")) == spec
    assert RecoveryAnalysisSpec.model_validate_json(spec.model_dump_json()) == spec

    result = evaluate_recovery(
        {metric: (0.1,) * 12 for metric in RecoveryMetricName},
        spec=spec,
    )
    assert RecoveryEvaluationResult.model_validate_json(result.model_dump_json()) == result


def test_persistent_state_reset_is_typed_attribution_only_and_never_holm_adjusted() -> None:
    spec = AttributionAnalysisSpec(
        practical_effect_threshold=0.02,
        bootstrap_resamples=256,
        bootstrap_seed=404,
        permutation_resamples=512,
        permutation_seed=505,
    )
    result = evaluate_persistent_state_attribution(
        (0.1,) * 21,
        spec=spec,
        causal_guardrails=(_causal_guardrail(),),
    )

    assert result.status is ScientificEvidenceStatus.PASS
    assert result.comparator_policy_id == ATTRIBUTION_COMPARATOR_ID
    assert result.attribution_only is True
    assert result.included_in_efficacy is False
    assert result.included_in_holm_family is False
    assert result.turn_zero_excluded_from_effect is True
    assert "holm" not in type(result).model_fields
    assert PersistentStateAttributionResult.model_validate_json(result.model_dump_json()) == result

    aliased = evaluate_persistent_state_attribution(
        (0.1,) * 21,
        spec=spec,
        causal_guardrails=(_causal_guardrail(),),
        behavioral_alias=True,
    )
    assert aliased.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE


def test_missing_or_invalid_causal_attribution_evidence_is_invalid_without_statistics() -> None:
    spec = AttributionAnalysisSpec(
        bootstrap_resamples=64,
        bootstrap_seed=606,
        permutation_resamples=64,
        permutation_seed=707,
    )
    missing_guardrails = evaluate_persistent_state_attribution(
        (0.1,) * 8,
        spec=spec,
        causal_guardrails=(),
    )
    invalid_guardrail = evaluate_persistent_state_attribution(
        (0.1,) * 8,
        spec=spec,
        causal_guardrails=(_causal_guardrail(ScientificGuardrailStatus.INVALID),),
    )

    for result in (missing_guardrails, invalid_guardrail):
        assert result.status is ScientificEvidenceStatus.INVALID
        assert result.bootstrap is None
        assert result.permutation is None


def test_attribution_rejects_malformed_contracts_and_preserves_all_three_boundaries() -> None:
    spec = AttributionAnalysisSpec(
        practical_effect_threshold=0.02,
        bootstrap_resamples=256,
        bootstrap_seed=808,
        permutation_resamples=512,
        permutation_seed=909,
    )
    guardrail = _causal_guardrail()
    with pytest.raises(TypeError, match="AttributionAnalysisSpec"):
        evaluate_persistent_state_attribution(
            (0.1,) * 8,
            spec=object(),  # type: ignore[arg-type]
            causal_guardrails=(guardrail,),
        )
    assert (
        evaluate_persistent_state_attribution(
            (),
            spec=spec,
            causal_guardrails=(guardrail,),
        ).status
        is ScientificEvidenceStatus.INVALID
    )
    assert (
        evaluate_persistent_state_attribution(
            (float("nan"),) * 8,
            spec=spec,
            causal_guardrails=(guardrail,),
        ).status
        is ScientificEvidenceStatus.INVALID
    )
    negative = evaluate_persistent_state_attribution(
        (0.0,) * 21,
        spec=spec,
        causal_guardrails=(guardrail,),
    )
    inconclusive = evaluate_persistent_state_attribution(
        (-0.2, 0.24) * 10 + (0.02,),
        spec=spec,
        causal_guardrails=(guardrail,),
    )
    failed_guardrail = evaluate_persistent_state_attribution(
        (0.1,) * 21,
        spec=spec,
        causal_guardrails=(_causal_guardrail(ScientificGuardrailStatus.FAIL),),
    )
    assert negative.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE
    assert inconclusive.status is ScientificEvidenceStatus.INCONCLUSIVE
    assert failed_guardrail.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE

    valid = evaluate_persistent_state_attribution(
        (0.1,) * 21,
        spec=spec,
        causal_guardrails=(guardrail,),
    )
    payload = valid.model_dump(mode="python")
    malformed = (
        ({**payload, "causal_guardrails": (guardrail, guardrail)}, "unique"),
        (
            {**payload, "status": ScientificEvidenceStatus.INVALID},
            "invalid attribution evidence",
        ),
        ({**payload, "bootstrap": None}, "complete nonempty"),
        ({**payload, "unit_count": valid.unit_count + 1}, "same matched units"),
        (
            {**payload, "status": ScientificEvidenceStatus.INCONCLUSIVE},
            "status does not match",
        ),
        ({**payload, "causal_guardrails": ()}, "requires causal guardrails"),
    )
    for invalid_payload, message in malformed:
        with pytest.raises(ValidationError, match=message):
            PersistentStateAttributionResult.model_validate(invalid_payload)


def test_recovery_guard_paths_reject_invalid_windows_counts_and_result_rebinding() -> None:
    with pytest.raises(ValidationError, match="exactly the three"):
        RecoveryAnalysisSpec(
            practical_thresholds={
                RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: 0.02,
            },
            bootstrap_seed=1,
        )
    with pytest.raises(ValidationError, match="nonnegative"):
        RecoveryAnalysisSpec(
            practical_thresholds={metric: -0.01 for metric in RecoveryMetricName},
            bootstrap_seed=1,
        )

    values: dict[RecoveryMetricName | str, Sequence[float]] = {
        metric: (0.1,) * 12 for metric in RecoveryMetricName
    }
    with pytest.raises(TypeError, match="RecoveryAnalysisSpec"):
        evaluate_recovery(values, spec=object())  # type: ignore[arg-type]
    for focal_count, comparator_count in ((True, 0), (-1, 0), (0, -1)):
        with pytest.raises(ValueError, match="nonnegative integers"):
            evaluate_recovery(
                values,
                spec=_recovery_spec(),
                right_censored_focal_units=focal_count,
                right_censored_comparator_units=comparator_count,
            )
    assert (
        evaluate_recovery(
            {"unknown_endpoint": (0.1,) * 12},
            spec=_recovery_spec(),
        ).status
        is ScientificEvidenceStatus.INVALID
    )
    unequal: dict[RecoveryMetricName | str, Sequence[float]] = {
        **values,
        RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: (0.1,) * 11,
    }
    empty: dict[RecoveryMetricName | str, Sequence[float]] = {
        metric: () for metric in RecoveryMetricName
    }
    nonfinite: dict[RecoveryMetricName | str, Sequence[float]] = {
        **values,
        RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: (float("nan"),),
    }
    for invalid_values in (unequal, empty, nonfinite):
        assert (
            evaluate_recovery(invalid_values, spec=_recovery_spec()).status
            is ScientificEvidenceStatus.INVALID
        )

    censored = evaluate_recovery(
        values,
        spec=_recovery_spec(),
        right_censored_focal_units=1,
        right_censored_comparator_units=2,
    )
    assert censored.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE
    assert censored.right_censored_focal_units == 1
    assert censored.right_censored_comparator_units == 2
    inconclusive = evaluate_recovery(
        {metric: (-0.2, 0.24) * 6 for metric in RecoveryMetricName},
        spec=_recovery_spec(),
    )
    assert inconclusive.status is ScientificEvidenceStatus.INCONCLUSIVE

    passed = evaluate_recovery(values, spec=_recovery_spec())
    metric = passed.metric_results[0]
    with pytest.raises(ValidationError, match="same matched units"):
        type(metric).model_validate(
            {**metric.model_dump(mode="python"), "unit_count": metric.unit_count + 1}
        )
    with pytest.raises(ValidationError, match="status does not match"):
        type(metric).model_validate(
            {
                **metric.model_dump(mode="python"),
                "status": ScientificEvidenceStatus.DECISIVE_NEGATIVE,
            }
        )
    with pytest.raises(ValidationError, match="partial statistics"):
        RecoveryEvaluationResult(
            metric_results=passed.metric_results,
            status=ScientificEvidenceStatus.INVALID,
            detail="invalid result cannot retain partial statistics",
        )
    with pytest.raises(ValidationError, match="exact canonical metric set"):
        RecoveryEvaluationResult(
            metric_results=passed.metric_results[:-1],
            status=ScientificEvidenceStatus.PASS,
            detail="incomplete family",
        )
    with pytest.raises(ValidationError, match="family status"):
        RecoveryEvaluationResult.model_validate(
            {
                **passed.model_dump(mode="python"),
                "status": ScientificEvidenceStatus.DECISIVE_NEGATIVE,
            }
        )


@pytest.mark.parametrize(
    ("task_scores", "repetition", "minimum", "maximum", "message"),
    (
        ((0.5,), (0.5, 0.4), 0.5, 0.5, "equal lengths"),
        ((0.5,), (0.5,), -0.1, 0.5, "minimum_task_score"),
        ((0.5,), (0.5,), 0.5, 1.1, "maximum_repetition_ratio"),
        ((1.1,), (0.5,), 0.5, 0.5, "trajectory values"),
    ),
)
def test_time_to_band_validates_the_complete_preregistered_trajectory(
    task_scores: tuple[float, ...],
    repetition: tuple[float, ...],
    minimum: float,
    maximum: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        time_to_return_to_target_band(
            task_scores,
            repetition,
            minimum_task_score=minimum,
            maximum_repetition_ratio=maximum,
        )

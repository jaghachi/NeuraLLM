"""Frozen Phase 5 decision vocabulary, score semantics, and truth table."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from neurallm.evaluation.scientific import (
    EFFICACY_COMPARATOR_IDS,
    REQUIRED_SCIENTIFIC_GUARDRAILS,
    ComparatorRole,
    EfficacyAnalysisSpec,
    EfficacyComparisonResult,
    ExperimentTier,
    FinalDecisionIneligibleError,
    GuardrailCleanTaskScore,
    LimitationDisposition,
    LimitationKind,
    ScientificDecisionInput,
    ScientificDecisionRecord,
    ScientificDecisionState,
    ScientificEvidenceGate,
    ScientificEvidenceKind,
    ScientificEvidenceStatus,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
    ScientificLimitation,
    ScientificReasonCode,
    decide_scientific_outcome,
    evaluate_efficacy_comparisons,
    guardrail_clean_task_difference,
)


def _passing_efficacy() -> tuple[EfficacyComparisonResult, ...]:
    spec = EfficacyAnalysisSpec(
        practical_effect_threshold=0.02,
        bootstrap_resamples=256,
        bootstrap_seed=101,
        permutation_resamples=512,
        permutation_seed=202,
    )
    differences = (0.1,) * 21
    return evaluate_efficacy_comparisons(
        {
            "best_static": differences,
            "heuristic_adaptive": differences,
            "random_matched": differences,
        },
        spec=spec,
    )


def _global_guardrails(
    status: ScientificGuardrailStatus = ScientificGuardrailStatus.PASS,
) -> tuple[ScientificGuardrailResult, ...]:
    return tuple(
        ScientificGuardrailResult(
            name=name,
            status=status,
            scope="confirmatory_run",
            detail=f"{name} is {status.value}",
        )
        for name in REQUIRED_SCIENTIFIC_GUARDRAILS
    )


def _decision_input(
    *,
    tier: ExperimentTier = ExperimentTier.CONFIRMATORY,
    guardrail_status: ScientificGuardrailStatus = ScientificGuardrailStatus.PASS,
    recovery_status: ScientificEvidenceStatus = ScientificEvidenceStatus.PASS,
    attribution_status: ScientificEvidenceStatus = ScientificEvidenceStatus.PASS,
    limitations: tuple[ScientificLimitation, ...] = (),
) -> ScientificDecisionInput:
    return ScientificDecisionInput(
        tier=tier,
        efficacy_comparisons=_passing_efficacy(),
        recovery=ScientificEvidenceGate(
            kind=ScientificEvidenceKind.RECOVERY,
            status=recovery_status,
            detail="required output-recovery family",
        ),
        attribution=ScientificEvidenceGate(
            kind=ScientificEvidenceKind.PERSISTENT_STATE_ATTRIBUTION,
            status=attribution_status,
            detail="matched-history persistent-state output attribution",
        ),
        guardrails=_global_guardrails(guardrail_status),
        limitations=limitations,
    )


def test_final_decision_vocabulary_is_exact_and_case_sensitive() -> None:
    assert tuple(state.value for state in ScientificDecisionState) == (
        "VALIDATED_POSITIVE",
        "VALIDATED_NEGATIVE",
        "INCONCLUSIVE",
        "INVALID_RUN",
    )


def test_guardrail_clean_task_score_gates_the_raw_score_without_imputation_or_blending() -> None:
    clean_focal = GuardrailCleanTaskScore(
        raw_task_score=0.73,
        gate_status=ScientificGuardrailStatus.PASS,
        gate_names=("instruction_adherence_non_regression",),
    )
    clean_comparator = GuardrailCleanTaskScore(
        raw_task_score=0.23,
        gate_status=ScientificGuardrailStatus.PASS,
        gate_names=("instruction_adherence_non_regression",),
    )
    failed = GuardrailCleanTaskScore(
        raw_task_score=0.73,
        gate_status=ScientificGuardrailStatus.FAIL,
        gate_names=("instruction_adherence_non_regression",),
    )

    assert clean_focal.gated_value == 0.73
    assert guardrail_clean_task_difference(clean_focal, clean_comparator) == pytest.approx(0.5)
    assert failed.raw_task_score == 0.73
    assert failed.gated_value is None
    assert guardrail_clean_task_difference(failed, clean_comparator) is None
    with pytest.raises(ValidationError):
        clean_focal.raw_task_score = 0.0
    with pytest.raises(ValidationError):
        GuardrailCleanTaskScore.model_validate(
            {
                "raw_task_score": 0.5,
                "gate_status": "pass",
                "gate_names": ["gate"],
                "blended_score": 0.5,
            }
        )


@pytest.mark.parametrize(
    (
        "guardrail_status",
        "recovery_status",
        "attribution_status",
        "expected",
    ),
    [
        (
            ScientificGuardrailStatus.INVALID,
            ScientificEvidenceStatus.DECISIVE_NEGATIVE,
            ScientificEvidenceStatus.INCONCLUSIVE,
            ScientificDecisionState.INVALID_RUN,
        ),
        (
            ScientificGuardrailStatus.FAIL,
            ScientificEvidenceStatus.INCONCLUSIVE,
            ScientificEvidenceStatus.PASS,
            ScientificDecisionState.VALIDATED_NEGATIVE,
        ),
        (
            ScientificGuardrailStatus.PASS,
            ScientificEvidenceStatus.INCONCLUSIVE,
            ScientificEvidenceStatus.PASS,
            ScientificDecisionState.INCONCLUSIVE,
        ),
        (
            ScientificGuardrailStatus.PASS,
            ScientificEvidenceStatus.PASS,
            ScientificEvidenceStatus.PASS,
            ScientificDecisionState.VALIDATED_POSITIVE,
        ),
    ],
)
def test_final_truth_table_has_explicit_precedence(
    guardrail_status: ScientificGuardrailStatus,
    recovery_status: ScientificEvidenceStatus,
    attribution_status: ScientificEvidenceStatus,
    expected: ScientificDecisionState,
) -> None:
    evidence = _decision_input(
        guardrail_status=guardrail_status,
        recovery_status=recovery_status,
        attribution_status=attribution_status,
    )

    assert decide_scientific_outcome(evidence).decision is expected


@pytest.mark.parametrize(
    "tier",
    [ExperimentTier.ENGINEERING_SMOKE, ExperimentTier.DEVELOPMENT_PILOT],
)
def test_smoke_and_pilot_are_ineligible_for_a_final_decision(tier: ExperimentTier) -> None:
    with pytest.raises(FinalDecisionIneligibleError, match="cannot emit a final decision"):
        decide_scientific_outcome(_decision_input(tier=tier))


def test_missing_required_guardrail_is_invalid_and_limitation_can_force_inconclusive() -> None:
    complete = _decision_input()
    missing = ScientificDecisionInput(
        **{
            **complete.model_dump(),
            "guardrails": complete.guardrails[:-1],
        }
    )
    invalid = decide_scientific_outcome(missing)
    assert invalid.decision is ScientificDecisionState.INVALID_RUN
    assert ScientificReasonCode.MISSING_REQUIRED_GUARDRAIL in invalid.reason_codes

    limitation = ScientificLimitation(
        kind=LimitationKind.SUBGROUP_CONFLICT,
        code="preregistered-subgroup-conflict",
        detail="required subgroups disagree across the frozen decision boundary",
        disposition=LimitationDisposition.INCONCLUSIVE,
    )
    inconclusive = decide_scientific_outcome(_decision_input(limitations=(limitation,)))
    assert inconclusive.decision is ScientificDecisionState.INCONCLUSIVE
    assert ScientificReasonCode.SUBGROUP_CONFLICT in inconclusive.reason_codes


def test_required_serious_comparison_drives_negative_and_inconclusive_states() -> None:
    spec = EfficacyAnalysisSpec(
        bootstrap_resamples=256,
        bootstrap_seed=808,
        permutation_resamples=512,
        permutation_seed=909,
    )
    positive = (0.1,) * 21
    comparison_families = (
        (
            (0.0,) * 21,
            ScientificDecisionState.VALIDATED_NEGATIVE,
            ScientificReasonCode.REQUIRED_COMPARATOR_FAILED,
        ),
        (
            (-0.2, 0.24) * 10 + (0.02,),
            ScientificDecisionState.INCONCLUSIVE,
            ScientificReasonCode.EFFICACY_UNRESOLVED,
        ),
    )
    base = _decision_input()
    for best_static, expected_decision, expected_reason in comparison_families:
        comparisons = evaluate_efficacy_comparisons(
            {
                "best_static": best_static,
                "heuristic_adaptive": positive,
                "random_matched": positive,
            },
            spec=spec,
        )
        evidence = ScientificDecisionInput.model_validate(
            {
                **base.model_dump(),
                "efficacy_comparisons": comparisons,
            }
        )
        decision = decide_scientific_outcome(evidence)
        assert decision.decision is expected_decision
        assert expected_reason in decision.reason_codes


def test_scientific_decision_input_and_record_round_trip_canonical_json() -> None:
    evidence = _decision_input()
    restored_evidence = ScientificDecisionInput.model_validate_json(evidence.model_dump_json())
    assert restored_evidence == evidence

    decision = decide_scientific_outcome(evidence)
    restored_decision = ScientificDecisionRecord.model_validate_json(decision.model_dump_json())
    assert restored_decision == decision
    assert restored_decision.decision_input_sha256 == decision.decision_input_sha256

    with pytest.raises(ValidationError, match="reason codes do not match"):
        ScientificDecisionRecord.model_validate(
            {
                **decision.model_dump(),
                "decision": ScientificDecisionState.VALIDATED_NEGATIVE,
            }
        )


def test_scientific_models_reject_role_gate_and_statistical_rebinding() -> None:
    clean = GuardrailCleanTaskScore(
        raw_task_score=0.5,
        gate_status=ScientificGuardrailStatus.PASS,
        gate_names=("gate",),
    )
    for gate_names in ((), ("gate", "gate")):
        with pytest.raises(ValidationError, match="sorted, and unique"):
            GuardrailCleanTaskScore(
                raw_task_score=0.5,
                gate_status=ScientificGuardrailStatus.PASS,
                gate_names=gate_names,
            )
    canonicalized = GuardrailCleanTaskScore(
        raw_task_score=0.5,
        gate_status=ScientificGuardrailStatus.PASS,
        gate_names=("z", "a"),
    )
    assert canonicalized.gate_names == ("a", "z")
    with pytest.raises(TypeError, match="typed task scores"):
        guardrail_clean_task_difference(object(), clean)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="Input should be 'best_static'"):
        EfficacyAnalysisSpec(
            serious_comparator_ids=("heuristic_adaptive", "best_static"),  # type: ignore[arg-type]
            bootstrap_seed=1,
            permutation_seed=2,
        )

    differences: dict[str, Sequence[float]] = {
        comparator: (0.1,) * 21 for comparator in EFFICACY_COMPARATOR_IDS
    }
    spec = EfficacyAnalysisSpec(
        bootstrap_resamples=256,
        bootstrap_seed=1,
        permutation_resamples=512,
        permutation_seed=2,
    )
    with pytest.raises(TypeError, match="EfficacyAnalysisSpec"):
        evaluate_efficacy_comparisons(
            differences,
            spec=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-efficacy comparator"):
        evaluate_efficacy_comparisons(
            differences,
            spec=spec,
            guardrails_by_comparator={"foreign": ()},
        )
    with pytest.raises(ValueError, match="non-efficacy comparator"):
        evaluate_efficacy_comparisons(
            differences,
            spec=spec,
            behavioral_alias_by_comparator={"foreign": False},
        )
    with pytest.raises(TypeError, match="must be booleans"):
        evaluate_efficacy_comparisons(
            differences,
            spec=spec,
            behavioral_alias_by_comparator={"best_static": 1},  # type: ignore[dict-item]
        )

    comparisons = evaluate_efficacy_comparisons(differences, spec=spec)
    serious = comparisons[0]
    negative_control = comparisons[2]
    assert serious.holm is not None
    passing_guardrail = ScientificGuardrailResult(
        name="gate",
        status=ScientificGuardrailStatus.PASS,
        scope="pair",
        detail="gate passed",
    )
    serious_payload = serious.model_dump(mode="python")
    malformed = (
        (
            {**serious_payload, "comparator_role": ComparatorRole.NEGATIVE_CONTROL},
            "role does not match",
        ),
        ({**serious_payload, "included_in_holm_family": False}, "Holm-family membership"),
        (
            {**serious_payload, "guardrails": (passing_guardrail, passing_guardrail)},
            "unique by name and scope",
        ),
        (
            {**serious_payload, "status": ScientificEvidenceStatus.INVALID},
            "must not contain inferential statistics",
        ),
        ({**serious_payload, "bootstrap": None}, "complete nonempty statistics"),
        ({**serious_payload, "unit_count": serious.unit_count + 1}, "same matched units"),
        ({**serious_payload, "holm": None}, "requires its Holm result"),
        (
            {
                **serious_payload,
                "holm": serious.holm.model_copy(update={"family_size": 3}),
            },
            "Holm evidence must match",
        ),
        (
            {
                **negative_control.model_dump(mode="python"),
                "holm": serious.holm,
            },
            "excluded from Holm",
        ),
        (
            {**serious_payload, "status": ScientificEvidenceStatus.INCONCLUSIVE},
            "status does not match",
        ),
    )
    for payload, message in malformed:
        with pytest.raises(ValidationError, match=message):
            EfficacyComparisonResult.model_validate(payload)


def test_decision_contract_rejects_rebinding_and_emits_each_family_reason() -> None:
    base = _decision_input()
    with pytest.raises(ValidationError, match="required guardrail names"):
        ScientificDecisionInput.model_validate(
            {**base.model_dump(mode="python"), "required_guardrail_names": ("other",)}
        )
    with pytest.raises(ValidationError, match="canonical comparator set"):
        ScientificDecisionInput.model_validate(
            {
                **base.model_dump(mode="python"),
                "efficacy_comparisons": base.efficacy_comparisons[:-1],
            }
        )
    short_comparisons = evaluate_efficacy_comparisons(
        {comparator: (0.1,) * 20 for comparator in EFFICACY_COMPARATOR_IDS},
        spec=EfficacyAnalysisSpec(
            bootstrap_resamples=256,
            bootstrap_seed=21,
            permutation_resamples=512,
            permutation_seed=22,
        ),
    )
    mismatched = (short_comparisons[0], *base.efficacy_comparisons[1:])
    with pytest.raises(ValidationError, match="exact matched-unit coverage"):
        ScientificDecisionInput.model_validate(
            {**base.model_dump(mode="python"), "efficacy_comparisons": mismatched}
        )
    wrong_kind = ScientificEvidenceGate(
        kind=ScientificEvidenceKind.PERSISTENT_STATE_ATTRIBUTION,
        status=ScientificEvidenceStatus.PASS,
        detail="wrong evidence kind",
    )
    with pytest.raises(ValidationError, match="recovery gate has the wrong"):
        ScientificDecisionInput.model_validate(
            {**base.model_dump(mode="python"), "recovery": wrong_kind}
        )
    with pytest.raises(ValidationError, match="attribution gate has the wrong"):
        ScientificDecisionInput.model_validate(
            {
                **base.model_dump(mode="python"),
                "attribution": wrong_kind.model_copy(
                    update={"kind": ScientificEvidenceKind.RECOVERY}
                ),
            }
        )
    with pytest.raises(ValidationError, match="guardrails must be unique"):
        ScientificDecisionInput.model_validate(
            {**base.model_dump(mode="python"), "guardrails": (*base.guardrails, base.guardrails[0])}
        )
    limitation = ScientificLimitation(
        kind=LimitationKind.OPTIONAL_METRIC_UNAVAILABLE,
        code="semantic_similarity_missing",
        detail="semantic similarity was unavailable",
        disposition=LimitationDisposition.INCONCLUSIVE,
    )
    with pytest.raises(ValidationError, match="limitations must be unique"):
        ScientificDecisionInput.model_validate(
            {**base.model_dump(mode="python"), "limitations": (limitation, limitation)}
        )

    decision = decide_scientific_outcome(base)
    with pytest.raises(ValidationError, match="at least one reason"):
        ScientificDecisionRecord.model_validate(
            {**decision.model_dump(mode="python"), "reason_codes": ()}
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        ScientificDecisionRecord.model_validate(
            {
                **decision.model_dump(mode="python"),
                "reason_codes": (
                    ScientificReasonCode.ALL_POSITIVE_GATES_PASSED,
                    ScientificReasonCode.ALL_POSITIVE_GATES_PASSED,
                ),
            }
        )
    with pytest.raises(ValidationError, match="reason codes do not match"):
        ScientificDecisionRecord.model_validate(
            {
                **decision.model_dump(mode="python"),
                "reason_codes": (ScientificReasonCode.RECOVERY_FAILED,),
            }
        )
    with pytest.raises(TypeError, match="ScientificDecisionInput"):
        decide_scientific_outcome(object())  # type: ignore[arg-type]

    positive = (0.1,) * 21
    negative_control_failed = evaluate_efficacy_comparisons(
        {
            "best_static": positive,
            "heuristic_adaptive": positive,
            "random_matched": (0.0,) * 21,
        },
        spec=EfficacyAnalysisSpec(
            bootstrap_resamples=256,
            bootstrap_seed=11,
            permutation_resamples=512,
            permutation_seed=12,
        ),
    )
    negative_control_input = ScientificDecisionInput.model_validate(
        {**base.model_dump(mode="python"), "efficacy_comparisons": negative_control_failed}
    )
    negative_control_decision = decide_scientific_outcome(negative_control_input)
    assert negative_control_decision.decision is ScientificDecisionState.VALIDATED_NEGATIVE
    assert (
        ScientificReasonCode.NEGATIVE_CONTROL_SANITY_FAILED
        in negative_control_decision.reason_codes
    )

    family_cases = (
        (
            _decision_input(attribution_status=ScientificEvidenceStatus.DECISIVE_NEGATIVE),
            ScientificReasonCode.ATTRIBUTION_FAILED,
            ScientificDecisionState.VALIDATED_NEGATIVE,
        ),
        (
            _decision_input(attribution_status=ScientificEvidenceStatus.INCONCLUSIVE),
            ScientificReasonCode.ATTRIBUTION_UNRESOLVED,
            ScientificDecisionState.INCONCLUSIVE,
        ),
        (
            _decision_input(attribution_status=ScientificEvidenceStatus.INVALID),
            ScientificReasonCode.ATTRIBUTION_INVALID,
            ScientificDecisionState.INVALID_RUN,
        ),
        (
            _decision_input(recovery_status=ScientificEvidenceStatus.DECISIVE_NEGATIVE),
            ScientificReasonCode.RECOVERY_FAILED,
            ScientificDecisionState.VALIDATED_NEGATIVE,
        ),
    )
    for evidence, reason, state in family_cases:
        outcome = decide_scientific_outcome(evidence)
        assert outcome.decision is state
        assert reason in outcome.reason_codes

    optional = decide_scientific_outcome(_decision_input(limitations=(limitation,)))
    assert optional.decision is ScientificDecisionState.INCONCLUSIVE
    assert ScientificReasonCode.OPTIONAL_METRIC_UNAVAILABLE in optional.reason_codes

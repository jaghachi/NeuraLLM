"""Synthetic exit-gate matrix for the pure Phase 3 evaluator."""

from __future__ import annotations

from random import Random

import pytest

import neurallm.evaluation.engine as engine_module
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    GuardrailName,
    GuardrailStatus,
    Phase3EvaluationResult,
    Phase3Verdict,
    SequenceExpectation,
    TurnEvaluationRecord,
    evaluate_phase3,
)

DATASET_HASH = "a" * 64
PROVIDER_HASH = "b" * 64
HISTORY_HASH = "c" * 64
POLICIES = ("focal", "heuristic", "random", "static")


class PolicyValues:
    """Compact synthetic policy outcomes repeated over the condition grid."""

    def __init__(
        self,
        *,
        task_score: float,
        adherence: float = 0.9,
        length: int = 100,
        repetition: float = 0.2,
        action_magnitude: float = 0.1,
        saturated: bool = False,
    ) -> None:
        self.task_score = task_score
        self.adherence = adherence
        self.length = length
        self.repetition = repetition
        self.action_magnitude = action_magnitude
        self.saturated = saturated


def make_design() -> ExpectedEvaluationDesign:
    return ExpectedEvaluationDesign(
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        dataset_sha256=DATASET_HASH,
        provider_identity_id=PROVIDER_HASH,
        sequences=tuple(
            SequenceExpectation(prompt_sequence_id=f"sequence-{index}", turn_count=2)
            for index in range(3)
        ),
        model_seeds=(11, 12, 13, 14),
        controller_seeds=(21, 22),
        policy_ids=POLICIES,
    )


def make_spec() -> EvaluationSpec:
    return EvaluationSpec(
        focal_policy_id="focal",
        required_serious_comparator_ids=("static", "heuristic"),
        negative_control_policy_ids=("random",),
        bootstrap_resamples=200,
        bootstrap_seed=101,
        permutation_resamples=200,
        permutation_seed=202,
    )


def make_records(
    values: dict[str, PolicyValues],
    *,
    design: ExpectedEvaluationDesign | None = None,
) -> tuple[TurnEvaluationRecord, ...]:
    frozen_design = design or make_design()
    records: list[TurnEvaluationRecord] = []
    for sequence in frozen_design.sequences:
        for model_seed in frozen_design.model_seeds:
            for controller_seed in frozen_design.controller_seeds:
                for policy_id in frozen_design.policy_ids:
                    policy = values[policy_id]
                    for turn_index in range(sequence.turn_count):
                        records.append(
                            TurnEvaluationRecord(
                                dataset_sha256=frozen_design.dataset_sha256,
                                prompt_sequence_id=sequence.prompt_sequence_id,
                                turn_index=turn_index,
                                policy_id=policy_id,
                                model_seed=model_seed,
                                controller_seed=controller_seed,
                                provider_identity_id=frozen_design.provider_identity_id,
                                has_previous_response=turn_index > 0,
                                previous_history_commitment_sha256=(
                                    HISTORY_HASH if turn_index > 0 else None
                                ),
                                task_score=policy.task_score,
                                instruction_adherence=policy.adherence,
                                response_length_tokens=policy.length,
                                repetition_ratio=policy.repetition,
                                action_magnitude=policy.action_magnitude,
                                action_within_bounds=True,
                                action_saturated=policy.saturated,
                            )
                        )
    return tuple(records)


def superior_values() -> dict[str, PolicyValues]:
    return {
        "focal": PolicyValues(task_score=0.85, action_magnitude=0.2),
        "static": PolicyValues(task_score=0.55),
        "heuristic": PolicyValues(task_score=0.60),
        "random": PolicyValues(task_score=0.45),
    }


def test_known_superior_focal_policy_is_selected_against_both_serious_comparators() -> None:
    result = evaluate_phase3(
        make_records(superior_values()), design=make_design(), spec=make_spec()
    )

    assert result.verdict is Phase3Verdict.SUPERIOR
    assert result.statistics_call_count == 6
    serious = tuple(
        comparison for comparison in result.comparisons if comparison.serious_comparator
    )
    assert len(serious) == 2
    assert all(comparison.verdict is Phase3Verdict.SUPERIOR for comparison in serious)
    assert all(comparison.holm is not None for comparison in serious)
    negative = next(
        comparison for comparison in result.comparisons if not comparison.serious_comparator
    )
    assert negative.holm is None


def test_known_inferior_focal_policy_is_rejected() -> None:
    values = superior_values()
    values["focal"] = PolicyValues(task_score=0.25, action_magnitude=0.2)

    result = evaluate_phase3(make_records(values), design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INFERIOR
    assert all(
        comparison.verdict is Phase3Verdict.INFERIOR
        for comparison in result.comparisons
        if comparison.serious_comparator
    )


def test_unresolved_mixed_effects_are_inconclusive() -> None:
    records = list(
        make_records(
            {
                "focal": PolicyValues(task_score=0.6, action_magnitude=0.2),
                "static": PolicyValues(task_score=0.6),
                "heuristic": PolicyValues(task_score=0.6),
                "random": PolicyValues(task_score=0.4),
            }
        )
    )
    for index, record in enumerate(records):
        if record.policy_id == "focal":
            task_score = 0.65 if record.model_seed in {11, 12} else 0.55
            records[index] = record.model_copy(update={"task_score": task_score})

    result = evaluate_phase3(records, design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INCONCLUSIVE
    serious = tuple(
        comparison for comparison in result.comparisons if comparison.serious_comparator
    )
    assert all(comparison.verdict is Phase3Verdict.INCONCLUSIVE for comparison in serious)
    assert all(
        comparison.bootstrap.lower < 0.0 < comparison.bootstrap.upper for comparison in serious
    )


def test_holm_correction_can_move_raw_significance_beyond_alpha() -> None:
    records = list(
        make_records(
            {
                "focal": PolicyValues(task_score=0.6, action_magnitude=0.2),
                "static": PolicyValues(task_score=0.5),
                "heuristic": PolicyValues(task_score=0.5),
                "random": PolicyValues(task_score=0.4),
            }
        )
    )
    negative_units = {("sequence-0", 11), ("sequence-0", 12)}
    for index, record in enumerate(records):
        if (
            record.policy_id == "focal"
            and (
                record.prompt_sequence_id,
                record.model_seed,
            )
            in negative_units
        ):
            records[index] = record.model_copy(update={"task_score": 0.4})
    spec = make_spec().model_copy(update={"permutation_resamples": 4_096})

    result = evaluate_phase3(records, design=make_design(), spec=spec)

    alpha = 1.0 - spec.confidence_level
    serious = tuple(
        comparison for comparison in result.comparisons if comparison.serious_comparator
    )
    assert result.verdict is Phase3Verdict.INCONCLUSIVE
    assert all(comparison.verdict is Phase3Verdict.INCONCLUSIVE for comparison in serious)
    assert all(comparison.bootstrap.lower > 0.0 for comparison in serious)
    assert all(
        comparison.mean_difference >= spec.practical_effect_threshold for comparison in serious
    )
    assert all(comparison.permutation.p_value <= alpha for comparison in serious)
    assert all(
        comparison.holm is not None and comparison.holm.adjusted_p_value > alpha
        for comparison in serious
    )


def test_identical_serious_policies_are_equivalent_and_behaviorally_aliased() -> None:
    identical = PolicyValues(task_score=0.6, action_magnitude=0.1)
    values = {
        "focal": identical,
        "static": identical,
        "heuristic": identical,
        "random": PolicyValues(task_score=0.4),
    }

    result = evaluate_phase3(make_records(values), design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.EQUIVALENT
    serious = tuple(
        comparison for comparison in result.comparisons if comparison.serious_comparator
    )
    assert all(comparison.verdict is Phase3Verdict.EQUIVALENT for comparison in serious)
    assert all(comparison.behavioral_alias for comparison in serious)
    for comparison in serious:
        alias_guardrail = next(
            guardrail
            for guardrail in comparison.guardrails
            if guardrail.name is GuardrailName.BEHAVIORAL_ALIAS_DETECTION
        )
        assert alias_guardrail.observed_value == 0.0
        assert alias_guardrail.threshold == 0.0


def test_incomplete_expected_coverage_is_invalid_before_any_statistics_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_statistics(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("statistics must not run after failed coverage")

    monkeypatch.setattr(engine_module, "paired_bootstrap_ci", unexpected_statistics)
    monkeypatch.setattr(
        engine_module,
        "paired_sign_flip_permutation_test",
        unexpected_statistics,
    )
    incomplete = make_records(superior_values())[:-1]

    result = evaluate_phase3(incomplete, design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INVALID
    assert result.coverage.exact is False
    assert len(result.coverage.missing_keys) == 1
    assert result.statistics_computed is False
    assert result.statistics_call_count == 0
    assert result.outcomes == ()
    assert result.comparisons == ()


@pytest.mark.parametrize(
    ("record_updates", "expected_guardrail"),
    [
        ({"dataset_sha256": "d" * 64}, GuardrailName.MATCHED_CONDITION_COVERAGE),
        ({"provider_identity_id": "d" * 64}, GuardrailName.PROVIDER_IDENTITY_STABILITY),
        (
            {
                "has_previous_response": True,
                "previous_history_commitment_sha256": HISTORY_HASH,
            },
            GuardrailName.TURN_ZERO_EQUIVALENCE,
        ),
        ({"action_within_bounds": False}, GuardrailName.ACTION_BOUND_COMPLIANCE),
        ({"task_score": None}, GuardrailName.METRIC_AVAILABILITY),
    ],
)
def test_integrity_guardrails_invalidate_before_statistics(
    record_updates: dict[str, object],
    expected_guardrail: GuardrailName,
) -> None:
    records = list(make_records(superior_values()))
    records[0] = records[0].model_copy(update=record_updates)

    result = evaluate_phase3(records, design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INVALID
    assert result.statistics_call_count == 0
    assert any(
        guardrail.name is expected_guardrail and guardrail.status is GuardrailStatus.INVALID
        for guardrail in result.global_guardrails
    )


def test_length_only_repetition_improvement_is_rejected_by_explicit_guardrail() -> None:
    values = {
        "focal": PolicyValues(task_score=0.6, length=50, repetition=0.05, action_magnitude=0.2),
        "static": PolicyValues(task_score=0.6, length=100, repetition=0.25),
        "heuristic": PolicyValues(task_score=0.6, length=100, repetition=0.25),
        "random": PolicyValues(task_score=0.4),
    }

    result = evaluate_phase3(make_records(values), design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INFERIOR
    for comparison in result.comparisons:
        if comparison.serious_comparator:
            length_guardrail = next(
                guardrail
                for guardrail in comparison.guardrails
                if guardrail.name is GuardrailName.RESPONSE_LENGTH_CONFOUND
            )
            assert length_guardrail.status is GuardrailStatus.FAIL


def test_length_confound_cannot_be_hidden_by_lengthening_other_matched_units() -> None:
    records = list(make_records(superior_values()))
    focal_by_sequence = {
        "sequence-0": {"response_length_tokens": 50, "repetition_ratio": 0.0},
        "sequence-1": {"response_length_tokens": 150, "repetition_ratio": 0.4},
        "sequence-2": {"response_length_tokens": 100, "repetition_ratio": 0.2},
    }
    comparator_by_sequence = {
        "sequence-0": {"response_length_tokens": 100, "repetition_ratio": 0.4},
        "sequence-1": {"response_length_tokens": 100, "repetition_ratio": 0.3},
        "sequence-2": {"response_length_tokens": 100, "repetition_ratio": 0.2},
    }
    for index, record in enumerate(records):
        if record.policy_id == "focal":
            records[index] = record.model_copy(update=focal_by_sequence[record.prompt_sequence_id])
        elif record.policy_id in {"static", "heuristic"}:
            records[index] = record.model_copy(
                update=comparator_by_sequence[record.prompt_sequence_id]
            )

    result = evaluate_phase3(records, design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INFERIOR
    for comparison in result.comparisons:
        if comparison.serious_comparator:
            length_guardrail = next(
                guardrail
                for guardrail in comparison.guardrails
                if guardrail.name is GuardrailName.RESPONSE_LENGTH_CONFOUND
            )
            assert length_guardrail.status is GuardrailStatus.FAIL
            assert length_guardrail.observed_value == pytest.approx(0.5)
            assert length_guardrail.threshold == pytest.approx(0.05)


def test_adherence_regression_and_action_saturation_are_substantive_failures() -> None:
    values = superior_values()
    values["focal"] = PolicyValues(
        task_score=0.85,
        adherence=0.5,
        action_magnitude=0.2,
        saturated=True,
    )

    result = evaluate_phase3(make_records(values), design=make_design(), spec=make_spec())

    assert result.verdict is Phase3Verdict.INFERIOR
    saturation = next(
        guardrail
        for guardrail in result.global_guardrails
        if guardrail.name is GuardrailName.ACTION_SATURATION_RATE
    )
    assert saturation.status is GuardrailStatus.FAIL
    assert all(
        any(
            guardrail.name is GuardrailName.INSTRUCTION_ADHERENCE_NON_REGRESSION
            and guardrail.status is GuardrailStatus.FAIL
            for guardrail in comparison.guardrails
        )
        for comparison in result.comparisons
        if comparison.serious_comparator
    )


def test_turns_and_controller_seeds_are_nested_into_exactly_twelve_matched_units() -> None:
    result = evaluate_phase3(
        make_records(superior_values()), design=make_design(), spec=make_spec()
    )

    assert len(result.outcomes) == 12 * len(POLICIES)
    assert {comparison.unit_count for comparison in result.comparisons} == {12}
    assert {outcome.controller_seed_count for outcome in result.outcomes} == {2}
    assert {outcome.turn_observation_count for outcome in result.outcomes} == {4}


def test_shuffled_inputs_produce_identical_canonical_input_and_result_hashes() -> None:
    records = list(make_records(superior_values()))
    first = evaluate_phase3(records, design=make_design(), spec=make_spec())
    Random(999).shuffle(records)
    shuffled = evaluate_phase3(records, design=make_design(), spec=make_spec())

    assert shuffled.input_sha256 == first.input_sha256
    assert shuffled.result_sha256 == first.result_sha256
    assert shuffled == first
    assert Phase3EvaluationResult.model_validate_json(first.model_dump_json()) == first

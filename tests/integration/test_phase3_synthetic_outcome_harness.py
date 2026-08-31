"""Fixture-driven, provider-free known-outcome checks for the Phase 3 evaluator.

``synthetic_effect_code`` is intentionally inert in production.  This test-only
harness consumes each checked-in code and constructs an independent evaluator
record grid.  It does not claim that the ordinary FakeProvider workflow creates
these score or guardrail patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from neurallm.evaluation import (
    DatasetPurpose,
    ExpectedEvaluationDesign,
    GuardrailName,
    GuardrailStatus,
    Phase3Verdict,
    SequenceExpectation,
    TurnEvaluationRecord,
    evaluate_phase3,
)
from neurallm.experiments.config import load_experiment_config
from neurallm.experiments.dataset import PromptSequence, load_dataset

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPOSITORY_ROOT / "configs/experiments/phase3-synthetic-evaluator.yaml"
_DATASET_PATH = _REPOSITORY_ROOT / "datasets/synthetic/phase3-evaluator-validation-v1.yaml"
_PROVIDER_IDENTITY_ID = "e" * 64
_HISTORY_COMMITMENT = "f" * 64

# Eight deterministic test-only model replicates provide enough independent
# matched units for the checked-in 95% inference rule. They are evaluator inputs,
# not provider calls and not an alteration of the frozen experiment plan.
_SCENARIO_MODEL_SEEDS = tuple(range(32_001, 32_009))
_EXPECTED_EFFECT_BY_SEQUENCE = {
    "synthetic-known-superior": 1.0,
    "synthetic-known-inferior": -1.0,
    "synthetic-identical-equivalent": 0.0,
    "synthetic-length-confound": 2.0,
}


@dataclass(frozen=True, slots=True)
class _PolicyOutcome:
    task_score: float
    instruction_adherence: float = 0.95
    response_length_tokens: int = 100
    repetition_ratio: float = 0.20
    action_magnitude: float = 0.10


@dataclass(frozen=True, slots=True)
class _ScenarioExpectation:
    verdict: Phase3Verdict
    alias_status: GuardrailStatus
    length_status: GuardrailStatus


_EXPECTATIONS = {
    1.0: _ScenarioExpectation(
        verdict=Phase3Verdict.SUPERIOR,
        alias_status=GuardrailStatus.PASS,
        length_status=GuardrailStatus.PASS,
    ),
    -1.0: _ScenarioExpectation(
        verdict=Phase3Verdict.INFERIOR,
        alias_status=GuardrailStatus.PASS,
        length_status=GuardrailStatus.PASS,
    ),
    0.0: _ScenarioExpectation(
        verdict=Phase3Verdict.EQUIVALENT,
        alias_status=GuardrailStatus.FAIL,
        length_status=GuardrailStatus.PASS,
    ),
    2.0: _ScenarioExpectation(
        verdict=Phase3Verdict.INFERIOR,
        alias_status=GuardrailStatus.PASS,
        length_status=GuardrailStatus.FAIL,
    ),
}


def _effect_code(sequence: PromptSequence) -> float:
    codes = {case.prompt_features.root["synthetic_effect_code"] for case in sequence.cases}
    assert len(codes) == 1, "one fixture sequence must encode exactly one scenario"
    return next(iter(codes))


def _scenario_outcomes(
    effect_code: float,
    *,
    focal_policy_id: str,
    comparator_policy_id: str,
    negative_control_policy_id: str,
) -> dict[str, _PolicyOutcome]:
    comparator = _PolicyOutcome(task_score=0.55)
    negative_control = _PolicyOutcome(task_score=0.45)
    if effect_code == 1.0:
        focal = _PolicyOutcome(task_score=0.85, action_magnitude=0.20)
    elif effect_code == -1.0:
        focal = _PolicyOutcome(task_score=0.25, action_magnitude=0.20)
    elif effect_code == 0.0:
        focal = comparator
    elif effect_code == 2.0:
        # The focal policy appears less repetitive only because its matched
        # response is half as long; the explicit length guardrail must reject it.
        focal = _PolicyOutcome(
            task_score=0.55,
            response_length_tokens=50,
            repetition_ratio=0.05,
            action_magnitude=0.20,
        )
    else:  # pragma: no cover - the assertion below reports unexpected fixture codes
        raise AssertionError(f"unsupported synthetic_effect_code: {effect_code}")
    return {
        focal_policy_id: focal,
        comparator_policy_id: comparator,
        negative_control_policy_id: negative_control,
    }


def _records_for_scenario(
    sequence: PromptSequence,
    *,
    dataset_sha256: str,
    controller_seeds: tuple[int, ...],
    focal_policy_id: str,
    outcomes: dict[str, _PolicyOutcome],
) -> tuple[TurnEvaluationRecord, ...]:
    records: list[TurnEvaluationRecord] = []
    for model_seed in _SCENARIO_MODEL_SEEDS:
        for controller_seed in controller_seeds:
            for policy_id, outcome in sorted(outcomes.items()):
                for turn_index, _case in enumerate(sequence.cases):
                    observes_history = policy_id == focal_policy_id and turn_index > 0
                    records.append(
                        TurnEvaluationRecord(
                            dataset_sha256=dataset_sha256,
                            prompt_sequence_id=sequence.sequence_id,
                            turn_index=turn_index,
                            policy_id=policy_id,
                            model_seed=model_seed,
                            controller_seed=controller_seed,
                            provider_identity_id=_PROVIDER_IDENTITY_ID,
                            has_previous_response=observes_history,
                            previous_history_commitment_sha256=(
                                _HISTORY_COMMITMENT if observes_history else None
                            ),
                            task_score=outcome.task_score,
                            instruction_adherence=outcome.instruction_adherence,
                            response_length_tokens=outcome.response_length_tokens,
                            repetition_ratio=outcome.repetition_ratio,
                            action_magnitude=outcome.action_magnitude,
                            action_within_bounds=True,
                            action_saturated=False,
                        )
                    )
    return tuple(records)


@pytest.mark.parametrize(
    "sequence_id",
    tuple(_EXPECTED_EFFECT_BY_SEQUENCE),
)
def test_checked_in_effect_codes_drive_independent_known_outcomes_without_a_provider(
    monkeypatch: pytest.MonkeyPatch,
    sequence_id: str,
) -> None:
    def _provider_construction_is_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("known-outcome evaluator harness must remain provider-free")

    monkeypatch.setattr(
        "neurallm.providers.fake.FakeProvider.__init__",
        _provider_construction_is_forbidden,
    )
    monkeypatch.setattr(
        "neurallm.providers.llama_cpp.LlamaCppProvider.__init__",
        _provider_construction_is_forbidden,
    )

    loaded_config = load_experiment_config(_CONFIG_PATH).config
    loaded_dataset = load_dataset(_DATASET_PATH).dataset
    sequence = next(
        sequence for sequence in loaded_dataset.sequences if sequence.sequence_id == sequence_id
    )
    effect_code = _effect_code(sequence)
    assert effect_code == _EXPECTED_EFFECT_BY_SEQUENCE[sequence_id]
    assert set(_EXPECTATIONS) == {
        _effect_code(fixture_sequence) for fixture_sequence in loaded_dataset.sequences
    }

    spec = loaded_config.evaluation
    assert spec is not None
    assert loaded_dataset.purpose is DatasetPurpose.SYNTHETIC
    assert len(spec.required_serious_comparator_ids) == 1
    assert len(spec.negative_control_policy_ids) == 1
    comparator_policy_id = spec.required_serious_comparator_ids[0]
    negative_control_policy_id = spec.negative_control_policy_ids[0]
    outcomes = _scenario_outcomes(
        effect_code,
        focal_policy_id=spec.focal_policy_id,
        comparator_policy_id=comparator_policy_id,
        negative_control_policy_id=negative_control_policy_id,
    )
    design = ExpectedEvaluationDesign(
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        dataset_sha256=loaded_dataset.dataset_hash,
        provider_identity_id=_PROVIDER_IDENTITY_ID,
        sequences=(
            SequenceExpectation(
                prompt_sequence_id=sequence.sequence_id,
                turn_count=len(sequence.cases),
            ),
        ),
        model_seeds=_SCENARIO_MODEL_SEEDS,
        controller_seeds=loaded_config.controller_seeds,
        policy_ids=loaded_config.configured_policy_ids,
    )
    result = evaluate_phase3(
        _records_for_scenario(
            sequence,
            dataset_sha256=loaded_dataset.dataset_hash,
            controller_seeds=loaded_config.controller_seeds,
            focal_policy_id=spec.focal_policy_id,
            outcomes=outcomes,
        ),
        design=design,
        spec=spec,
    )

    expectation = _EXPECTATIONS[effect_code]
    assert result.coverage.exact is True
    assert all(guardrail.status is GuardrailStatus.PASS for guardrail in result.global_guardrails)
    assert result.statistics_call_count == 4
    assert result.verdict is expectation.verdict

    serious = next(comparison for comparison in result.comparisons if comparison.serious_comparator)
    assert serious.comparator_policy_id == comparator_policy_id
    assert serious.unit_count == len(_SCENARIO_MODEL_SEEDS)
    assert serious.verdict is expectation.verdict
    guardrails = {guardrail.name: guardrail for guardrail in serious.guardrails}
    assert (
        guardrails[GuardrailName.INSTRUCTION_ADHERENCE_NON_REGRESSION].status
        is GuardrailStatus.PASS
    )
    assert guardrails[GuardrailName.BEHAVIORAL_ALIAS_DETECTION].status is expectation.alias_status
    assert guardrails[GuardrailName.RESPONSE_LENGTH_CONFOUND].status is expectation.length_status
    assert serious.behavioral_alias is (effect_code == 0.0)

    if effect_code == 0.0:
        alias = guardrails[GuardrailName.BEHAVIORAL_ALIAS_DETECTION]
        assert serious.behavioral_alias is True
        assert alias.observed_value == 0.0
        assert alias.threshold == spec.behavioral_alias_tolerance
    elif effect_code == 2.0:
        length = guardrails[GuardrailName.RESPONSE_LENGTH_CONFOUND]
        assert length.observed_value == pytest.approx(0.5)
        assert length.threshold == spec.maximum_length_reduction_ratio

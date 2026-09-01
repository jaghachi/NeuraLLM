"""Strict immutable models for the Phase 3 statistical evaluator.

The records in this module deliberately do not depend on providers, storage, or
the experiment runner.  They are the typed boundary between immutable run
evidence and the pure Phase 3 evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import (
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    Sha256Hex,
    SqliteInt64,
    UnitInterval,
)
from neurallm.domain.serialization import canonical_sha256


class _StrictFrozenModel(BaseModel):
    """Fail-closed base for canonical evaluation inputs and results."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class DatasetPurpose(StrEnum):
    """Dataset access role enforced at scientific workflow boundaries."""

    DEVELOPMENT = "development"
    EVALUATION = "evaluation"
    SYNTHETIC = "synthetic"


class Phase3Verdict(StrEnum):
    """Phase-3-only outcome vocabulary, never a final Phase 5 decision."""

    SUPERIOR = "superior"
    INFERIOR = "inferior"
    EQUIVALENT = "equivalent"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


class GuardrailStatus(StrEnum):
    """Outcome of one explicit evaluator guardrail."""

    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


class GuardrailName(StrEnum):
    """Versioned guardrails required by the experiment contract."""

    INSTRUCTION_ADHERENCE_NON_REGRESSION = "instruction_adherence_non_regression"
    RESPONSE_LENGTH_CONFOUND = "response_length_confound"
    MATCHED_CONDITION_COVERAGE = "matched_condition_coverage"
    PROVIDER_IDENTITY_STABILITY = "provider_identity_stability"
    TURN_ZERO_EQUIVALENCE = "turn_zero_equivalence"
    ACTION_BOUND_COMPLIANCE = "action_bound_compliance"
    ACTION_SATURATION_RATE = "action_saturation_rate"
    BEHAVIORAL_ALIAS_DETECTION = "behavioral_alias_detection"
    METRIC_AVAILABILITY = "metric_availability"


class EvaluationSpec(_StrictFrozenModel):
    """Complete preregistered Phase 3 statistical and guardrail contract."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["phase3-evaluator-v1"] = "phase3-evaluator-v1"
    primary_metric: Literal["task_score"] = "task_score"
    sequence_aggregation_version: Literal["mean-controller-seed-then-turn-v1"] = (
        "mean-controller-seed-then-turn-v1"
    )
    focal_policy_id: NonEmptyString
    required_serious_comparator_ids: tuple[NonEmptyString, ...]
    negative_control_policy_ids: tuple[NonEmptyString, ...] = ()
    bootstrap_resamples: int = Field(default=10_000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, allow_inf_nan=False)
    bootstrap_seed: SqliteInt64
    permutation_resamples: int = Field(default=10_000, ge=1)
    permutation_seed: SqliteInt64
    permutation_method_version: Literal["paired-sign-flip-exact-or-monte-carlo-v1"] = (
        "paired-sign-flip-exact-or-monte-carlo-v1"
    )
    multiplicity_correction_version: Literal["holm-v1"] = "holm-v1"
    practical_effect_threshold: float = Field(
        default=0.02,
        gt=0.0,
        allow_inf_nan=False,
    )
    equivalence_margin: float = Field(default=0.005, ge=0.0, allow_inf_nan=False)
    maximum_adherence_regression: float = Field(
        default=0.01,
        ge=0.0,
        allow_inf_nan=False,
    )
    maximum_length_reduction_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    maximum_action_saturation_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    required_matched_coverage: float = Field(default=1.0, ge=1.0, le=1.0)
    behavioral_alias_tolerance: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @field_validator(
        "required_serious_comparator_ids",
        "negative_control_policy_ids",
        mode="before",
    )
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("required_serious_comparator_ids", "negative_control_policy_ids")
    @classmethod
    def _sort_unique_policy_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("evaluation policy identifiers must not contain duplicates")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _validate_policy_roles_and_thresholds(self) -> Self:
        if not self.required_serious_comparator_ids:
            raise ValueError("at least one required serious comparator is required")
        comparator_ids = set(self.required_serious_comparator_ids)
        negative_ids = set(self.negative_control_policy_ids)
        if self.focal_policy_id in comparator_ids | negative_ids:
            raise ValueError("focal policy must not also be a comparator")
        if comparator_ids & negative_ids:
            raise ValueError("serious comparators and negative controls must be disjoint")
        if self.equivalence_margin > self.practical_effect_threshold:
            raise ValueError("equivalence margin must not exceed the practical threshold")
        return self


class SequenceExpectation(_StrictFrozenModel):
    """Expected number of correlated turns in one prompt sequence."""

    prompt_sequence_id: NonEmptyString
    turn_count: PositiveInt


class ExpectedEvaluationDesign(_StrictFrozenModel):
    """Exact condition grid expected before any statistical calculation."""

    schema_version: Literal[1] = 1
    dataset_purpose: DatasetPurpose
    dataset_sha256: Sha256Hex
    dataset_seal_sha256: Sha256Hex | None = None
    provider_identity_id: Sha256Hex
    sequences: tuple[SequenceExpectation, ...]
    model_seeds: tuple[SqliteInt64, ...]
    controller_seeds: tuple[SqliteInt64, ...]
    policy_ids: tuple[NonEmptyString, ...]

    @field_validator("dataset_purpose", mode="before")
    @classmethod
    def _accept_serialized_dataset_purpose(cls, value: object) -> object:
        if isinstance(value, str):
            return DatasetPurpose(value)
        return value

    @field_validator("sequences", mode="before")
    @classmethod
    def _accept_sequence_mapping(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return tuple(
                SequenceExpectation(prompt_sequence_id=key, turn_count=turn_count)
                for key, turn_count in value.items()
            )
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("model_seeds", "controller_seeds", "policy_ids", mode="before")
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("sequences")
    @classmethod
    def _sort_unique_sequences(
        cls,
        values: tuple[SequenceExpectation, ...],
    ) -> tuple[SequenceExpectation, ...]:
        if not values:
            raise ValueError("evaluation design requires at least one prompt sequence")
        identifiers = [value.prompt_sequence_id for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("prompt sequence identifiers must not contain duplicates")
        return tuple(sorted(values, key=lambda value: value.prompt_sequence_id))

    @field_validator("model_seeds", "controller_seeds")
    @classmethod
    def _sort_unique_nonempty_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("evaluation design axes must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("evaluation design axes must not contain duplicates")
        return tuple(sorted(values))

    @field_validator("policy_ids")
    @classmethod
    def _sort_unique_nonempty_policy_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not values:
            raise ValueError("evaluation design axes must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("evaluation design axes must not contain duplicates")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _reject_development_evaluation(self) -> Self:
        if self.dataset_purpose is DatasetPurpose.DEVELOPMENT:
            raise ValueError("the statistical evaluator cannot consume development data")
        if self.dataset_purpose is DatasetPurpose.EVALUATION:
            if self.dataset_seal_sha256 is None:
                raise ValueError("evaluation data requires a frozen dataset seal identity")
        elif self.dataset_seal_sha256 is not None:
            raise ValueError("only evaluation data may declare a dataset seal identity")
        return self


class TurnRecordKey(_StrictFrozenModel):
    """Complete logical key for one typed evaluator input record."""

    prompt_sequence_id: NonEmptyString
    turn_index: NonNegativeInt
    policy_id: NonEmptyString
    model_seed: SqliteInt64
    controller_seed: SqliteInt64


class MatchedUnitKey(_StrictFrozenModel):
    """Primary statistical unit: prompt sequence by model seed."""

    prompt_sequence_id: NonEmptyString
    model_seed: SqliteInt64


class TurnEvaluationRecord(_StrictFrozenModel):
    """Provider- and storage-independent evidence for one completed turn."""

    schema_version: Literal[1] = 1
    dataset_sha256: Sha256Hex
    prompt_sequence_id: NonEmptyString
    turn_index: NonNegativeInt
    policy_id: NonEmptyString
    model_seed: SqliteInt64
    controller_seed: SqliteInt64
    provider_identity_id: Sha256Hex
    has_previous_response: bool
    previous_history_commitment_sha256: Sha256Hex | None
    task_score: UnitInterval | None
    instruction_adherence: UnitInterval | None
    response_length_tokens: NonNegativeInt | None
    repetition_ratio: UnitInterval | None
    action_magnitude: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    action_within_bounds: bool = True
    action_saturated: bool = False

    @property
    def key(self) -> TurnRecordKey:
        """Return the exact condition key used by coverage validation."""

        return TurnRecordKey(
            prompt_sequence_id=self.prompt_sequence_id,
            turn_index=self.turn_index,
            policy_id=self.policy_id,
            model_seed=self.model_seed,
            controller_seed=self.controller_seed,
        )

    @property
    def required_metrics_available(self) -> bool:
        """Return whether every metric needed by Phase 3 is present."""

        return all(
            value is not None
            for value in (
                self.task_score,
                self.instruction_adherence,
                self.response_length_tokens,
                self.repetition_ratio,
            )
        )


class CoverageResult(_StrictFrozenModel):
    """Exact expected-versus-observed condition coverage."""

    exact: bool
    expected_count: NonNegativeInt
    observed_count: NonNegativeInt
    missing_keys: tuple[TurnRecordKey, ...] = ()
    unexpected_keys: tuple[TurnRecordKey, ...] = ()
    duplicate_keys: tuple[TurnRecordKey, ...] = ()

    @model_validator(mode="after")
    def _validate_exact_flag(self) -> Self:
        reconstructed_exact = (
            self.expected_count == self.observed_count
            and not self.missing_keys
            and not self.unexpected_keys
            and not self.duplicate_keys
        )
        if self.exact != reconstructed_exact:
            raise ValueError("coverage exact flag does not match its count and key evidence")
        return self


class GuardrailResult(_StrictFrozenModel):
    """One machine-readable guardrail result."""

    name: GuardrailName
    status: GuardrailStatus
    policy_id: NonEmptyString | None = None
    comparator_policy_id: NonEmptyString | None = None
    observed_value: float | None = Field(default=None, allow_inf_nan=False)
    threshold: float | None = Field(default=None, allow_inf_nan=False)
    detail: NonEmptyString


class SequencePolicyOutcome(_StrictFrozenModel):
    """Nested-turn/controller aggregation for one policy and matched unit."""

    unit_key: MatchedUnitKey
    policy_id: NonEmptyString
    controller_seed_count: PositiveInt
    turn_observation_count: PositiveInt
    task_score: UnitInterval
    instruction_adherence: UnitInterval
    response_length_tokens: float = Field(ge=0.0, allow_inf_nan=False)
    repetition_ratio: UnitInterval
    action_magnitude: float = Field(ge=0.0, allow_inf_nan=False)
    action_saturation_rate: UnitInterval


class BootstrapResult(_StrictFrozenModel):
    """Recorded paired-bootstrap estimate and percentile interval."""

    method_version: Literal["paired-bootstrap-percentile-v1"] = "paired-bootstrap-percentile-v1"
    seed: SqliteInt64
    resamples: PositiveInt
    confidence_level: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    sample_size: PositiveInt
    estimate: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)


class PermutationTestResult(_StrictFrozenModel):
    """Recorded paired sign-flip permutation result."""

    method_version: Literal["paired-sign-flip-exact-or-monte-carlo-v1"] = (
        "paired-sign-flip-exact-or-monte-carlo-v1"
    )
    seed: SqliteInt64
    requested_resamples: PositiveInt
    performed_permutations: PositiveInt
    exact: bool
    sample_size: PositiveInt
    observed_mean: float = Field(allow_inf_nan=False)
    p_value: UnitInterval


class HolmAdjustedPValue(_StrictFrozenModel):
    """One explicitly versioned member of a Holm family."""

    method_version: Literal["holm-v1"] = "holm-v1"
    comparator_policy_id: NonEmptyString
    raw_p_value: UnitInterval
    adjusted_p_value: UnitInterval
    rank: PositiveInt
    family_size: PositiveInt


class PairwiseComparisonResult(_StrictFrozenModel):
    """Focal-minus-comparator paired evidence at the sequence/model unit."""

    comparator_policy_id: NonEmptyString
    serious_comparator: bool
    unit_count: PositiveInt
    mean_difference: float = Field(allow_inf_nan=False)
    bootstrap: BootstrapResult
    permutation: PermutationTestResult
    holm: HolmAdjustedPValue | None = None
    behavioral_alias: bool
    guardrails: tuple[GuardrailResult, ...]
    verdict: Phase3Verdict


class Phase3EvaluationResult(_StrictFrozenModel):
    """Deterministic Phase-3-only decision skeleton and its full evidence."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["phase3-evaluator-v1"] = "phase3-evaluator-v1"
    claim_scope: Literal["phase-3-statistical-behavior-only"] = "phase-3-statistical-behavior-only"
    verdict: Phase3Verdict
    input_sha256: Sha256Hex
    result_sha256: Sha256Hex
    coverage: CoverageResult
    outcomes: tuple[SequencePolicyOutcome, ...]
    global_guardrails: tuple[GuardrailResult, ...]
    comparisons: tuple[PairwiseComparisonResult, ...]
    statistics_computed: bool
    statistics_call_count: NonNegativeInt

    @model_validator(mode="after")
    def _validate_result_hash_and_statistics_state(self) -> Self:
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"result_sha256"}))
        if self.result_sha256 != expected_hash:
            raise ValueError("result hash does not match the canonical evaluation result")
        if self.statistics_computed != (self.statistics_call_count > 0):
            raise ValueError("statistics flag must exactly match the recorded call count")
        return self


def phase3_result_sha256(payload: Mapping[str, object]) -> str:
    """Hash a complete result payload before constructing its validated model."""

    return canonical_sha256(payload)


__all__ = [
    "BootstrapResult",
    "CoverageResult",
    "DatasetPurpose",
    "EvaluationSpec",
    "ExpectedEvaluationDesign",
    "GuardrailName",
    "GuardrailResult",
    "GuardrailStatus",
    "HolmAdjustedPValue",
    "MatchedUnitKey",
    "PairwiseComparisonResult",
    "PermutationTestResult",
    "Phase3EvaluationResult",
    "Phase3Verdict",
    "SequenceExpectation",
    "SequencePolicyOutcome",
    "TurnEvaluationRecord",
    "TurnRecordKey",
    "phase3_result_sha256",
]

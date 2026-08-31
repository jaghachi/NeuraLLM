"""Frozen confirmatory analysis contract and complete result envelope."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from neurallm.domain.models import (
    NonEmptyString,
    NonNegativeInt,
    Sha256Hex,
    UnitInterval,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.attribution import (
    AttributionAnalysisSpec,
    PersistentStateAttributionResult,
)
from neurallm.evaluation.models import CoverageResult, MatchedUnitKey
from neurallm.evaluation.recovery import RecoveryAnalysisSpec, RecoveryEvaluationResult
from neurallm.evaluation.scientific import (
    EfficacyAnalysisSpec,
    EfficacyComparisonResult,
    ExperimentTier,
    GuardrailCleanTaskScore,
    LimitationDisposition,
    ScientificDecisionInput,
    ScientificDecisionRecord,
    ScientificFrozenModel,
    ScientificGuardrailResult,
    ScientificLimitation,
    decide_scientific_outcome,
)


class RecoveryEventSpec(ScientificFrozenModel):
    """One preregistered stressor and its ordered eligible recovery turns."""

    prompt_sequence_id: NonEmptyString
    stressor_turn_index: NonNegativeInt
    recovery_turn_indexes: tuple[NonNegativeInt, ...]
    minimum_task_score_target: UnitInterval
    maximum_repetition_ratio_target: UnitInterval

    @field_validator("recovery_turn_indexes", mode="before")
    @classmethod
    def _accept_yaml_turns(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_recovery_window(self) -> Self:
        if not self.recovery_turn_indexes:
            raise ValueError("a recovery event requires at least one eligible recovery turn")
        if self.recovery_turn_indexes != tuple(sorted(set(self.recovery_turn_indexes))):
            raise ValueError("recovery turn indexes must be sorted and unique")
        if any(index <= self.stressor_turn_index for index in self.recovery_turn_indexes):
            raise ValueError("recovery turns must occur strictly after the stressor")
        return self


class ConfirmatoryAnalysisSpec(ScientificFrozenModel):
    """Complete post-pilot contract frozen before confirmatory execution."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["confirmatory-analysis-v1"] = "confirmatory-analysis-v1"
    primary_endpoint: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    efficacy: EfficacyAnalysisSpec
    recovery: RecoveryAnalysisSpec
    attribution: AttributionAnalysisSpec
    recovery_events: tuple[RecoveryEventSpec, ...]
    optional_metric_dispositions: Mapping[NonEmptyString, LimitationDisposition]
    subgroup_fields: tuple[NonEmptyString, ...] = ("prompt_family",)
    subgroup_conflict_rule: Literal["opposite-resolved-serious-effect-v1"] = (
        "opposite-resolved-serious-effect-v1"
    )

    @field_validator("recovery_events", "subgroup_fields", mode="before")
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("recovery_events")
    @classmethod
    def _sort_unique_events(
        cls,
        values: tuple[RecoveryEventSpec, ...],
    ) -> tuple[RecoveryEventSpec, ...]:
        if not values:
            raise ValueError("confirmatory analysis requires preregistered recovery events")
        identifiers = tuple(event.prompt_sequence_id for event in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recovery events must target unique prompt sequences")
        return tuple(sorted(values, key=lambda event: event.prompt_sequence_id))

    @field_validator("optional_metric_dispositions", mode="before")
    @classmethod
    def _accept_serialized_optional_dispositions(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                name: (
                    LimitationDisposition(disposition)
                    if isinstance(disposition, str)
                    else disposition
                )
                for name, disposition in value.items()
            }
        return value

    @field_validator("optional_metric_dispositions")
    @classmethod
    def _freeze_optional_dispositions(
        cls,
        values: Mapping[str, LimitationDisposition],
    ) -> Mapping[str, LimitationDisposition]:
        if not values:
            raise ValueError("optional metric missingness dispositions must be explicit")
        return MappingProxyType(dict(sorted(values.items())))

    @field_serializer("optional_metric_dispositions")
    def _serialize_optional_dispositions(
        self,
        values: Mapping[str, LimitationDisposition],
    ) -> dict[str, str]:
        return {name: disposition.value for name, disposition in values.items()}

    @field_validator("subgroup_fields")
    @classmethod
    def _sort_unique_subgroups(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("subgroup fields must be nonempty, sorted, and unique")
        return values


class ScientificUnitOutcome(ScientificFrozenModel):
    """Auditable raw primary and recovery metrics for one efficacy unit."""

    unit_key: MatchedUnitKey
    policy_id: Literal[
        "best_static",
        "heuristic_adaptive",
        "neural_persistent",
        "random_matched",
    ]
    guardrail_clean_task_score: GuardrailCleanTaskScore
    instruction_adherence: UnitInterval
    repetition_ratio: UnitInterval
    response_length_tokens: float = Field(ge=0.0, allow_inf_nan=False)


class ConfirmatoryEvaluationResult(ScientificFrozenModel):
    """One hash-bound confirmatory result and all decision evidence families."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["confirmatory-evaluation-v1"] = "confirmatory-evaluation-v1"
    claim_scope: Literal["confirmatory-model-backed-scientific-decision"] = (
        "confirmatory-model-backed-scientific-decision"
    )
    analysis_contract_sha256: Sha256Hex
    causal_mechanism_validated: Literal[True] = True
    claim_eligible: bool
    run_manifest_sha256: Sha256Hex | None
    run_finalization_sha256: Sha256Hex | None
    input_sha256: Sha256Hex
    result_sha256: Sha256Hex
    coverage: CoverageResult
    unit_outcomes: tuple[ScientificUnitOutcome, ...]
    efficacy_comparisons: tuple[EfficacyComparisonResult, ...]
    recovery: RecoveryEvaluationResult
    attribution: PersistentStateAttributionResult
    guardrails: tuple[ScientificGuardrailResult, ...]
    limitations: tuple[ScientificLimitation, ...]
    decision: ScientificDecisionRecord
    statistics_call_count: NonNegativeInt

    @field_validator(
        "unit_outcomes",
        "efficacy_comparisons",
        "guardrails",
        "limitations",
        mode="before",
    )
    @classmethod
    def _accept_serialized_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_complete_result(self) -> Self:
        has_closed_run_bindings = (
            self.run_manifest_sha256 is not None and self.run_finalization_sha256 is not None
        )
        if self.claim_eligible != has_closed_run_bindings:
            raise ValueError(
                "claim eligibility requires both closed-run manifest and finalization bindings"
            )
        comparator_ids = tuple(
            comparison.comparator_policy_id for comparison in self.efficacy_comparisons
        )
        if comparator_ids != ("best_static", "heuristic_adaptive", "random_matched"):
            raise ValueError("confirmatory result requires exactly three efficacy comparisons")
        if not self.unit_outcomes:
            raise ValueError("confirmatory result requires auditable unit outcomes")
        decision_input = ScientificDecisionInput(
            tier=ExperimentTier.CONFIRMATORY,
            efficacy_comparisons=self.efficacy_comparisons,
            recovery=self.recovery.decision_gate,
            attribution=self.attribution.decision_gate,
            guardrails=self.guardrails,
            limitations=self.limitations,
        )
        if self.decision.decision_input_sha256 != canonical_sha256(decision_input):
            raise ValueError("scientific decision does not hash the enclosed evidence")
        if self.decision != decide_scientific_outcome(decision_input):
            raise ValueError("scientific decision does not match the enclosed evidence")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"result_sha256"}))
        if self.result_sha256 != expected_hash:
            raise ValueError("confirmatory result hash does not match its canonical evidence")
        return self


def confirmatory_result_sha256(payload: Mapping[str, object]) -> str:
    """Hash a complete result payload before validated construction."""

    return canonical_sha256(payload)


__all__ = [
    "ConfirmatoryAnalysisSpec",
    "ConfirmatoryEvaluationResult",
    "RecoveryEventSpec",
    "ScientificUnitOutcome",
    "confirmatory_result_sha256",
]

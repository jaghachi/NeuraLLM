"""Versioned contracts for model-backed experiment tiers and preregistration."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import NonEmptyString, PositiveInt, Sha256Hex
from neurallm.domain.serialization import canonical_sha256

MODEL_BACKED_POLICY_IDS = (
    "best_static",
    "heuristic_adaptive",
    "neural_matched_history_state_reset",
    "neural_persistent",
    "random_matched",
)
EFFICACY_POLICY_IDS = (
    "best_static",
    "heuristic_adaptive",
    "neural_persistent",
    "random_matched",
)
FOCAL_POLICY_ID = "neural_persistent"
REQUIRED_SERIOUS_COMPARATOR_IDS = (
    "best_static",
    "heuristic_adaptive",
)
NEGATIVE_CONTROL_POLICY_IDS = ("random_matched",)
ATTRIBUTION_POLICY_ID = "neural_matched_history_state_reset"
ATTRIBUTION_HISTORY_SOURCE_POLICY_ID = FOCAL_POLICY_ID

ENGINEERING_SMOKE_DECISION_RULE_VERSION = "engineering-smoke-no-scientific-decision-v1"
DEVELOPMENT_PILOT_DECISION_RULE_VERSION = "development-pilot-no-scientific-decision-v1"
CONFIRMATORY_DECISION_RULE_VERSION = "confirmatory-scientific-decision-v1"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class RunTier(StrEnum):
    """Scientific role of one explicitly scheduled model-backed run."""

    ENGINEERING_SMOKE = "engineering_smoke"
    DEVELOPMENT_PILOT = "development_pilot"
    CONFIRMATORY = "confirmatory"


class ScheduleSpec(_StrictFrozenModel):
    """Explicit dimensions and logical request count for one run schedule."""

    sequence_count: PositiveInt
    turns_per_sequence: PositiveInt
    model_seed_count: PositiveInt
    controller_seed_count: PositiveInt
    policy_count: Literal[5] = 5
    logical_generation_count: PositiveInt

    @model_validator(mode="after")
    def _validate_generation_count(self) -> Self:
        expected = (
            self.sequence_count
            * self.turns_per_sequence
            * self.model_seed_count
            * self.controller_seed_count
            * self.policy_count
        )
        if self.logical_generation_count != expected:
            raise ValueError(
                "logical_generation_count must equal the complete declared schedule product"
            )
        return self


class AttributionEdge(_StrictFrozenModel):
    """The only attribution-only history edge admitted to model-backed runs."""

    policy_id: Literal["neural_matched_history_state_reset"] = "neural_matched_history_state_reset"
    history_source_policy_id: Literal["neural_persistent"] = "neural_persistent"


class ExperimentProtocol(_StrictFrozenModel):
    """Complete tier, schedule, and policy-role topology for model-backed work."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["model-backed-experiment-protocol-v1"] = (
        "model-backed-experiment-protocol-v1"
    )
    run_tier: RunTier
    schedule: ScheduleSpec
    policy_ids: tuple[NonEmptyString, ...] = MODEL_BACKED_POLICY_IDS
    efficacy_policy_ids: tuple[NonEmptyString, ...] = EFFICACY_POLICY_IDS
    focal_policy_id: Literal["neural_persistent"] = "neural_persistent"
    required_serious_comparator_ids: tuple[NonEmptyString, ...] = REQUIRED_SERIOUS_COMPARATOR_IDS
    negative_control_policy_ids: tuple[NonEmptyString, ...] = NEGATIVE_CONTROL_POLICY_IDS
    attribution: AttributionEdge = Field(default_factory=AttributionEdge)

    @field_validator("run_tier", mode="before")
    @classmethod
    def _accept_yaml_run_tier(cls, value: object) -> object:
        if isinstance(value, str):
            return RunTier(value)
        return value

    @field_validator(
        "policy_ids",
        "efficacy_policy_ids",
        "required_serious_comparator_ids",
        "negative_control_policy_ids",
        mode="before",
    )
    @classmethod
    def _accept_yaml_policy_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_exact_policy_roles(self) -> Self:
        if self.policy_ids != MODEL_BACKED_POLICY_IDS:
            raise ValueError("model-backed protocol requires the exact five policy identifiers")
        if self.efficacy_policy_ids != EFFICACY_POLICY_IDS:
            raise ValueError("model-backed protocol requires the exact four efficacy policies")
        if self.required_serious_comparator_ids != REQUIRED_SERIOUS_COMPARATOR_IDS:
            raise ValueError("model-backed protocol requires both serious comparators")
        if self.negative_control_policy_ids != NEGATIVE_CONTROL_POLICY_IDS:
            raise ValueError(
                "model-backed protocol requires random_matched as its negative control"
            )
        if self.schedule.policy_count != len(self.policy_ids):
            raise ValueError("schedule policy_count must equal the exact policy-role count")
        return self

    @property
    def decision_rule_version(self) -> str:
        """Return the rule identity appropriate to this explicit run tier."""

        if self.run_tier is RunTier.ENGINEERING_SMOKE:
            return ENGINEERING_SMOKE_DECISION_RULE_VERSION
        if self.run_tier is RunTier.DEVELOPMENT_PILOT:
            return DEVELOPMENT_PILOT_DECISION_RULE_VERSION
        return CONFIRMATORY_DECISION_RULE_VERSION


class PreregistrationSeal(_StrictFrozenModel):
    """Published expected scientific identity for one confirmatory protocol."""

    schema_version: Literal[1] = 1
    seal_protocol_version: Literal["confirmatory-preregistration-seal-v1"] = (
        "confirmatory-preregistration-seal-v1"
    )
    experiment_id: NonEmptyString
    run_tier: Literal["confirmatory"] = "confirmatory"
    scientific_identity_sha256: Sha256Hex

    @property
    def seal_sha256(self) -> str:
        """Return the canonical identity of the published seal record."""

        return canonical_sha256(self)


__all__ = [
    "ATTRIBUTION_HISTORY_SOURCE_POLICY_ID",
    "ATTRIBUTION_POLICY_ID",
    "CONFIRMATORY_DECISION_RULE_VERSION",
    "DEVELOPMENT_PILOT_DECISION_RULE_VERSION",
    "EFFICACY_POLICY_IDS",
    "ENGINEERING_SMOKE_DECISION_RULE_VERSION",
    "FOCAL_POLICY_ID",
    "MODEL_BACKED_POLICY_IDS",
    "NEGATIVE_CONTROL_POLICY_IDS",
    "REQUIRED_SERIOUS_COMPARATOR_IDS",
    "AttributionEdge",
    "ExperimentProtocol",
    "PreregistrationSeal",
    "RunTier",
    "ScheduleSpec",
]

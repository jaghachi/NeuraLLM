"""Strict, immutable configuration records for control policies."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class BestStaticPolicySpec(_StrictFrozenModel):
    """Configuration identity for the development-selected static profile."""

    kind: Literal["best_static"] = "best_static"
    policy_id: Literal["best_static"] = "best_static"
    implementation_version: Literal["best-static-v1"] = "best-static-v1"
    state_schema_version: Literal["stateless-v1"] = "stateless-v1"
    history_access: Literal["none"] = "none"


class RandomMatchedPolicySpec(_StrictFrozenModel):
    """Configuration identity for deterministic bounded random actions."""

    kind: Literal["random_matched"] = "random_matched"
    policy_id: Literal["random_matched"] = "random_matched"
    implementation_version: Literal["random-matched-sha256-v1"] = "random-matched-sha256-v1"
    state_schema_version: Literal["random-matched-state-v1"] = "random-matched-state-v1"
    history_access: Literal["none"] = "none"


class HeuristicAdaptivePolicySpec(_StrictFrozenModel):
    """Transparent thresholds and reaction strengths for the heuristic arm."""

    kind: Literal["heuristic_adaptive"] = "heuristic_adaptive"
    policy_id: Literal["heuristic_adaptive"] = "heuristic_adaptive"
    implementation_version: Literal["heuristic-adaptive-v1"] = "heuristic-adaptive-v1"
    state_schema_version: Literal["heuristic-adaptive-state-v1"] = "heuristic-adaptive-state-v1"
    history_access: Literal["own_previous_response"] = "own_previous_response"
    high_repetition_threshold: float = Field(default=0.20, ge=0.0, lt=1.0)
    minimum_instruction_adherence: float = Field(default=0.90, gt=0.0, le=1.0)
    minimum_response_length_tokens: int = Field(default=8, ge=0)
    maximum_response_length_tokens: int = Field(default=256, gt=0)
    repetition_reaction_fraction: float = Field(default=0.75, gt=0.0, le=1.0)
    adherence_reaction_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    length_reaction_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    clean_decay_fraction: float = Field(default=0.50, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_response_length_band(self) -> Self:
        if self.minimum_response_length_tokens >= self.maximum_response_length_tokens:
            raise ValueError(
                "minimum_response_length_tokens must be less than maximum_response_length_tokens"
            )
        return self


class NeuralPersistentPolicySpec(_StrictFrozenModel):
    """Identity of the persistent simulated-neural controller."""

    kind: Literal["neural_persistent"] = "neural_persistent"
    policy_id: Literal["neural_persistent"] = "neural_persistent"
    implementation_version: Literal["simulated-neural-v1"] = "simulated-neural-v1"
    state_schema_version: Literal["simulated-neural-state-v1"] = "simulated-neural-state-v1"
    history_access: Literal["own_previous_response"] = "own_previous_response"
    state_intervention: Literal["none"] = "none"


class NeuralMatchedHistoryStateResetPolicySpec(_StrictFrozenModel):
    """Identity of the attribution-only matched-history substrate reset arm."""

    kind: Literal["neural_matched_history_state_reset"] = "neural_matched_history_state_reset"
    policy_id: Literal["neural_matched_history_state_reset"] = "neural_matched_history_state_reset"
    implementation_version: Literal["simulated-neural-v1"] = "simulated-neural-v1"
    state_schema_version: Literal["simulated-neural-state-v1"] = "simulated-neural-state-v1"
    history_access: Literal["matched_focal_previous_response"] = "matched_focal_previous_response"
    history_source_policy_id: Literal["neural_persistent"] = "neural_persistent"
    state_intervention: Literal["reset_substrate"] = "reset_substrate"


PolicySpec = Annotated[
    BestStaticPolicySpec
    | RandomMatchedPolicySpec
    | HeuristicAdaptivePolicySpec
    | NeuralPersistentPolicySpec
    | NeuralMatchedHistoryStateResetPolicySpec,
    Field(discriminator="kind"),
]


__all__ = [
    "BestStaticPolicySpec",
    "HeuristicAdaptivePolicySpec",
    "NeuralMatchedHistoryStateResetPolicySpec",
    "NeuralPersistentPolicySpec",
    "PolicySpec",
    "RandomMatchedPolicySpec",
]

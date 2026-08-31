"""Strict immutable value objects for the NeuraLLM experiment kernel."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[
    float,
    Field(gt=0.0, allow_inf_nan=False),
]
TopP = Annotated[
    float,
    Field(gt=0.0, le=1.0, allow_inf_nan=False),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
SqliteInt64 = Annotated[int, Field(ge=-(2**63), le=2**63 - 1)]
NonEmptyString = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_non_blank),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
GitCommitSha = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{40}$"),
]
UnitInterval = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]

_PHASE4_MATCHED_HISTORY_POLICY_SOURCES = {
    "neural_matched_history_state_reset": "neural_persistent",
}
_MODEL_BACKED_POLICY_IDS = {
    "best_static",
    "heuristic_adaptive",
    "neural_matched_history_state_reset",
    "neural_persistent",
    "random_matched",
}
_MODEL_BACKED_RULE_TIERS = {
    "engineering-smoke-no-scientific-decision-v1": "engineering_smoke",
    "development-pilot-no-scientific-decision-v1": "development_pilot",
    "confirmatory-scientific-decision-v1": "confirmatory",
    "confirmatory-scientific-decision-v2": "confirmatory",
}


class StrictFrozenModel(BaseModel):
    """Shared fail-closed configuration for scientific domain records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class PromptFeatures(RootModel[Mapping[NonEmptyString, FiniteFloat]]):
    """An immutable, string-keyed collection of finite prompt features."""

    model_config = ConfigDict(frozen=True, strict=True, validate_default=True)

    @field_validator("root")
    @classmethod
    def _freeze_values(
        cls,
        value: Mapping[str, float],
    ) -> Mapping[str, float]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("root")
    def _serialize_values(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    def __getitem__(self, key: str) -> float:
        return self.root[key]

    def __len__(self) -> int:
        return len(self.root)


class DecodingParameters(StrictFrozenModel):
    """Complete generation settings, including the fixed generation budget."""

    temperature: PositiveFiniteFloat
    top_p: TopP
    top_k: NonNegativeInt
    presence_penalty: FiniteFloat
    max_tokens: PositiveInt
    seed: SqliteInt64


class ControllerAction(StrictFrozenModel):
    """Finite per-turn movement proposed by a control policy.

    The generation budget is deliberately absent. ``max_tokens`` belongs to
    :class:`DecodingParameters` and cannot be changed through this action. A
    run's explicit :class:`ActionBounds` validates the action; the initial
    pilot defaults are configuration values rather than immutable type limits.
    """

    temperature_delta: FiniteFloat
    top_p_delta: FiniteFloat
    top_k_delta: int
    presence_penalty_delta: FiniteFloat


class MetricValue[MetricScalarT: (int, float)](StrictFrozenModel):
    """A value together with the provenance needed to interpret it."""

    value: MetricScalarT | None
    availability: bool
    metric_version: NonEmptyString
    input_hash: Sha256Hex

    @field_validator("value")
    @classmethod
    def _reject_nonfinite_value(
        cls,
        value: MetricScalarT | None,
    ) -> MetricScalarT | None:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("metric value must be finite")
        return value

    @model_validator(mode="after")
    def _validate_availability(self) -> Self:
        if self.availability != (self.value is not None):
            raise ValueError("availability must be true exactly when value is present")
        return self


FloatMetricValue = MetricValue[FiniteFloat]
UnitIntervalMetricValue = MetricValue[UnitInterval]
CountMetricValue = MetricValue[NonNegativeInt]


class ResponseMetrics(StrictFrozenModel):
    """Causally available metrics computed for one completed response."""

    task_score: UnitIntervalMetricValue
    instruction_adherence: UnitIntervalMetricValue
    response_length_tokens: CountMetricValue
    repetition_ratio: UnitIntervalMetricValue
    repeated_3_gram_ratio: UnitIntervalMetricValue
    repeated_4_gram_ratio: UnitIntervalMetricValue
    distinct_2: UnitIntervalMetricValue
    distinct_3: UnitIntervalMetricValue
    late_window_repetition_ratio: UnitIntervalMetricValue
    format_validity: UnitIntervalMetricValue
    semantic_similarity: UnitIntervalMetricValue


class ControllerObservation(StrictFrozenModel):
    """Only information available when a policy selects the next action."""

    turn_index: NonNegativeInt
    prompt_family: NonEmptyString
    current_prompt_features: PromptFeatures
    previous_response_metrics: ResponseMetrics | None
    has_previous_response: bool

    @model_validator(mode="after")
    def _validate_history(self) -> Self:
        history_is_present = self.previous_response_metrics is not None
        if self.has_previous_response != history_is_present:
            raise ValueError("has_previous_response must exactly match previous_response_metrics")
        if self.turn_index == 0 and history_is_present:
            raise ValueError("turn zero requires explicit null/false previous-response history")
        return self


class ExperimentCondition(StrictFrozenModel):
    """The complete logical identity of one policy/turn condition."""

    experiment_id: NonEmptyString
    dataset_version: NonEmptyString
    prompt_sequence_id: NonEmptyString
    turn_index: NonNegativeInt
    policy_id: NonEmptyString
    model_seed: SqliteInt64
    controller_seed: SqliteInt64
    provider_identity_id: Sha256Hex
    base_decoding_profile_id: NonEmptyString

    @property
    def condition_id(self) -> str:
        from neurallm.domain.identifiers import condition_id

        return condition_id(self)


class ProviderIdentity(StrictFrozenModel):
    """Stable provider/model/build identity recorded before generation."""

    provider_type: NonEmptyString
    implementation_version: NonEmptyString
    model_alias: NonEmptyString
    build_id: NonEmptyString
    provider_config_hash: Sha256Hex
    model_path: NonEmptyString | None = None
    model_sha256: Sha256Hex | None = None
    chat_template_sha256: Sha256Hex | None = None

    @property
    def identity_id(self) -> str:
        from neurallm.domain.identifiers import provider_identity_id

        return provider_identity_id(self)


class SeedSchedule(StrictFrozenModel):
    """The two independent deterministic seed streams for a run."""

    model_seeds: tuple[SqliteInt64, ...]
    controller_seeds: tuple[SqliteInt64, ...]

    @model_validator(mode="after")
    def _validate_seed_streams(self) -> Self:
        for name, values in (
            ("model_seeds", self.model_seeds),
            ("controller_seeds", self.controller_seeds),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        return self


class ActionBounds(StrictFrozenModel):
    """Frozen action bounds recorded in each run manifest."""

    temperature_delta: tuple[FiniteFloat, FiniteFloat] = (-0.10, 0.10)
    top_p_delta: tuple[FiniteFloat, FiniteFloat] = (-0.05, 0.05)
    top_k_delta: tuple[int, int] = (-10, 10)
    presence_penalty_delta: tuple[FiniteFloat, FiniteFloat] = (-0.20, 0.20)

    @model_validator(mode="after")
    def _validate_intervals(self) -> Self:
        for field_name in (
            "temperature_delta",
            "top_p_delta",
            "top_k_delta",
            "presence_penalty_delta",
        ):
            lower, upper = getattr(self, field_name)
            if lower > upper:
                raise ValueError(f"{field_name} lower bound exceeds upper bound")
            if not lower <= 0 <= upper:
                raise ValueError(f"{field_name} bounds must include zero")
        return self

    def contains(self, action: ControllerAction) -> bool:
        """Return whether an action satisfies these frozen run bounds."""

        return (
            self.temperature_delta[0] <= action.temperature_delta <= self.temperature_delta[1]
            and self.top_p_delta[0] <= action.top_p_delta <= self.top_p_delta[1]
            and self.top_k_delta[0] <= action.top_k_delta <= self.top_k_delta[1]
            and self.presence_penalty_delta[0]
            <= action.presence_penalty_delta
            <= self.presence_penalty_delta[1]
        )

    def require(self, action: ControllerAction) -> ControllerAction:
        """Return a valid action or fail closed on a bound violation."""

        if not self.contains(action):
            raise ValueError("controller action exceeds the configured run bounds")
        return action


class DecodingBounds(StrictFrozenModel):
    """Legal absolute decoding ranges frozen into an experiment plan."""

    temperature: tuple[PositiveFiniteFloat, PositiveFiniteFloat] = (0.01, 2.0)
    top_p: tuple[TopP, TopP] = (0.01, 1.0)
    top_k: tuple[NonNegativeInt, NonNegativeInt] = (0, 200)
    presence_penalty: tuple[FiniteFloat, FiniteFloat] = (-2.0, 2.0)

    @model_validator(mode="after")
    def _validate_intervals(self) -> Self:
        for field_name in ("temperature", "top_p", "top_k", "presence_penalty"):
            lower, upper = getattr(self, field_name)
            if lower > upper:
                raise ValueError(f"{field_name} lower bound exceeds upper bound")
        return self


class RunManifest(StrictFrozenModel):
    """Immutable provenance and scientific configuration for a run."""

    source_commit: GitCommitSha
    working_tree_clean: bool
    experiment_config_hash: Sha256Hex
    dataset_hash: Sha256Hex
    provider_config_hash: Sha256Hex
    provider_identity: ProviderIdentity
    provider_effective_configuration_json: NonEmptyString
    policy_config_hashes: Mapping[NonEmptyString, Sha256Hex]
    matched_history_policy_sources: Mapping[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    metric_versions: Mapping[NonEmptyString, NonEmptyString]
    seed_schedule: SeedSchedule
    action_bounds: ActionBounds
    decoding_bounds: DecodingBounds = DecodingBounds()
    decision_rule_version: NonEmptyString
    database_schema_version: PositiveInt
    evaluation_spec_json: NonEmptyString | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    evaluation_spec_sha256: Sha256Hex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    turn_input_evidence_sha256: Sha256Hex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    phase3_analysis_contract_sha256: Sha256Hex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    run_tier: NonEmptyString | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    scientific_identity_sha256: Sha256Hex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    preregistration_sha256: Sha256Hex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    confirmatory_analysis_contract_sha256: Sha256Hex | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("policy_config_hashes", "metric_versions")
    @classmethod
    def _freeze_mappings(
        cls,
        value: Mapping[str, str],
    ) -> Mapping[str, str]:
        if not value:
            raise ValueError("manifest mappings must not be empty")
        return MappingProxyType(dict(sorted(value.items())))

    @field_validator("matched_history_policy_sources")
    @classmethod
    def _freeze_matched_history_sources(
        cls,
        value: Mapping[str, str],
    ) -> Mapping[str, str]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("policy_config_hashes", "metric_versions")
    def _serialize_mappings(
        self,
        value: Mapping[str, str],
    ) -> dict[str, str]:
        return dict(value)

    @field_serializer("matched_history_policy_sources")
    def _serialize_matched_history_sources(
        self,
        value: Mapping[str, str],
    ) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def _validate_provider_config_hash(self) -> Self:
        configured_policies = set(self.policy_config_hashes)
        matched_sources = dict(self.matched_history_policy_sources)
        if matched_sources and matched_sources != _PHASE4_MATCHED_HISTORY_POLICY_SOURCES:
            raise ValueError("manifest permits only the frozen matched-history policy edge")
        phase4 = self.decision_rule_version == "phase4-neural-mechanism-only-v1"
        model_backed_tier = _MODEL_BACKED_RULE_TIERS.get(self.decision_rule_version)
        has_attribution_edge = bool(self.matched_history_policy_sources)
        if phase4 and not has_attribution_edge:
            raise ValueError("Phase 4 decision rule requires its matched-history edge")
        if not phase4 and model_backed_tier is None and has_attribution_edge:
            raise ValueError("matched-history edge requires a causal attribution protocol")
        if phase4 and configured_policies != {
            "neural_persistent",
            "neural_matched_history_state_reset",
        }:
            raise ValueError("Phase 4 manifest requires exactly the two neural policies")
        if model_backed_tier is not None:
            if not has_attribution_edge or configured_policies != _MODEL_BACKED_POLICY_IDS:
                raise ValueError(
                    "model-backed manifest requires the exact five policies and attribution edge"
                )
            if self.run_tier != model_backed_tier:
                raise ValueError("model-backed run tier does not match its decision rule")
            if self.scientific_identity_sha256 is None:
                raise ValueError("model-backed manifest requires the frozen scientific identity")
            confirmatory = model_backed_tier == "confirmatory"
            if confirmatory != (self.preregistration_sha256 is not None):
                raise ValueError(
                    "confirmatory manifest and preregistration identity must appear together"
                )
            if confirmatory != (self.confirmatory_analysis_contract_sha256 is not None):
                raise ValueError(
                    "confirmatory manifest and final analysis contract must appear together"
                )
            if confirmatory != (self.turn_input_evidence_sha256 is not None):
                raise ValueError(
                    "confirmatory manifest and frozen turn-input identity must appear together"
                )
        elif (
            self.run_tier is not None
            or self.scientific_identity_sha256 is not None
            or self.preregistration_sha256 is not None
            or self.confirmatory_analysis_contract_sha256 is not None
            or self.turn_input_evidence_sha256 is not None
        ):
            raise ValueError("only model-backed manifests may carry protocol identities")
        for policy_id, source_policy_id in self.matched_history_policy_sources.items():
            if policy_id == source_policy_id:
                raise ValueError("matched history source must name another policy")
            if policy_id not in configured_policies or source_policy_id not in configured_policies:
                raise ValueError("matched history policies must both exist in policy_config_hashes")
        if self.provider_config_hash != self.provider_identity.provider_config_hash:
            raise ValueError(
                "provider_config_hash must match provider_identity.provider_config_hash"
            )
        try:
            effective: object = json.loads(self.provider_effective_configuration_json)
            if not isinstance(effective, dict) or not all(
                isinstance(key, str) for key in effective
            ):
                raise ValueError("provider effective configuration must be a JSON object")
            from neurallm.domain.serialization import canonical_json, canonical_sha256

            if canonical_json(effective) != self.provider_effective_configuration_json:
                raise ValueError("provider effective configuration must be canonical JSON")
            if canonical_sha256(effective) != self.provider_config_hash:
                raise ValueError("provider effective configuration hash does not match manifest")
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "provider effective configuration must be finite canonical JSON"
            ) from exc
        if (self.evaluation_spec_json is None) != (self.evaluation_spec_sha256 is None):
            raise ValueError("evaluation spec JSON and SHA-256 must appear together")
        if self.evaluation_spec_json is not None:
            assert self.evaluation_spec_sha256 is not None
            try:
                evaluation_spec: object = json.loads(self.evaluation_spec_json)
                if not isinstance(evaluation_spec, dict) or not all(
                    isinstance(key, str) for key in evaluation_spec
                ):
                    raise ValueError("evaluation spec must be a JSON object")
                if canonical_json(evaluation_spec) != self.evaluation_spec_json:
                    raise ValueError("evaluation spec must be canonical JSON")
                if canonical_sha256(evaluation_spec) != self.evaluation_spec_sha256:
                    raise ValueError("evaluation spec hash does not match manifest")
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError("evaluation spec must be finite canonical JSON") from exc
        phase3 = self.decision_rule_version == "phase3-baseline-evaluator-v1"
        if phase3 and self.phase3_analysis_contract_sha256 is None:
            raise ValueError("Phase 3 run manifest requires its pre-execution analysis contract")
        if not phase3 and self.phase3_analysis_contract_sha256 is not None:
            raise ValueError("only a Phase 3 run manifest may carry an analysis contract")
        return self

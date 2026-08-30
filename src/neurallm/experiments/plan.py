"""Deterministic expansion of validated scientific inputs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.experiments.config import LoadedExperimentConfig
from neurallm.experiments.dataset import LoadedDataset
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.base import GenerationRequest
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION

PHASE2_DECISION_RULE_VERSION = "phase2-no-scientific-decision-v1"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PlannedTurn(_StrictFrozenModel):
    """One deterministic logical request in the ordered schedule."""

    condition: ExperimentCondition
    prompt_case_id: str = Field(min_length=1)
    prompt_family: str = Field(min_length=1)
    prompt_features: PromptFeatures
    prompt: str = Field(min_length=1)
    validator: ValidatorSpec
    decoding_parameters: DecodingParameters

    @property
    def generation_request(self) -> GenerationRequest:
        return GenerationRequest(
            prompt=self.prompt,
            decoding_parameters=self.decoding_parameters,
            condition=self.condition,
        )

    @property
    def logical_request_sha256(self) -> str:
        return canonical_sha256(self.generation_request)


class ExperimentPlan(_StrictFrozenModel):
    """Complete ordered schedule and scientific artifact identity."""

    experiment_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    experiment_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_identity: ProviderIdentity
    provider_effective_configuration_json: str = Field(min_length=2)
    action_bounds: ActionBounds
    decoding_bounds: DecodingBounds
    metric_versions: Mapping[str, str]
    decision_rule_version: str = Field(min_length=1)
    database_schema_version: int = Field(gt=0)
    turns: tuple[PlannedTurn, ...]

    @field_validator("metric_versions")
    @classmethod
    def _freeze_metric_versions(cls, values: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(sorted(values.items())))

    @field_serializer("metric_versions")
    def _serialize_metric_versions(self, values: Mapping[str, str]) -> dict[str, str]:
        return dict(values)

    @field_validator("turns")
    @classmethod
    def _validate_turns(cls, values: tuple[PlannedTurn, ...]) -> tuple[PlannedTurn, ...]:
        if not values:
            raise ValueError("experiment plan must contain at least one turn")
        condition_ids = [turn.condition.condition_id for turn in values]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("experiment plan contains duplicate conditions")
        request_ids = [turn.logical_request_sha256 for turn in values]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("experiment plan contains duplicate logical requests")
        return values

    @property
    def scientific_identity_sha256(self) -> str:
        return canonical_sha256(self)


def build_plan(
    loaded_config: LoadedExperimentConfig,
    loaded_dataset: LoadedDataset,
) -> ExperimentPlan:
    """Expand inputs in a host- and input-iteration-independent order."""

    if not isinstance(loaded_config, LoadedExperimentConfig):
        raise TypeError("loaded_config must be a LoadedExperimentConfig")
    if not isinstance(loaded_dataset, LoadedDataset):
        raise TypeError("loaded_dataset must be a LoadedDataset")
    config = loaded_config.config
    dataset = loaded_dataset.dataset
    if dataset.version != config.dataset.version:
        raise ValueError("dataset version does not match experiment configuration")
    if dict(config.metric_versions) != METRIC_VERSIONS:
        raise ValueError("configured metric versions do not match the implementation")
    if config.database_schema_version != CURRENT_SCHEMA_VERSION:
        raise ValueError("configured database schema version does not match the implementation")
    if config.decision_rule_version != PHASE2_DECISION_RULE_VERSION:
        raise ValueError("configured decision rule version does not match Phase 2")

    turns: list[PlannedTurn] = []
    provider_identity_id = config.provider.expected_identity.identity_id
    for sequence in sorted(dataset.sequences, key=lambda item: item.sequence_id):
        for policy_id in sorted(config.policy_ids):
            for model_seed in sorted(config.model_seeds):
                parameters = config.base_decoding_profile.with_seed(model_seed)
                for controller_seed in sorted(config.controller_seeds):
                    for turn_index, prompt_case in enumerate(sequence.cases):
                        condition = ExperimentCondition(
                            experiment_id=config.experiment_id,
                            dataset_version=dataset.version,
                            prompt_sequence_id=sequence.sequence_id,
                            turn_index=turn_index,
                            policy_id=policy_id,
                            model_seed=model_seed,
                            controller_seed=controller_seed,
                            provider_identity_id=provider_identity_id,
                            base_decoding_profile_id=config.base_decoding_profile_id,
                        )
                        turns.append(
                            PlannedTurn(
                                condition=condition,
                                prompt_case_id=prompt_case.case_id,
                                prompt_family=prompt_case.prompt_family,
                                prompt_features=prompt_case.prompt_features,
                                prompt=prompt_case.prompt,
                                validator=prompt_case.validator,
                                decoding_parameters=parameters,
                            )
                        )

    return ExperimentPlan(
        experiment_id=config.experiment_id,
        dataset_version=dataset.version,
        experiment_config_hash=config.experiment_config_hash,
        dataset_hash=dataset.dataset_hash,
        provider_identity=config.provider.expected_identity,
        provider_effective_configuration_json=(
            config.provider.expected_effective_configuration_json
        ),
        action_bounds=config.action_bounds,
        decoding_bounds=config.decoding_bounds,
        metric_versions=config.metric_versions,
        decision_rule_version=config.decision_rule_version,
        database_schema_version=config.database_schema_version,
        turns=tuple(turns),
    )


__all__ = [
    "PHASE2_DECISION_RULE_VERSION",
    "ExperimentPlan",
    "PlannedTurn",
    "build_plan",
]

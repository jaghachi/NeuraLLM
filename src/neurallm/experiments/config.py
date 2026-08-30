"""Strict experiment configuration and explicit path resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    DecodingParameters,
    ProviderIdentity,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.experiments.yaml_loader import load_yaml_mapping


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class DatasetReference(_StrictFrozenModel):
    """Logical dataset identity plus an explicit source reference."""

    path: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @field_validator("path", "version")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class ProviderSelection(_StrictFrozenModel):
    """Explicit provider kind, expected identity, and optional config path."""

    kind: Literal["fake", "llama_cpp"]
    expected_identity: ProviderIdentity
    expected_effective_configuration_json: str = Field(min_length=2)
    config_path: str | None = None

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        if self.expected_identity.provider_type != self.kind:
            raise ValueError("expected provider identity type must match provider kind")
        if self.kind == "fake" and self.config_path is not None:
            raise ValueError("fake provider does not accept a provider config path")
        if self.kind == "llama_cpp" and (self.config_path is None or not self.config_path.strip()):
            raise ValueError("llama_cpp provider requires an explicit config_path")
        try:
            effective: object = json.loads(self.expected_effective_configuration_json)
            if not isinstance(effective, dict) or not all(
                isinstance(key, str) for key in effective
            ):
                raise ValueError("expected effective configuration must be a JSON object")
            if canonical_json(effective) != self.expected_effective_configuration_json:
                raise ValueError("expected effective configuration must be canonical JSON")
            if canonical_sha256(effective) != self.expected_identity.provider_config_hash:
                raise ValueError(
                    "expected effective configuration does not match provider_config_hash"
                )
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "expected effective configuration must be finite canonical JSON"
            ) from exc
        return self


class BaseDecodingProfile(_StrictFrozenModel):
    """Generation settings frozen by the plan before model-seed expansion."""

    temperature: float = Field(gt=0.0, allow_inf_nan=False)
    top_p: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    top_k: int = Field(ge=0)
    presence_penalty: float = Field(allow_inf_nan=False)
    max_tokens: int = Field(gt=0)

    def with_seed(self, seed: int) -> DecodingParameters:
        return DecodingParameters(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
            seed=seed,
        )


class ExperimentConfig(_StrictFrozenModel):
    """Complete Phase 2 scientific configuration before dataset expansion."""

    schema_version: Literal[1]
    experiment_id: str = Field(min_length=1)
    dataset: DatasetReference
    provider: ProviderSelection
    policy_ids: tuple[str, ...]
    model_seeds: tuple[int, ...]
    controller_seeds: tuple[int, ...]
    base_decoding_profile_id: str = Field(min_length=1)
    base_decoding_profile: BaseDecodingProfile
    action_bounds: ActionBounds
    decoding_bounds: DecodingBounds
    metric_versions: Mapping[str, str]
    decision_rule_version: str = Field(min_length=1)
    database_schema_version: int = Field(gt=0)
    artifact_root: str = Field(min_length=1)

    @field_validator("policy_ids", "model_seeds", "controller_seeds", mode="before")
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("action_bounds", mode="before")
    @classmethod
    def _accept_yaml_action_bounds(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in (
            "temperature_delta",
            "top_p_delta",
            "top_k_delta",
            "presence_penalty_delta",
        ):
            field_value = normalized.get(field_name)
            if isinstance(field_value, list):
                normalized[field_name] = tuple(field_value)
        return normalized

    @field_validator("decoding_bounds", mode="before")
    @classmethod
    def _accept_yaml_decoding_bounds(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in (
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
        ):
            field_value = normalized.get(field_name)
            if isinstance(field_value, list):
                normalized[field_name] = tuple(field_value)
        return normalized

    @field_validator(
        "experiment_id",
        "base_decoding_profile_id",
        "decision_rule_version",
        "artifact_root",
    )
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("policy_ids")
    @classmethod
    def _validate_policy_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("policy_ids must not be empty")
        if any(not value.strip() for value in values):
            raise ValueError("policy_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("policy_ids must not contain duplicates")
        return tuple(sorted(values))

    @field_validator("model_seeds", "controller_seeds")
    @classmethod
    def _sort_unique_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("seed schedules must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("seed schedules must not contain duplicates")
        return tuple(sorted(values))

    @field_validator("metric_versions")
    @classmethod
    def _freeze_metric_versions(cls, values: Mapping[str, str]) -> Mapping[str, str]:
        if not values:
            raise ValueError("metric_versions must not be empty")
        normalized: dict[str, str] = {}
        for name, version in values.items():
            if not name.strip() or not version.strip():
                raise ValueError("metric names and versions must not be blank")
            normalized[name] = version
        return MappingProxyType(dict(sorted(normalized.items())))

    @field_serializer("metric_versions")
    def _serialize_metric_versions(self, values: Mapping[str, str]) -> dict[str, str]:
        return dict(values)

    @model_validator(mode="after")
    def _validate_seed_schedules(self) -> Self:
        SeedSchedule(
            model_seeds=self.model_seeds,
            controller_seeds=self.controller_seeds,
        )
        return self

    @property
    def experiment_config_hash(self) -> str:
        """Hash scientific configuration while excluding machine-local paths."""

        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "dataset_version": self.dataset.version,
                "provider_kind": self.provider.kind,
                "provider_identity": self.provider.expected_identity,
                "provider_effective_configuration_json": (
                    self.provider.expected_effective_configuration_json
                ),
                "policy_ids": self.policy_ids,
                "model_seeds": self.model_seeds,
                "controller_seeds": self.controller_seeds,
                "base_decoding_profile_id": self.base_decoding_profile_id,
                "base_decoding_profile": self.base_decoding_profile,
                "action_bounds": self.action_bounds,
                "decoding_bounds": self.decoding_bounds,
                "metric_versions": self.metric_versions,
                "decision_rule_version": self.decision_rule_version,
                "database_schema_version": self.database_schema_version,
            }
        )


@dataclass(frozen=True, slots=True)
class LoadedExperimentConfig:
    """Validated config plus explicit resolved incidental paths."""

    config: ExperimentConfig
    source_path: Path
    dataset_path: Path
    provider_config_path: Path | None
    artifact_root: Path


def _resolve_reference(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def load_experiment_config(path: Path) -> LoadedExperimentConfig:
    """Validate one explicit YAML configuration and resolve its references."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    source_path = path.expanduser().resolve(strict=True)
    config = ExperimentConfig.model_validate(load_yaml_mapping(source_path))
    base = source_path.parent
    dataset_path = _resolve_reference(base, config.dataset.path)
    provider_path = (
        None
        if config.provider.config_path is None
        else _resolve_reference(base, config.provider.config_path)
    )
    artifact_root = _resolve_reference(base, config.artifact_root)
    return LoadedExperimentConfig(
        config=config,
        source_path=source_path,
        dataset_path=dataset_path,
        provider_config_path=provider_path,
        artifact_root=artifact_root,
    )


__all__ = [
    "BaseDecodingProfile",
    "DatasetReference",
    "ExperimentConfig",
    "LoadedExperimentConfig",
    "ProviderSelection",
    "load_experiment_config",
]

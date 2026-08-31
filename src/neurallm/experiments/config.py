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

from neurallm.control.specs import PolicySpec
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    DecodingParameters,
    ProviderIdentity,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose, EvaluationSpec
from neurallm.evaluation.selection import StaticSelectionRecord
from neurallm.experiments.dataset import DatasetSeal
from neurallm.experiments.yaml_loader import load_yaml_mapping


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class DatasetReference(_StrictFrozenModel):
    """Logical dataset identity plus an explicit source reference."""

    path: str = Field(min_length=1)
    version: str = Field(min_length=1)
    purpose: DatasetPurpose | None = None
    expected_dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    seal: DatasetSeal | None = None

    @field_validator("purpose", mode="before")
    @classmethod
    def _accept_yaml_purpose(cls, value: object) -> object:
        if isinstance(value, str):
            return DatasetPurpose(value)
        return value

    @field_validator("path", "version")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_phase3_identity(self) -> Self:
        if self.purpose is None:
            if self.expected_dataset_sha256 is not None or self.seal is not None:
                raise ValueError("legacy dataset reference cannot declare partial Phase 3 identity")
            return self
        if self.expected_dataset_sha256 is None:
            raise ValueError("typed dataset reference requires expected_dataset_sha256")
        if self.purpose is DatasetPurpose.EVALUATION:
            if self.seal is None:
                raise ValueError("evaluation dataset reference requires a canonical seal")
            if (
                self.seal.dataset_version != self.version
                or self.seal.dataset_sha256 != self.expected_dataset_sha256
            ):
                raise ValueError("evaluation dataset reference disagrees with its seal")
        elif self.seal is not None:
            raise ValueError("only evaluation dataset references may include a seal")
        return self


class DevelopmentSelectionInput(_StrictFrozenModel):
    """Dataset boundary accepted by development-only baseline selection."""

    dataset: DatasetReference

    @model_validator(mode="after")
    def _require_development_purpose(self) -> Self:
        if self.dataset.purpose is not DatasetPurpose.DEVELOPMENT:
            raise ValueError("static selection input must have development purpose")
        return self


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
    """Complete scientific configuration before deterministic dataset expansion."""

    schema_version: Literal[1]
    experiment_id: str = Field(min_length=1)
    dataset: DatasetReference
    provider: ProviderSelection
    policy_ids: tuple[str, ...] | None = None
    policy_specs: tuple[PolicySpec, ...] | None = None
    evaluation: EvaluationSpec | None = None
    development_selection_input: DevelopmentSelectionInput | None = None
    static_selection_record: StaticSelectionRecord | None = None
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

    @field_validator(
        "policy_ids",
        "policy_specs",
        "model_seeds",
        "controller_seeds",
        mode="before",
    )
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
    def _validate_policy_ids(cls, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("policy_ids must not be empty")
        if any(not value.strip() for value in values):
            raise ValueError("policy_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("policy_ids must not contain duplicates")
        return tuple(sorted(values))

    @field_validator("policy_specs")
    @classmethod
    def _validate_policy_specs(
        cls,
        values: tuple[PolicySpec, ...] | None,
    ) -> tuple[PolicySpec, ...] | None:
        if values is None:
            return None
        if not values:
            raise ValueError("policy_specs must not be empty")
        policy_ids = tuple(spec.policy_id for spec in values)
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy_specs must not contain duplicate policy identifiers")
        return tuple(sorted(values, key=lambda spec: spec.policy_id))

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
        if self.policy_ids is None and self.policy_specs is None:
            raise ValueError("either legacy policy_ids or typed policy_specs are required")
        if self.policy_ids is not None and self.policy_specs is not None:
            raise ValueError("policy_ids and policy_specs are mutually exclusive")
        if self.static_selection_record is not None:
            if self.development_selection_input is None:
                raise ValueError("static selection record requires its declared development input")
            development_hash = self.development_selection_input.dataset.expected_dataset_sha256
            if development_hash != self.static_selection_record.development_dataset_sha256:
                raise ValueError(
                    "static selection record does not match the declared development input"
                )
        if self.evaluation is None:
            if self.dataset.purpose in {
                DatasetPurpose.EVALUATION,
                DatasetPurpose.SYNTHETIC,
            }:
                raise ValueError("evaluation and synthetic datasets require an EvaluationSpec")
            return self
        if self.policy_specs is None:
            raise ValueError("EvaluationSpec requires typed policy_specs")
        if self.dataset.purpose not in {
            DatasetPurpose.EVALUATION,
            DatasetPurpose.SYNTHETIC,
        }:
            raise ValueError("EvaluationSpec requires an evaluation- or synthetic-purpose dataset")
        declared_policy_ids = set(self.configured_policy_ids)
        evaluated_policy_ids = {
            self.evaluation.focal_policy_id,
            *self.evaluation.required_serious_comparator_ids,
            *self.evaluation.negative_control_policy_ids,
        }
        if declared_policy_ids != evaluated_policy_ids:
            raise ValueError(
                "EvaluationSpec policy roles must exactly cover configured policy_specs"
            )
        if "best_static" not in declared_policy_ids:
            raise ValueError("Phase 3 evaluation requires the best_static comparator")
        if self.development_selection_input is None or self.static_selection_record is None:
            raise ValueError("Phase 3 evaluation requires frozen development selection evidence")
        winner = self.static_selection_record.winning_profile
        if self.base_decoding_profile_id != winner.profile_id or self.base_decoding_profile != (
            BaseDecodingProfile(
                temperature=winner.temperature,
                top_p=winner.top_p,
                top_k=winner.top_k,
                presence_penalty=winner.presence_penalty,
                max_tokens=winner.max_tokens,
            )
        ):
            raise ValueError("best_static winner must be the shared frozen base profile")
        return self

    @property
    def configured_policy_ids(self) -> tuple[str, ...]:
        """Return the canonical policy IDs for legacy or typed configuration."""

        if self.policy_specs is not None:
            return tuple(spec.policy_id for spec in self.policy_specs)
        assert self.policy_ids is not None
        return self.policy_ids

    @property
    def evaluation_spec_sha256(self) -> str | None:
        """Return the complete EvaluationSpec identity when Phase 3 is configured."""

        return None if self.evaluation is None else canonical_sha256(self.evaluation)

    @property
    def experiment_config_hash(self) -> str:
        """Hash scientific configuration while excluding machine-local paths."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset.version,
            "provider_kind": self.provider.kind,
            "provider_identity": self.provider.expected_identity,
            "provider_effective_configuration_json": (
                self.provider.expected_effective_configuration_json
            ),
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
        if self.policy_specs is None:
            payload["policy_ids"] = self.policy_ids
        else:
            payload["policy_specs"] = self.policy_specs
        if self.dataset.purpose is not None:
            payload.update(
                {
                    "dataset_purpose": self.dataset.purpose,
                    "expected_dataset_sha256": self.dataset.expected_dataset_sha256,
                    "dataset_seal": self.dataset.seal,
                }
            )
        if self.evaluation is not None:
            payload["evaluation"] = self.evaluation
            payload["evaluation_spec_sha256"] = self.evaluation_spec_sha256
            payload["development_selection_input"] = self.development_selection_input
            payload["static_selection_record"] = self.static_selection_record
        return canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class LoadedExperimentConfig:
    """Validated config plus explicit resolved incidental paths."""

    config: ExperimentConfig
    source_path: Path
    dataset_path: Path
    provider_config_path: Path | None
    artifact_root: Path
    development_selection_dataset_path: Path | None = None


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
    development_selection_path = (
        None
        if config.development_selection_input is None
        else _resolve_reference(base, config.development_selection_input.dataset.path)
    )
    return LoadedExperimentConfig(
        config=config,
        source_path=source_path,
        dataset_path=dataset_path,
        provider_config_path=provider_path,
        artifact_root=artifact_root,
        development_selection_dataset_path=development_selection_path,
    )


__all__ = [
    "BaseDecodingProfile",
    "DatasetReference",
    "DevelopmentSelectionInput",
    "ExperimentConfig",
    "LoadedExperimentConfig",
    "ProviderSelection",
    "load_experiment_config",
]

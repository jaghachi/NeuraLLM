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

from neurallm.control.specs import (
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    PolicySpec,
)
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    DecodingParameters,
    ProviderIdentity,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.confirmatory import ConfirmatoryAnalysisSpec
from neurallm.evaluation.models import DatasetPurpose, EvaluationSpec
from neurallm.evaluation.pilot_grid import DevelopmentPilotCandidateGrid
from neurallm.evaluation.pilot_selection import DevelopmentPilotStaticSelectionEvidence
from neurallm.evaluation.selection import StaticSelectionRecord
from neurallm.experiments.dataset import DatasetSeal
from neurallm.experiments.protocol import (
    ExperimentProtocol,
    PreregistrationSeal,
    RunTier,
)
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


class StaticSelectionEvidenceReference(_StrictFrozenModel):
    """Incidental path plus expected identity for an external pilot artifact."""

    path: str = Field(min_length=1)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateGridReference(_StrictFrozenModel):
    """Incidental path plus expected identity for the predeclared pilot grid."""

    path: str = Field(min_length=1)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    protocol: ExperimentProtocol | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    preregistration: PreregistrationSeal | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    confirmatory_analysis: ConfirmatoryAnalysisSpec | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    evaluation: EvaluationSpec | None = None
    development_selection_input: DevelopmentSelectionInput | None = None
    candidate_grid: DevelopmentPilotCandidateGrid | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    static_selection_record: StaticSelectionRecord | None = None
    static_selection_evidence: DevelopmentPilotStaticSelectionEvidence | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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
        base = self.base_decoding_profile
        bounds = self.decoding_bounds
        if not (
            bounds.temperature[0] <= base.temperature <= bounds.temperature[1]
            and bounds.top_p[0] <= base.top_p <= bounds.top_p[1]
            and bounds.top_k[0] <= base.top_k <= bounds.top_k[1]
            and bounds.presence_penalty[0] <= base.presence_penalty <= bounds.presence_penalty[1]
        ):
            raise ValueError("base decoding profile exceeds the configured legal bounds")
        if self.policy_ids is None and self.policy_specs is None:
            raise ValueError("either legacy policy_ids or typed policy_specs are required")
        if self.policy_ids is not None and self.policy_specs is not None:
            raise ValueError("policy_ids and policy_specs are mutually exclusive")
        matched_history_specs = tuple(
            spec
            for spec in (self.policy_specs or ())
            if spec.history_access == "matched_focal_previous_response"
        )
        configured_policy_ids = set(self.configured_policy_ids)
        neural_specs = tuple(
            spec
            for spec in (self.policy_specs or ())
            if isinstance(
                spec,
                (NeuralPersistentPolicySpec, NeuralMatchedHistoryStateResetPolicySpec),
            )
        )
        for spec in matched_history_specs:
            source_policy_id = getattr(spec, "history_source_policy_id", None)
            if source_policy_id not in configured_policy_ids:
                raise ValueError("matched-history policy requires its declared focal source policy")
        if self.static_selection_record is not None and self.static_selection_evidence is not None:
            raise ValueError(
                "inline static selection and model-backed selection evidence are mutually exclusive"
            )
        resolved_selection = self.resolved_static_selection_record
        if resolved_selection is not None:
            if self.development_selection_input is None:
                raise ValueError("static selection record requires its declared development input")
            development_hash = self.development_selection_input.dataset.expected_dataset_sha256
            if development_hash != resolved_selection.development_dataset_sha256:
                raise ValueError(
                    "static selection record does not match the declared development input"
                )
        if self.static_selection_evidence is not None:
            assert self.development_selection_input is not None
            source_versions = {
                turn.condition.dataset_version
                for candidate in self.static_selection_evidence.candidates
                for turn in candidate.turns
            }
            if source_versions != {self.development_selection_input.dataset.version}:
                raise ValueError(
                    "static selection evidence does not match the development dataset version"
                )
        if self.protocol is not None:
            if self.policy_specs is None:
                raise ValueError("model-backed protocol requires typed policy_specs")
            if self.configured_policy_ids != self.protocol.policy_ids:
                raise ValueError("model-backed protocol requires its exact five policy specs")
            if len(matched_history_specs) != 1 or (
                matched_history_specs[0].policy_id != self.protocol.attribution.policy_id
                or getattr(matched_history_specs[0], "history_source_policy_id", None)
                != self.protocol.attribution.history_source_policy_id
            ):
                raise ValueError("model-backed protocol requires its exact attribution edge")
            if len(neural_specs) != 2:
                raise ValueError("model-backed protocol requires both declared neural policies")
            schedule = self.protocol.schedule
            if schedule.model_seed_count != len(self.model_seeds):
                raise ValueError("protocol model_seed_count does not match model_seeds")
            if schedule.controller_seed_count != len(self.controller_seeds):
                raise ValueError("protocol controller_seed_count does not match controller_seeds")
            if schedule.policy_count != len(self.configured_policy_ids):
                raise ValueError("protocol policy_count does not match configured policies")
            if self.decision_rule_version != self.protocol.decision_rule_version:
                raise ValueError("decision_rule_version does not match the model-backed tier")

            tier = self.protocol.run_tier
            if tier in {RunTier.ENGINEERING_SMOKE, RunTier.DEVELOPMENT_PILOT}:
                if self.dataset.purpose is not DatasetPurpose.DEVELOPMENT:
                    raise ValueError("smoke and pilot protocols require development-purpose data")
                if self.preregistration is not None:
                    raise ValueError("only confirmatory protocols may carry preregistration")
                if self.evaluation is not None:
                    raise ValueError(
                        "smoke and pilot protocols cannot produce confirmatory evaluation"
                    )
                if self.confirmatory_analysis is not None:
                    raise ValueError(
                        "smoke and pilot protocols cannot carry confirmatory analysis rules"
                    )
                if self.static_selection_evidence is not None:
                    raise ValueError("smoke and pilot protocols cannot consume selection evidence")
                if tier is RunTier.ENGINEERING_SMOKE:
                    if self.candidate_grid is not None:
                        raise ValueError("engineering smoke cannot carry a candidate grid")
                    return self
                if self.candidate_grid is None:
                    raise ValueError("development pilot requires a predeclared candidate grid")
                if (
                    self.candidate_grid.dataset_purpose is not self.dataset.purpose
                    or self.candidate_grid.dataset_version != self.dataset.version
                    or self.candidate_grid.dataset_sha256 != self.dataset.expected_dataset_sha256
                ):
                    raise ValueError(
                        "development pilot candidate grid differs from its dataset identity"
                    )
                declared_profiles = {
                    profile.profile_id: profile
                    for profile in self.candidate_grid.candidate_profiles
                }
                selected_profile = declared_profiles.get(self.base_decoding_profile_id)
                if selected_profile is None or self.base_decoding_profile != BaseDecodingProfile(
                    temperature=selected_profile.temperature,
                    top_p=selected_profile.top_p,
                    top_k=selected_profile.top_k,
                    presence_penalty=selected_profile.presence_penalty,
                    max_tokens=selected_profile.max_tokens,
                ):
                    raise ValueError(
                        "development pilot base profile must equal one declared grid profile"
                    )
                return self

            if self.candidate_grid is not None:
                raise ValueError("candidate grid is accepted only by development pilots")
            if self.dataset.purpose is not DatasetPurpose.EVALUATION:
                raise ValueError("confirmatory protocol requires sealed evaluation-purpose data")
            if self.provider.kind != "llama_cpp":
                raise ValueError("confirmatory protocol requires the explicit llama_cpp provider")
            if self.provider.expected_identity.model_sha256 is None:
                raise ValueError("confirmatory protocol requires a model-artifact SHA-256 identity")
            if self.evaluation is None:
                raise ValueError("confirmatory protocol requires an EvaluationSpec")
            if self.confirmatory_analysis is None:
                raise ValueError("confirmatory protocol requires frozen final analysis rules")
            if self.preregistration is not None and (
                self.preregistration.experiment_id != self.experiment_id
                or self.preregistration.run_tier != tier.value
            ):
                raise ValueError("preregistration seal does not match the confirmatory protocol")
            if (
                self.evaluation.focal_policy_id != self.protocol.focal_policy_id
                or self.evaluation.required_serious_comparator_ids
                != self.protocol.required_serious_comparator_ids
                or self.evaluation.negative_control_policy_ids
                != self.protocol.negative_control_policy_ids
            ):
                raise ValueError("EvaluationSpec roles do not match model-backed efficacy roles")
            efficacy = self.confirmatory_analysis.efficacy
            if (
                efficacy.focal_policy_id != self.evaluation.focal_policy_id
                or efficacy.serious_comparator_ids
                != self.evaluation.required_serious_comparator_ids
                or efficacy.negative_control_policy_id
                != self.evaluation.negative_control_policy_ids[0]
                or efficacy.practical_effect_threshold != self.evaluation.practical_effect_threshold
                or efficacy.bootstrap_resamples != self.evaluation.bootstrap_resamples
                or efficacy.confidence_level != self.evaluation.confidence_level
                or efficacy.bootstrap_seed != self.evaluation.bootstrap_seed
                or efficacy.permutation_resamples != self.evaluation.permutation_resamples
                or efficacy.permutation_seed != self.evaluation.permutation_seed
            ):
                raise ValueError(
                    "confirmatory efficacy rules disagree with the frozen EvaluationSpec"
                )
            if (
                self.confirmatory_analysis.recovery.serious_comparator_ids
                != self.evaluation.required_serious_comparator_ids
            ):
                raise ValueError(
                    "confirmatory recovery roles disagree with the frozen EvaluationSpec"
                )
            if (
                self.development_selection_input is None
                or self.static_selection_evidence is None
                or self.static_selection_record is not None
            ):
                raise ValueError(
                    "confirmatory protocol requires external model-backed selection evidence"
                )
            winner = self.static_selection_evidence.selection_record.winning_profile
            if self.base_decoding_profile_id != winner.profile_id or (
                self.base_decoding_profile
                != BaseDecodingProfile(
                    temperature=winner.temperature,
                    top_p=winner.top_p,
                    top_k=winner.top_k,
                    presence_penalty=winner.presence_penalty,
                    max_tokens=winner.max_tokens,
                )
            ):
                raise ValueError("best_static winner must be the shared frozen base profile")
            pilot_manifest = self.static_selection_evidence.candidates[0].source_run_manifest
            if (
                self.provider.expected_identity != pilot_manifest.provider_identity
                or self.provider.expected_effective_configuration_json
                != pilot_manifest.provider_effective_configuration_json
            ):
                raise ValueError(
                    "confirmatory provider binding differs from development-pilot evidence"
                )
            assert self.policy_specs is not None
            policy_config_hashes = {
                spec.policy_id: canonical_sha256(spec) for spec in self.policy_specs
            }
            if dict(pilot_manifest.policy_config_hashes) != policy_config_hashes:
                raise ValueError(
                    "confirmatory policy specifications differ from development-pilot evidence"
                )
            if (
                self.action_bounds != pilot_manifest.action_bounds
                or self.decoding_bounds != pilot_manifest.decoding_bounds
            ):
                raise ValueError(
                    "confirmatory action or decoding bounds differ from development-pilot evidence"
                )
            if dict(self.metric_versions) != dict(pilot_manifest.metric_versions):
                raise ValueError(
                    "confirmatory metric versions differ from development-pilot evidence"
                )
            if self.database_schema_version != pilot_manifest.database_schema_version:
                raise ValueError(
                    "confirmatory database schema differs from development-pilot evidence"
                )
            return self

        if self.preregistration is not None:
            raise ValueError("preregistration requires a model-backed protocol")
        if self.confirmatory_analysis is not None:
            raise ValueError("confirmatory analysis rules require a model-backed protocol")
        if self.static_selection_evidence is not None:
            raise ValueError("model-backed selection evidence is accepted only by confirmatory")
        if self.candidate_grid is not None:
            raise ValueError("candidate grid requires a development-pilot protocol")
        if neural_specs and self.evaluation is not None:
            raise ValueError("neural policies are not admitted to Phase 3 efficacy evaluation")
        if matched_history_specs:
            phase4_policy_ids = {
                "neural_persistent",
                "neural_matched_history_state_reset",
            }
            if configured_policy_ids != phase4_policy_ids:
                raise ValueError("Phase 4 requires exactly the two neural attribution policies")
            if self.dataset.purpose is not DatasetPurpose.DEVELOPMENT:
                raise ValueError("Phase 4 requires a pinned development-purpose dataset")
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
    def resolved_static_selection_record(self) -> StaticSelectionRecord | None:
        """Return legacy inline or externally proven static-selection evidence."""

        if self.static_selection_evidence is not None:
            return self.static_selection_evidence.selection_record
        return self.static_selection_record

    @property
    def static_selection_evidence_sha256(self) -> str | None:
        """Return the external pilot evidence identity when one is configured."""

        if self.static_selection_evidence is None:
            return None
        return self.static_selection_evidence.evidence_sha256

    @property
    def candidate_grid_sha256(self) -> str | None:
        """Return the pre-execution development-pilot grid identity."""

        if self.candidate_grid is None:
            return None
        return self.candidate_grid.candidate_grid_sha256

    @property
    def experiment_config_hash(self) -> str:
        """Hash scientific configuration excluding incidental reference paths."""

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
            if self.static_selection_evidence is None:
                payload["static_selection_record"] = self.static_selection_record
            else:
                payload["static_selection_evidence"] = self.static_selection_evidence
        if self.protocol is not None:
            payload["protocol"] = self.protocol
        if self.candidate_grid is not None:
            payload["candidate_grid"] = self.candidate_grid
            payload["candidate_grid_sha256"] = self.candidate_grid_sha256
        if self.confirmatory_analysis is not None:
            payload["confirmatory_analysis"] = self.confirmatory_analysis
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
    candidate_grid_path: Path | None = None
    static_selection_evidence_path: Path | None = None


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
    source_payload = load_yaml_mapping(source_path)
    base = source_path.parent
    candidate_grid_path: Path | None = None
    candidate_grid_payload = source_payload.get("candidate_grid")
    if candidate_grid_payload is not None:
        grid_reference = CandidateGridReference.model_validate(candidate_grid_payload)
        candidate_grid_path = _resolve_reference(base, grid_reference.path)
        from neurallm.evaluation.pilot_grid import load_development_pilot_candidate_grid

        candidate_grid = load_development_pilot_candidate_grid(
            candidate_grid_path,
            expected_sha256=grid_reference.expected_sha256,
        )
        source_payload["candidate_grid"] = candidate_grid.model_dump(mode="python")
    evidence_path: Path | None = None
    evidence_payload = source_payload.get("static_selection_evidence")
    if evidence_payload is not None:
        evidence_reference = StaticSelectionEvidenceReference.model_validate(evidence_payload)
        evidence_path = _resolve_reference(base, evidence_reference.path)
        from neurallm.experiments.static_selection import load_static_selection_evidence

        evidence = load_static_selection_evidence(evidence_path)
        if evidence.evidence_sha256 != evidence_reference.expected_sha256:
            raise ValueError("static selection evidence differs from its expected SHA-256")
        source_payload["static_selection_evidence"] = evidence.model_dump(mode="python")
    config = ExperimentConfig.model_validate(source_payload)
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
        candidate_grid_path=candidate_grid_path,
        static_selection_evidence_path=evidence_path,
    )


__all__ = [
    "BaseDecodingProfile",
    "CandidateGridReference",
    "ConfirmatoryAnalysisSpec",
    "DatasetReference",
    "DevelopmentSelectionInput",
    "ExperimentConfig",
    "LoadedExperimentConfig",
    "ExperimentProtocol",
    "PreregistrationSeal",
    "ProviderSelection",
    "RunTier",
    "StaticSelectionEvidenceReference",
    "load_experiment_config",
]

"""Deterministic expansion of validated scientific inputs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Self

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
    ExperimentCondition,
    PromptFeatures,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import EvaluationSpec
from neurallm.evaluation.selection import StaticSelectionRecord
from neurallm.experiments.config import LoadedExperimentConfig
from neurallm.experiments.dataset import (
    DatasetPurpose,
    DatasetSeal,
    LoadedDataset,
    PromptCase,
    PromptSequence,
    validate_dataset_identity,
)
from neurallm.experiments.matching import (
    MatchedCoverageExpectation,
    materialize_matched_coverage,
)
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.base import GenerationRequest
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION

PHASE2_DECISION_RULE_VERSION = "phase2-no-scientific-decision-v1"
PHASE3_DECISION_RULE_VERSION = "phase3-baseline-evaluator-v1"
PHASE4_DECISION_RULE_VERSION = "phase4-neural-mechanism-only-v1"


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
    dataset_purpose: DatasetPurpose | None = None
    experiment_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_seal: DatasetSeal | None = None
    provider_identity: ProviderIdentity
    provider_effective_configuration_json: str = Field(min_length=2)
    action_bounds: ActionBounds
    decoding_bounds: DecodingBounds
    metric_versions: Mapping[str, str]
    decision_rule_version: str = Field(min_length=1)
    database_schema_version: int = Field(gt=0)
    evaluation: EvaluationSpec | None = None
    evaluation_spec_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    static_selection_record: StaticSelectionRecord | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    static_selection_result_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    matched_units: tuple[MatchedCoverageExpectation, ...] = ()
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

    @model_validator(mode="after")
    def _validate_evaluation_identity_and_coverage(self) -> Self:
        if self.evaluation is None:
            if (
                self.evaluation_spec_sha256 is not None
                or self.static_selection_record is not None
                or self.static_selection_result_sha256 is not None
                or self.matched_units
            ):
                raise ValueError("Phase 2 plan cannot contain Phase 3 evaluation evidence")
            if self.decision_rule_version == PHASE4_DECISION_RULE_VERSION:
                policy_ids = {turn.condition.policy_id for turn in self.turns}
                if policy_ids != {
                    "neural_persistent",
                    "neural_matched_history_state_reset",
                }:
                    raise ValueError(
                        "Phase 4 plan requires exactly the two neural attribution policies"
                    )
                if self.dataset_purpose is not DatasetPurpose.DEVELOPMENT:
                    raise ValueError("Phase 4 plan requires a pinned development-purpose dataset")
                persistent_coordinates = {
                    (
                        turn.condition.experiment_id,
                        turn.condition.dataset_version,
                        turn.condition.prompt_sequence_id,
                        turn.condition.model_seed,
                        turn.condition.controller_seed,
                        turn.condition.provider_identity_id,
                        turn.condition.base_decoding_profile_id,
                        turn.condition.turn_index,
                    )
                    for turn in self.turns
                    if turn.condition.policy_id == "neural_persistent"
                }
                reset_coordinates = {
                    (
                        turn.condition.experiment_id,
                        turn.condition.dataset_version,
                        turn.condition.prompt_sequence_id,
                        turn.condition.model_seed,
                        turn.condition.controller_seed,
                        turn.condition.provider_identity_id,
                        turn.condition.base_decoding_profile_id,
                        turn.condition.turn_index,
                    )
                    for turn in self.turns
                    if turn.condition.policy_id == "neural_matched_history_state_reset"
                }
                if persistent_coordinates != reset_coordinates:
                    raise ValueError("Phase 4 plan requires exact paired turn coverage")
            return self
        if self.evaluation_spec_sha256 != canonical_sha256(self.evaluation):
            raise ValueError("evaluation_spec_sha256 does not match EvaluationSpec")
        if self.static_selection_record is None or self.static_selection_result_sha256 is None:
            raise ValueError("Phase 3 plan requires frozen static-selection evidence")
        if (
            self.static_selection_result_sha256
            != self.static_selection_record.selection_result_sha256
        ):
            raise ValueError("static selection hash does not match its canonical evidence")
        if not self.matched_units:
            raise ValueError("Phase 3 plan requires matched coverage expectations")
        expected_ids = tuple(
            sorted(
                condition_id
                for unit in self.matched_units
                for condition_id in unit.expected_condition_ids
            )
        )
        actual_ids = tuple(sorted(turn.condition.condition_id for turn in self.turns))
        if expected_ids != actual_ids:
            raise ValueError("matched coverage expectations do not exactly cover plan turns")
        unit_keys = tuple(unit.unit_key for unit in self.matched_units)
        if len(unit_keys) != len(set(unit_keys)):
            raise ValueError("matched coverage expectations contain duplicate analysis units")
        return self

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
    validate_dataset_identity(
        dataset,
        expected_version=config.dataset.version,
        expected_purpose=config.dataset.purpose,
        expected_sha256=config.dataset.expected_dataset_sha256,
        seal=config.dataset.seal,
    )
    if dict(config.metric_versions) != METRIC_VERSIONS:
        raise ValueError("configured metric versions do not match the implementation")
    if not 1 <= config.database_schema_version <= CURRENT_SCHEMA_VERSION:
        raise ValueError("configured database schema version is not supported")
    has_matched_history = any(
        spec.history_access == "matched_focal_previous_response"
        for spec in (config.policy_specs or ())
    )
    if has_matched_history and config.dataset.purpose is not DatasetPurpose.DEVELOPMENT:
        raise ValueError("Phase 4 requires a pinned development-purpose dataset")
    if config.evaluation is None:
        expected_rule = (
            PHASE4_DECISION_RULE_VERSION if has_matched_history else PHASE2_DECISION_RULE_VERSION
        )
        if config.decision_rule_version != expected_rule:
            phase = "Phase 4" if has_matched_history else "Phase 2"
            raise ValueError(f"configured decision rule version does not match {phase}")
        if has_matched_history and config.database_schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError("Phase 4 requires the current database schema version")
    else:
        if config.database_schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError("Phase 3 requires the current database schema version")
        if config.decision_rule_version != PHASE3_DECISION_RULE_VERSION:
            raise ValueError("configured decision rule version does not match Phase 3")

    turns: list[PlannedTurn] = []
    provider_identity_id = config.provider.expected_identity.identity_id

    def append_turn(
        sequence: PromptSequence,
        policy_id: str,
        model_seed: int,
        controller_seed: int,
        turn_index: int,
        prompt_case: PromptCase,
    ) -> None:
        parameters = config.base_decoding_profile.with_seed(model_seed)
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

    policy_specs = config.policy_specs or ()
    matched_source_ids = {
        spec.policy_id
        for spec in policy_specs
        if spec.history_access == "matched_focal_previous_response"
    }
    sequences = sorted(dataset.sequences, key=lambda item: item.sequence_id)
    if matched_source_ids:
        policy_order = tuple(
            sorted(
                config.configured_policy_ids,
                key=lambda policy_id: (policy_id in matched_source_ids, policy_id),
            )
        )
        for sequence in sequences:
            for model_seed in sorted(config.model_seeds):
                for controller_seed in sorted(config.controller_seeds):
                    for turn_index, prompt_case in enumerate(sequence.cases):
                        for policy_id in policy_order:
                            append_turn(
                                sequence,
                                policy_id,
                                model_seed,
                                controller_seed,
                                turn_index,
                                prompt_case,
                            )
    else:
        for sequence in sequences:
            for policy_id in config.configured_policy_ids:
                for model_seed in sorted(config.model_seeds):
                    for controller_seed in sorted(config.controller_seeds):
                        for turn_index, prompt_case in enumerate(sequence.cases):
                            append_turn(
                                sequence,
                                policy_id,
                                model_seed,
                                controller_seed,
                                turn_index,
                                prompt_case,
                            )

    matched_units = (
        ()
        if config.evaluation is None
        else materialize_matched_coverage(
            tuple(turn.condition for turn in turns),
            experiment_id=config.experiment_id,
            dataset_version=dataset.version,
            sequence_turn_indexes={
                sequence.sequence_id: tuple(range(len(sequence.cases)))
                for sequence in dataset.sequences
            },
            policy_ids=config.configured_policy_ids,
            model_seeds=config.model_seeds,
            controller_seeds=config.controller_seeds,
        )
    )

    return ExperimentPlan(
        experiment_id=config.experiment_id,
        dataset_version=dataset.version,
        dataset_purpose=dataset.purpose,
        experiment_config_hash=config.experiment_config_hash,
        dataset_hash=dataset.dataset_hash,
        dataset_seal=config.dataset.seal,
        provider_identity=config.provider.expected_identity,
        provider_effective_configuration_json=(
            config.provider.expected_effective_configuration_json
        ),
        action_bounds=config.action_bounds,
        decoding_bounds=config.decoding_bounds,
        metric_versions=config.metric_versions,
        decision_rule_version=config.decision_rule_version,
        database_schema_version=config.database_schema_version,
        evaluation=config.evaluation,
        evaluation_spec_sha256=config.evaluation_spec_sha256,
        static_selection_record=config.static_selection_record,
        static_selection_result_sha256=(
            None
            if config.static_selection_record is None
            else config.static_selection_record.selection_result_sha256
        ),
        matched_units=matched_units,
        turns=tuple(turns),
    )


__all__ = [
    "PHASE2_DECISION_RULE_VERSION",
    "PHASE3_DECISION_RULE_VERSION",
    "PHASE4_DECISION_RULE_VERSION",
    "ExperimentPlan",
    "PlannedTurn",
    "build_plan",
]

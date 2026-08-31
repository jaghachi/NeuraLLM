"""Typed records and checkpoint states for transactional turn storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import (
    ExperimentCondition,
    NonEmptyString,
    PromptFeatures,
    ResponseMetrics,
    Sha256Hex,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    GuardrailResult,
    PairwiseComparisonResult,
    Phase3EvaluationResult,
)
from neurallm.evaluation.selection import StaticSelectionRecord
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.base import GenerationRequest, GenerationResponse


class TurnState(StrEnum):
    """Forward-only durable checkpoints for one logical generation."""

    PREPARED = "PREPARED"
    DISPATCHING = "DISPATCHING"
    RESPONSE_PERSISTED = "RESPONSE_PERSISTED"
    METRICS_COMPUTED = "METRICS_COMPUTED"
    COMMITTED = "COMMITTED"
    UNCERTAIN_DISPATCH = "UNCERTAIN_DISPATCH"


class ResumeAction(StrEnum):
    """Only the safe next operations exposed by resumption."""

    DISPATCH_PREPARED = "DISPATCH_PREPARED"
    COMPUTE_METRICS = "COMPUTE_METRICS"
    COMMIT = "COMMIT"
    SKIP_COMMITTED = "SKIP_COMMITTED"


class HistoryBinding(BaseModel):
    """Exact committed predecessor selected by the experiment runner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    previous_condition_id: Sha256Hex
    previous_history_commitment_sha256: Sha256Hex


class TurnInputEvidence(BaseModel):
    """Complete prompt-side evidence needed to reconstruct deterministic metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    condition_id: Sha256Hex
    prompt_case_id: NonEmptyString
    prompt_family: NonEmptyString
    prompt_features: PromptFeatures
    validator: ValidatorSpec


class RunFinalization(BaseModel):
    """Canonical, immutable evidence that one complete run schedule is closed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    expected_condition_ids: tuple[Sha256Hex, ...]
    expected_condition_count: int
    manifest_sha256: Sha256Hex
    scientific_result_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_schedule_identity(self) -> Self:
        """Require one nonempty, sorted, duplicate-free schedule identity."""

        if not self.expected_condition_ids:
            raise ValueError("finalized run schedule must not be empty")
        if self.expected_condition_ids != tuple(sorted(set(self.expected_condition_ids))):
            raise ValueError("finalized condition IDs must be sorted and unique")
        if self.expected_condition_count != len(self.expected_condition_ids):
            raise ValueError("finalized condition count must equal the condition ID count")
        return self


class AnalysisManifest(BaseModel):
    """Immutable binding between one closed run and its Phase 3 evaluator inputs."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    implementation_version: Literal["phase3-analysis-storage-v1"] = "phase3-analysis-storage-v1"
    action_magnitude_version: Literal["rms-normalized-to-action-bounds-v1"] = (
        "rms-normalized-to-action-bounds-v1"
    )
    run_manifest_sha256: Sha256Hex
    run_finalization_sha256: Sha256Hex
    scientific_result_sha256: Sha256Hex
    experiment_plan_sha256: Sha256Hex
    evaluation_spec: EvaluationSpec
    evaluation_spec_sha256: Sha256Hex
    static_selection_record: StaticSelectionRecord
    static_selection_result_sha256: Sha256Hex
    evaluation_design: ExpectedEvaluationDesign
    dataset_sha256: Sha256Hex
    dataset_purpose: DatasetPurpose
    dataset_seal_sha256: Sha256Hex | None = None
    evaluation_input_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_dataset_boundary(self) -> Self:
        """Require sealed confirmatory data and explicitly unsealed synthetic data."""

        if self.dataset_purpose is DatasetPurpose.DEVELOPMENT:
            raise ValueError("Phase 3 analysis cannot consume development data")
        if self.dataset_purpose is DatasetPurpose.EVALUATION and self.dataset_seal_sha256 is None:
            raise ValueError("evaluation analysis requires a dataset seal hash")
        if (
            self.dataset_purpose is DatasetPurpose.SYNTHETIC
            and self.dataset_seal_sha256 is not None
        ):
            raise ValueError("synthetic analysis must not claim an evaluation seal")
        if self.evaluation_spec_sha256 != canonical_sha256(self.evaluation_spec):
            raise ValueError("evaluation spec hash does not match its canonical evidence")
        if (
            self.static_selection_result_sha256
            != self.static_selection_record.selection_result_sha256
        ):
            raise ValueError("static selection hash does not match its canonical evidence")
        if (
            self.evaluation_design.dataset_purpose is not self.dataset_purpose
            or self.evaluation_design.dataset_sha256 != self.dataset_sha256
            or self.evaluation_design.dataset_seal_sha256 != self.dataset_seal_sha256
        ):
            raise ValueError("evaluation design disagrees with the analysis dataset identity")
        return self


class AnalysisFinalization(BaseModel):
    """Canonical closure over a complete persisted Phase 3 evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    implementation_version: Literal["phase3-analysis-finalization-v1"] = (
        "phase3-analysis-finalization-v1"
    )
    analysis_manifest_sha256: Sha256Hex
    evaluation_result_sha256: Sha256Hex
    decision_sha256: Sha256Hex
    comparison_result_sha256s: tuple[Sha256Hex, ...]
    guardrail_result_sha256s: tuple[Sha256Hex, ...]
    comparison_count: int = Field(ge=0)
    guardrail_count: int = Field(ge=1)

    @field_validator("comparison_result_sha256s", "guardrail_result_sha256s")
    @classmethod
    def validate_sorted_unique_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("analysis evidence hashes must be sorted and unique")
        return values

    @model_validator(mode="after")
    def validate_evidence_counts(self) -> Self:
        if self.comparison_count != len(self.comparison_result_sha256s):
            raise ValueError("comparison count must equal the persisted comparison hashes")
        if self.guardrail_count != len(self.guardrail_result_sha256s):
            raise ValueError("guardrail count must equal the persisted guardrail hashes")
        return self


@dataclass(frozen=True, slots=True)
class StoredTurn:
    """Validated materialized view of one turn in the canonical store."""

    condition_id: str
    request_sha256: str
    state: TurnState
    condition: ExperimentCondition
    request: GenerationRequest
    history: HistoryBinding | None
    response: GenerationResponse | None
    metrics: ResponseMetrics | None
    policy_state_json: str | None
    policy_trace_json: str | None
    history_commitment_sha256: str | None
    uncertain_reason: str | None


@dataclass(frozen=True, slots=True)
class CommittedHistory:
    """Validated response history and policy state for a future turn."""

    condition_id: str
    condition: ExperimentCondition
    request_sha256: str
    response_sha256: str
    metrics: ResponseMetrics
    policy_state_json: str
    policy_trace_json: str
    previous_history_commitment_sha256: str | None
    history_commitment_sha256: str


@dataclass(frozen=True, slots=True)
class StoredAnalysis:
    """Hash-validated materialized view of one finalized Phase 3 analysis."""

    manifest: AnalysisManifest
    result: Phase3EvaluationResult
    comparisons: tuple[PairwiseComparisonResult, ...]
    guardrails: tuple[GuardrailResult, ...]
    finalization: AnalysisFinalization


__all__ = [
    "AnalysisFinalization",
    "AnalysisManifest",
    "CommittedHistory",
    "HistoryBinding",
    "ResumeAction",
    "RunFinalization",
    "StoredTurn",
    "StoredAnalysis",
    "TurnInputEvidence",
    "TurnState",
]

"""Typed records and checkpoint states for transactional turn storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from neurallm.domain.models import ExperimentCondition, ResponseMetrics, Sha256Hex
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


__all__ = [
    "CommittedHistory",
    "HistoryBinding",
    "ResumeAction",
    "RunFinalization",
    "StoredTurn",
    "TurnState",
]

"""Development-only selection of a frozen strong static profile."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from neurallm.domain.models import NonEmptyString, Sha256Hex, UnitInterval
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import DatasetPurpose, MatchedUnitKey, _StrictFrozenModel


class StaticProfile(_StrictFrozenModel):
    """One declared candidate decoding profile."""

    profile_id: NonEmptyString
    temperature: float = Field(gt=0.0, allow_inf_nan=False)
    top_p: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    top_k: int = Field(ge=0)
    presence_penalty: float = Field(allow_inf_nan=False)
    max_tokens: int = Field(gt=0)


class StaticCandidateResult(_StrictFrozenModel):
    """Development sequence-unit scores for one static candidate."""

    profile: StaticProfile
    unit_scores: tuple[UnitInterval, ...]

    @field_validator("unit_scores", mode="before")
    @classmethod
    def _accept_yaml_scores(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("unit_scores")
    @classmethod
    def _require_scores(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("a static candidate requires development unit scores")
        return values

    @property
    def mean_score(self) -> float:
        """Return the sequence-unit mean used by the selection rule."""

        return sum(self.unit_scores) / len(self.unit_scores)


def _selection_payload(
    development_dataset_sha256: str,
    development_unit_keys: tuple[MatchedUnitKey, ...],
    candidate_results: tuple[StaticCandidateResult, ...],
    winning_profile: StaticProfile,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation_version": "development-static-selection-v1",
        "dataset_purpose": DatasetPurpose.DEVELOPMENT,
        "development_dataset_sha256": development_dataset_sha256,
        "development_unit_keys": development_unit_keys,
        "selection_metric": "mean-sequence-task-score-v1",
        "tie_break_rule": "highest-mean-then-lexical-profile-id-v1",
        "candidate_results": candidate_results,
        "winning_profile": winning_profile,
    }


class StaticSelectionRecord(_StrictFrozenModel):
    """Canonical evidence for a development-only frozen static winner."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["development-static-selection-v1"] = (
        "development-static-selection-v1"
    )
    dataset_purpose: Literal[DatasetPurpose.DEVELOPMENT] = DatasetPurpose.DEVELOPMENT
    development_dataset_sha256: Sha256Hex
    development_unit_keys: tuple[MatchedUnitKey, ...]
    selection_metric: Literal["mean-sequence-task-score-v1"] = "mean-sequence-task-score-v1"
    tie_break_rule: Literal["highest-mean-then-lexical-profile-id-v1"] = (
        "highest-mean-then-lexical-profile-id-v1"
    )
    candidate_results: tuple[StaticCandidateResult, ...]
    winning_profile: StaticProfile
    selection_result_sha256: Sha256Hex

    @field_validator("dataset_purpose", mode="before")
    @classmethod
    def _accept_serialized_development_purpose(cls, value: object) -> object:
        if isinstance(value, str):
            return DatasetPurpose(value)
        return value

    @field_validator("candidate_results", mode="before")
    @classmethod
    def _accept_yaml_candidates(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("development_unit_keys", mode="before")
    @classmethod
    def _accept_yaml_unit_keys(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_selection_evidence(self) -> Self:
        if len(self.candidate_results) < 2:
            raise ValueError("static selection requires at least two declared candidates")
        profile_ids = tuple(result.profile.profile_id for result in self.candidate_results)
        if profile_ids != tuple(sorted(set(profile_ids))):
            raise ValueError("static candidate results must be sorted and profile-unique")
        canonical_unit_keys = tuple(
            sorted(
                set(self.development_unit_keys),
                key=lambda key: (key.prompt_sequence_id, key.model_seed),
            )
        )
        if not canonical_unit_keys or self.development_unit_keys != canonical_unit_keys:
            raise ValueError("development unit keys must be nonempty, sorted, and unique")
        unit_counts = {len(result.unit_scores) for result in self.candidate_results}
        if unit_counts != {len(self.development_unit_keys)}:
            raise ValueError(
                "every static candidate score vector must align with the development unit keys"
            )
        max_token_budgets = {result.profile.max_tokens for result in self.candidate_results}
        if len(max_token_budgets) != 1:
            raise ValueError("static candidates must use one fixed max_tokens budget")
        expected_winner = min(
            self.candidate_results,
            key=lambda result: (-result.mean_score, result.profile.profile_id),
        ).profile
        if self.winning_profile != expected_winner:
            raise ValueError("winning profile does not match the frozen selection rule")
        expected_hash = canonical_sha256(
            _selection_payload(
                self.development_dataset_sha256,
                self.development_unit_keys,
                self.candidate_results,
                self.winning_profile,
            )
        )
        if self.selection_result_sha256 != expected_hash:
            raise ValueError("selection result hash does not match its canonical evidence")
        return self


def select_best_static(
    candidate_results: tuple[StaticCandidateResult, ...],
    *,
    dataset_purpose: DatasetPurpose,
    dataset_sha256: str,
    development_unit_keys: tuple[MatchedUnitKey, ...],
) -> StaticSelectionRecord:
    """Select deterministically on development evidence and reject evaluation leakage."""

    if dataset_purpose is not DatasetPurpose.DEVELOPMENT:
        raise ValueError("static profile selection accepts development data only")
    ordered = tuple(sorted(candidate_results, key=lambda result: result.profile.profile_id))
    if len(ordered) < 2:
        raise ValueError("static selection requires at least two declared candidates")
    winner = min(
        ordered, key=lambda result: (-result.mean_score, result.profile.profile_id)
    ).profile
    digest = canonical_sha256(
        _selection_payload(dataset_sha256, development_unit_keys, ordered, winner)
    )
    return StaticSelectionRecord(
        development_dataset_sha256=dataset_sha256,
        development_unit_keys=development_unit_keys,
        candidate_results=ordered,
        winning_profile=winner,
        selection_result_sha256=digest,
    )


__all__ = [
    "StaticCandidateResult",
    "StaticProfile",
    "StaticSelectionRecord",
    "select_best_static",
]

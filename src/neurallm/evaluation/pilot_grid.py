"""Predeclared candidate grid for model-backed development pilots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import Sha256Hex
from neurallm.domain.serialization import canonical_json_bytes, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.evaluation.selection import StaticProfile


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


MODEL_BACKED_STATIC_CANDIDATE_PROFILES = (
    StaticProfile(
        profile_id="static-balanced-v1",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=192,
    ),
    StaticProfile(
        profile_id="static-conservative-v1",
        temperature=0.55,
        top_p=0.85,
        top_k=30,
        presence_penalty=0.0,
        max_tokens=192,
    ),
    StaticProfile(
        profile_id="static-exploratory-v1",
        temperature=0.85,
        top_p=0.95,
        top_k=60,
        presence_penalty=0.1,
        max_tokens=192,
    ),
)


class DevelopmentPilotCandidateGrid(_StrictFrozenModel):
    """Immutable three-profile grid committed before any pilot execution."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["model-backed-static-candidate-grid-v1"] = (
        "model-backed-static-candidate-grid-v1"
    )
    dataset_version: str = Field(min_length=1)
    dataset_purpose: Literal[DatasetPurpose.DEVELOPMENT] = DatasetPurpose.DEVELOPMENT
    dataset_sha256: Sha256Hex
    selection_metric: Literal["mean-sequence-task-score-v1"] = "mean-sequence-task-score-v1"
    tie_break_rule: Literal["highest-mean-then-lexical-profile-id-v1"] = (
        "highest-mean-then-lexical-profile-id-v1"
    )
    candidate_profiles: tuple[StaticProfile, ...]

    @field_validator("dataset_purpose", mode="before")
    @classmethod
    def _accept_development_purpose(cls, value: object) -> object:
        return DatasetPurpose(value) if isinstance(value, str) else value

    @field_validator("candidate_profiles", mode="before")
    @classmethod
    def _accept_json_profiles(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("dataset_version")
    @classmethod
    def _reject_blank_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate grid dataset version must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_exact_grid(self) -> Self:
        if self.candidate_profiles != MODEL_BACKED_STATIC_CANDIDATE_PROFILES:
            raise ValueError(
                "candidate grid must equal the exact sorted Phase 3 static profile grid"
            )
        return self

    @property
    def candidate_grid_sha256(self) -> str:
        """Return the canonical scientific identity of the complete grid."""

        return canonical_sha256(self)


def load_development_pilot_candidate_grid(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> DevelopmentPilotCandidateGrid:
    """Load exact canonical JSON and optionally enforce its predeclared identity."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 hex digest")
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    grid = DevelopmentPilotCandidateGrid.model_validate_json(raw)
    canonical = canonical_json_bytes(grid)
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("candidate grid must use exact canonical JSON bytes")
    if expected_sha256 is not None and grid.candidate_grid_sha256 != expected_sha256:
        raise ValueError("candidate grid differs from its expected SHA-256")
    return grid


__all__ = [
    "DevelopmentPilotCandidateGrid",
    "MODEL_BACKED_STATIC_CANDIDATE_PROFILES",
    "load_development_pilot_candidate_grid",
]

"""Verified status aggregation over explicitly supplied closed-run evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neurallm.domain.models import NonEmptyString, PositiveInt, Sha256Hex, StrictFrozenModel
from neurallm.domain.serialization import canonical_json
from neurallm.evaluation.pilot_grid import (
    MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
    load_development_pilot_candidate_grid,
)
from neurallm.experiments.static_selection import build_static_selection_evidence
from neurallm.reporting.artifacts import ArtifactExportSummary, export_closed_run

ScientificDecision = Literal[
    "VALIDATED_POSITIVE",
    "VALIDATED_NEGATIVE",
    "INCONCLUSIVE",
    "INVALID_RUN",
]
RunTier = Literal["engineering_smoke", "development_pilot", "confirmatory"]
StatusReadiness = Literal[
    "READY_FOR_LIVE_SMOKE",
    "READY_FOR_DEVELOPMENT_PILOT",
    "READY_FOR_ADDITIONAL_DEVELOPMENT_PILOT",
    "READY_FOR_STATIC_SELECTION",
    "CONFIRMATORY_RUN_COMPLETED",
]
_EXPECTED_COMMITTED_TURNS_BY_TIER: Mapping[RunTier, int] = {
    "engineering_smoke": 20,
    "development_pilot": 240,
    "confirmatory": 2_400,
}


class _DecisionStatusProjection(BaseModel):
    """Strict status-bearing fields projected from a regenerated decision artifact."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    implementation_phase: int = Field(ge=2, le=5)
    provider_type: NonEmptyString
    committed_turns: PositiveInt
    manifest_sha256: Sha256Hex
    scientific_result_sha256: Sha256Hex
    scientific_decision: ScientificDecision | None
    database_integrity_verified: Literal[True]
    run_tier: RunTier | None = None

    @model_validator(mode="after")
    def _validate_tier_and_decision(self) -> Self:
        if self.implementation_phase == 5 and self.run_tier is None:
            raise ValueError("Phase 5 status evidence must declare a run tier")
        if self.implementation_phase != 5 and self.run_tier is not None:
            raise ValueError("only Phase 5 status evidence may declare a run tier")
        if self.run_tier is not None:
            expected_turns = _EXPECTED_COMMITTED_TURNS_BY_TIER[self.run_tier]
            if self.committed_turns != expected_turns:
                raise ValueError(
                    f"{self.run_tier} status evidence must contain exactly "
                    f"{expected_turns} committed turns"
                )
        if self.run_tier == "confirmatory":
            if self.provider_type != "llama_cpp":
                raise ValueError("confirmatory status evidence must use llama_cpp")
            if self.scientific_decision is None:
                raise ValueError("confirmatory status evidence must contain a decision")
        elif self.scientific_decision is not None:
            raise ValueError("non-confirmatory status evidence cannot contain a decision")
        return self


class VerifiedStatusEvidence(StrictFrozenModel):
    """One integrity-checked closed run contributing to a status snapshot."""

    source_kind: Literal["run_directory", "status_artifact"]
    run_directory: NonEmptyString
    status_artifact: NonEmptyString
    implementation_phase: int = Field(ge=2, le=5)
    provider_type: NonEmptyString
    committed_turns: PositiveInt
    manifest_sha256: Sha256Hex
    scientific_result_sha256: Sha256Hex
    run_tier: RunTier | None
    scientific_decision: ScientificDecision | None


class StatusSnapshot(StrictFrozenModel):
    """Deterministic repository readiness plus explicitly verified run evidence."""

    readiness: StatusReadiness = "READY_FOR_LIVE_SMOKE"
    scientific_decision: ScientificDecision | None = None
    live_provider_validated: bool = False
    live_smoke_completed: bool = False
    development_pilot_completed: bool = False
    static_selection_ready: bool = False
    confirmatory_run_completed: bool = False
    status_evidence: tuple[VerifiedStatusEvidence, ...] = ()


def _resolve_explicit_path(path: Path, *, expected_kind: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{expected_kind} must be a pathlib.Path")
    resolved = path.resolve(strict=True)
    if expected_kind == "run directory" and not resolved.is_dir():
        raise ValueError("status run directory must be a directory")
    if expected_kind in {"status artifact", "candidate grid"} and not resolved.is_file():
        raise ValueError(f"{expected_kind} must be a file")
    return resolved


def _canonical_json_object(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("status artifact must contain one JSON object with string keys")
    if raw not in {canonical_json(payload), f"{canonical_json(payload)}\n"}:
        raise ValueError("status artifact must use canonical JSON with at most one newline")
    return payload


def _verify_summary(
    run_directory: Path,
    projection: _DecisionStatusProjection,
    summary: ArtifactExportSummary,
) -> None:
    if summary.output_directory != run_directory:
        raise ValueError("exported status evidence resolved to a different run directory")
    expected = {
        "implementation_phase": summary.implementation_phase,
        "committed_turns": summary.committed_turns,
        "manifest_sha256": summary.manifest_sha256,
        "scientific_result_sha256": summary.scientific_result_sha256,
        "scientific_decision": summary.scientific_decision,
    }
    observed = {
        "implementation_phase": projection.implementation_phase,
        "committed_turns": projection.committed_turns,
        "manifest_sha256": projection.manifest_sha256,
        "scientific_result_sha256": projection.scientific_result_sha256,
        "scientific_decision": projection.scientific_decision,
    }
    if observed != expected:
        raise ValueError("regenerated decision artifact disagrees with the verified run export")


def _verified_evidence(
    run_directory: Path,
    *,
    source_kind: Literal["run_directory", "status_artifact"],
    original_status_payload: Mapping[str, object] | None = None,
) -> VerifiedStatusEvidence:
    summary = export_closed_run(run_directory)
    status_artifact = run_directory / "decision.json"
    regenerated_payload = _canonical_json_object(status_artifact)
    if original_status_payload is not None and regenerated_payload != original_status_payload:
        raise ValueError("status artifact does not match its verified canonical run store")
    projection = _DecisionStatusProjection.model_validate(regenerated_payload)
    _verify_summary(run_directory, projection, summary)
    return VerifiedStatusEvidence(
        source_kind=source_kind,
        run_directory=str(run_directory),
        status_artifact=str(status_artifact),
        implementation_phase=projection.implementation_phase,
        provider_type=projection.provider_type,
        committed_turns=projection.committed_turns,
        manifest_sha256=projection.manifest_sha256,
        scientific_result_sha256=projection.scientific_result_sha256,
        run_tier=projection.run_tier,
        scientific_decision=projection.scientific_decision,
    )


def _unique_sorted(paths: Sequence[Path], *, expected_kind: str) -> tuple[Path, ...]:
    resolved = {_resolve_explicit_path(path, expected_kind=expected_kind) for path in paths}
    return tuple(sorted(resolved, key=lambda path: str(path)))


def load_verified_status(
    run_directories: Sequence[Path] = (),
    status_artifacts: Sequence[Path] = (),
    candidate_grid_path: Path | None = None,
) -> StatusSnapshot:
    """Aggregate only explicit closed runs whose canonical stores reproduce their status."""

    if run_directories and status_artifacts:
        raise ValueError("status accepts run directories or status artifacts, not both")
    if candidate_grid_path is not None and not isinstance(candidate_grid_path, Path):
        raise TypeError("candidate grid must be a pathlib.Path or None")

    evidence: list[VerifiedStatusEvidence] = []
    for run_directory in _unique_sorted(run_directories, expected_kind="run directory"):
        evidence.append(_verified_evidence(run_directory, source_kind="run_directory"))

    for status_artifact in _unique_sorted(
        status_artifacts,
        expected_kind="status artifact",
    ):
        if status_artifact.name != "decision.json":
            raise ValueError("status artifact must be the adjacent decision.json")
        original_payload = _canonical_json_object(status_artifact)
        evidence.append(
            _verified_evidence(
                status_artifact.parent,
                source_kind="status_artifact",
                original_status_payload=original_payload,
            )
        )

    confirmatory = tuple(item for item in evidence if item.run_tier == "confirmatory")
    if len(confirmatory) > 1:
        raise ValueError("status evidence contains multiple distinct confirmatory runs")

    live_evidence = tuple(item for item in evidence if item.provider_type == "llama_cpp")
    live_smoke_completed = any(item.run_tier == "engineering_smoke" for item in live_evidence)
    development_pilot_completed = any(
        item.run_tier == "development_pilot" for item in live_evidence
    )
    pilot_run_directories = tuple(
        Path(item.run_directory) for item in live_evidence if item.run_tier == "development_pilot"
    )
    static_selection_ready = False
    declared_candidate_count = len(MODEL_BACKED_STATIC_CANDIDATE_PROFILES)
    candidate_grid = None
    if candidate_grid_path is not None:
        resolved_candidate_grid = _resolve_explicit_path(
            candidate_grid_path,
            expected_kind="candidate grid",
        )
        candidate_grid = load_development_pilot_candidate_grid(resolved_candidate_grid)
    if candidate_grid is None:
        if len(pilot_run_directories) >= declared_candidate_count:
            raise ValueError(
                f"{declared_candidate_count} or more development-pilot runs require "
                "an explicit candidate grid"
            )
    elif len(pilot_run_directories) >= len(candidate_grid.candidate_profiles):
        build_static_selection_evidence(pilot_run_directories, candidate_grid)
        static_selection_ready = True
    confirmatory_run_completed = bool(confirmatory)
    if confirmatory_run_completed:
        readiness: StatusReadiness = "CONFIRMATORY_RUN_COMPLETED"
    elif static_selection_ready:
        readiness = "READY_FOR_STATIC_SELECTION"
    elif development_pilot_completed:
        readiness = "READY_FOR_ADDITIONAL_DEVELOPMENT_PILOT"
    elif live_smoke_completed:
        readiness = "READY_FOR_DEVELOPMENT_PILOT"
    else:
        readiness = "READY_FOR_LIVE_SMOKE"

    return StatusSnapshot(
        readiness=readiness,
        scientific_decision=(None if not confirmatory else confirmatory[0].scientific_decision),
        live_provider_validated=bool(live_evidence),
        live_smoke_completed=live_smoke_completed,
        development_pilot_completed=development_pilot_completed,
        static_selection_ready=static_selection_ready,
        confirmatory_run_completed=confirmatory_run_completed,
        status_evidence=tuple(evidence),
    )


__all__ = [
    "ScientificDecision",
    "StatusSnapshot",
    "VerifiedStatusEvidence",
    "load_verified_status",
]

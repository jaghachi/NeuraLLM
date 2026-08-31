"""Canonical constructors for development-pilot static-selection evidence."""

from __future__ import annotations

from neurallm.domain.models import RunManifest
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.evaluation.pilot_grid import DevelopmentPilotCandidateGrid
from neurallm.evaluation.pilot_selection import (
    DevelopmentPilotCandidateEvidence,
    DevelopmentPilotStaticSelectionEvidence,
    DevelopmentPilotTurnEvidence,
    aggregate_pilot_unit_scores,
    pilot_turn_sort_key,
)
from neurallm.evaluation.selection import (
    StaticCandidateResult,
    StaticProfile,
    select_best_static,
)
from neurallm.storage.models import RunFinalization


def build_development_pilot_candidate_evidence(
    *,
    source_run_manifest: RunManifest,
    source_run_finalization: RunFinalization,
    profile: StaticProfile,
    turns: tuple[DevelopmentPilotTurnEvidence, ...],
) -> DevelopmentPilotCandidateEvidence:
    """Build and validate one candidate from reconstructed committed turns."""

    ordered_turns = tuple(sorted(turns, key=pilot_turn_sort_key))
    unit_keys, unit_scores = aggregate_pilot_unit_scores(ordered_turns)
    manifest_sha256 = canonical_sha256(source_run_manifest)
    finalization_sha256 = canonical_sha256(source_run_finalization)
    analysis_payload = {
        "schema_version": 1,
        "implementation_version": "development-pilot-static-candidate-analysis-v1",
        "source_run_manifest_sha256": manifest_sha256,
        "source_run_finalization_sha256": finalization_sha256,
        "source_scientific_result_sha256": source_run_finalization.scientific_result_sha256,
        "dataset_purpose": DatasetPurpose.DEVELOPMENT,
        "dataset_sha256": source_run_manifest.dataset_hash,
        "policy_id": "best_static",
        "aggregation_version": "mean-controller-seed-then-turn-v1",
        "profile": profile,
        "turns": ordered_turns,
        "development_unit_keys": unit_keys,
        "unit_scores": unit_scores,
    }
    return DevelopmentPilotCandidateEvidence(
        source_run_manifest=source_run_manifest,
        source_run_manifest_sha256=manifest_sha256,
        source_run_finalization=source_run_finalization,
        source_run_finalization_sha256=finalization_sha256,
        profile=profile,
        turns=ordered_turns,
        development_unit_keys=unit_keys,
        unit_scores=unit_scores,
        analysis_input_sha256=canonical_sha256(analysis_payload),
    )


def build_development_pilot_static_selection_evidence(
    candidates: tuple[DevelopmentPilotCandidateEvidence, ...],
    candidate_grid: DevelopmentPilotCandidateGrid,
) -> DevelopmentPilotStaticSelectionEvidence:
    """Select the winner and freeze the complete cross-run evidence envelope."""

    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.profile.profile_id))
    if not isinstance(candidate_grid, DevelopmentPilotCandidateGrid):
        raise TypeError("candidate_grid must be a DevelopmentPilotCandidateGrid")
    if len(ordered) != len(candidate_grid.candidate_profiles):
        raise ValueError("model-backed static selection requires the complete candidate grid")
    first = ordered[0]
    selection_record = select_best_static(
        tuple(
            StaticCandidateResult(profile=candidate.profile, unit_scores=candidate.unit_scores)
            for candidate in ordered
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=first.source_run_manifest.dataset_hash,
        development_unit_keys=first.development_unit_keys,
    )
    payload = {
        "schema_version": 1,
        "implementation_version": "development-pilot-static-selection-evidence-v1",
        "dataset_purpose": DatasetPurpose.DEVELOPMENT,
        "aggregation_version": "mean-controller-seed-then-turn-v1",
        "candidate_grid": candidate_grid,
        "candidate_grid_sha256": candidate_grid.candidate_grid_sha256,
        "candidates": ordered,
        "selection_record": selection_record,
    }
    return DevelopmentPilotStaticSelectionEvidence(
        candidate_grid=candidate_grid,
        candidate_grid_sha256=candidate_grid.candidate_grid_sha256,
        candidates=ordered,
        selection_record=selection_record,
        evidence_sha256=canonical_sha256(payload),
    )


__all__ = [
    "build_development_pilot_candidate_evidence",
    "build_development_pilot_static_selection_evidence",
]

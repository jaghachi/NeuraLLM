"""Hash-bound evidence for development-pilot static-profile selection."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from neurallm.domain.models import (
    RunManifest,
    Sha256Hex,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import DatasetPurpose, MatchedUnitKey
from neurallm.evaluation.pilot_grid import DevelopmentPilotCandidateGrid
from neurallm.evaluation.pilot_selection_turns import (
    DevelopmentPilotTurnEvidence,
    aggregate_pilot_unit_scores,
    pilot_turn_sort_key,
    prompt_input_sha256,
)
from neurallm.evaluation.selection import (
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    select_best_static,
)
from neurallm.providers.llama_cpp import require_llama_cpp_provider_binding
from neurallm.providers.llama_cpp_evidence import (
    reconstruct_llama_cpp_generation_binding,
)
from neurallm.storage.models import RunFinalization

_PILOT_TURN_COUNT = 240
_STATIC_TURN_COUNT = 48


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


def _candidate_analysis_payload(
    candidate: DevelopmentPilotCandidateEvidence,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation_version": "development-pilot-static-candidate-analysis-v1",
        "source_run_manifest_sha256": candidate.source_run_manifest_sha256,
        "source_run_finalization_sha256": candidate.source_run_finalization_sha256,
        "source_scientific_result_sha256": (
            candidate.source_run_finalization.scientific_result_sha256
        ),
        "dataset_purpose": DatasetPurpose.DEVELOPMENT,
        "dataset_sha256": candidate.source_run_manifest.dataset_hash,
        "policy_id": "best_static",
        "aggregation_version": "mean-controller-seed-then-turn-v1",
        "profile": candidate.profile,
        "turns": candidate.turns,
        "development_unit_keys": candidate.development_unit_keys,
        "unit_scores": candidate.unit_scores,
    }


class DevelopmentPilotCandidateEvidence(_StrictFrozenModel):
    """One live pilot run's complete static-candidate analysis binding."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["development-pilot-static-candidate-v1"] = (
        "development-pilot-static-candidate-v1"
    )
    source_run_manifest: RunManifest
    source_run_manifest_sha256: Sha256Hex
    source_run_finalization: RunFinalization
    source_run_finalization_sha256: Sha256Hex
    profile: StaticProfile
    turns: tuple[DevelopmentPilotTurnEvidence, ...]
    development_unit_keys: tuple[MatchedUnitKey, ...]
    unit_scores: tuple[float, ...]
    analysis_input_sha256: Sha256Hex

    @field_validator("turns", "development_unit_keys", "unit_scores", mode="before")
    @classmethod
    def _accept_json_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_candidate_evidence(self) -> Self:
        manifest = self.source_run_manifest
        finalization = self.source_run_finalization
        if self.source_run_manifest_sha256 != canonical_sha256(manifest):
            raise ValueError("pilot source manifest hash does not match its evidence")
        if self.source_run_finalization_sha256 != canonical_sha256(finalization):
            raise ValueError("pilot source finalization hash does not match its evidence")
        if finalization.manifest_sha256 != self.source_run_manifest_sha256:
            raise ValueError("pilot source finalization targets another manifest")
        accounting = finalization.execution_accounting
        if (
            finalization.expected_condition_count != _PILOT_TURN_COUNT
            or accounting is None
            or accounting.planned_logical_generations != _PILOT_TURN_COUNT
            or accounting.committed_logical_generations != _PILOT_TURN_COUNT
            or accounting.successful_responses != _PILOT_TURN_COUNT
            or accounting.uncertain_dispatches != 0
        ):
            raise ValueError("static selection requires an exactly closed 240-turn pilot")
        if (
            manifest.run_tier != "development_pilot"
            or manifest.decision_rule_version != "development-pilot-no-scientific-decision-v1"
            or manifest.provider_identity.provider_type != "llama_cpp"
            or manifest.provider_identity.model_sha256 is None
            or not manifest.working_tree_clean
            or manifest.candidate_grid_sha256 is None
        ):
            raise ValueError(
                "static selection requires a clean digest-bound llama_cpp development pilot"
            )
        try:
            require_llama_cpp_provider_binding(
                manifest.provider_identity,
                manifest.provider_effective_configuration_json,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pilot source lacks internally consistent llama_cpp preflight evidence"
            ) from exc
        if any(
            value is not None
            for value in (
                manifest.evaluation_spec_json,
                manifest.evaluation_spec_sha256,
                manifest.turn_input_evidence_sha256,
                manifest.phase3_analysis_contract_sha256,
                manifest.preregistration_sha256,
                manifest.confirmatory_analysis_contract_sha256,
                manifest.static_selection_evidence_sha256,
            )
        ):
            raise ValueError("development-pilot source manifest contains claim-bearing evidence")
        if len(self.turns) != _STATIC_TURN_COUNT or self.turns != tuple(
            sorted(self.turns, key=pilot_turn_sort_key)
        ):
            raise ValueError("pilot candidate requires 48 canonically ordered best_static turns")
        if len({turn.condition.condition_id for turn in self.turns}) != len(self.turns):
            raise ValueError("pilot candidate contains duplicate best_static conditions")
        if any(
            turn.condition.condition_id not in finalization.expected_condition_ids
            for turn in self.turns
        ):
            raise ValueError("pilot candidate turn is absent from source finalization")

        sequences = {turn.condition.prompt_sequence_id for turn in self.turns}
        model_seeds = {turn.condition.model_seed for turn in self.turns}
        controller_seeds = {turn.condition.controller_seed for turn in self.turns}
        if (len(sequences), len(model_seeds), len(controller_seeds)) != (6, 2, 1):
            raise ValueError("pilot candidate does not have the exact 6x2x1 selection axes")
        expected_coordinates = {
            (sequence_id, model_seed, controller_seed, turn_index)
            for sequence_id in sequences
            for model_seed in model_seeds
            for controller_seed in controller_seeds
            for turn_index in range(4)
        }
        actual_coordinates = {
            (
                turn.condition.prompt_sequence_id,
                turn.condition.model_seed,
                turn.condition.controller_seed,
                turn.condition.turn_index,
            )
            for turn in self.turns
        }
        if actual_coordinates != expected_coordinates:
            raise ValueError("pilot candidate best_static turns do not form the exact grid")
        if model_seeds != set(manifest.seed_schedule.model_seeds) or controller_seeds != set(
            manifest.seed_schedule.controller_seeds
        ):
            raise ValueError("pilot candidate turns differ from the manifest seed schedule")
        if len({turn.condition.dataset_version for turn in self.turns}) != 1:
            raise ValueError("pilot candidate spans multiple dataset versions")
        if len({turn.condition.experiment_id for turn in self.turns}) != 1:
            raise ValueError("pilot candidate spans multiple experiment IDs")
        profile_values = (
            self.profile.temperature,
            self.profile.top_p,
            self.profile.top_k,
            self.profile.presence_penalty,
            self.profile.max_tokens,
        )
        for turn in self.turns:
            condition = turn.condition
            parameters = turn.decoding_parameters
            if (
                condition.provider_identity_id != manifest.provider_identity.identity_id
                or condition.base_decoding_profile_id != self.profile.profile_id
                or (
                    parameters.temperature,
                    parameters.top_p,
                    parameters.top_k,
                    parameters.presence_penalty,
                    parameters.max_tokens,
                )
                != profile_values
            ):
                raise ValueError("pilot candidate profile differs from committed request evidence")
            request, response = reconstruct_llama_cpp_generation_binding(
                condition=condition,
                decoding_parameters=parameters,
                provider_identity=manifest.provider_identity,
                metadata=turn.generation_metadata,
            )
            if (
                canonical_sha256(request) != turn.request_sha256
                or canonical_sha256(response) != turn.response_sha256
            ):
                raise ValueError(
                    "pilot candidate wire evidence does not bind its domain request/response"
                )
        expected_keys, expected_scores = aggregate_pilot_unit_scores(self.turns)
        if self.development_unit_keys != expected_keys or self.unit_scores != expected_scores:
            raise ValueError("pilot candidate scores do not match committed turn metrics")
        if self.analysis_input_sha256 != canonical_sha256(_candidate_analysis_payload(self)):
            raise ValueError("pilot candidate analysis hash does not match its evidence")
        return self


def _selection_evidence_payload(
    evidence: DevelopmentPilotStaticSelectionEvidence,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation_version": "development-pilot-static-selection-evidence-v1",
        "dataset_purpose": DatasetPurpose.DEVELOPMENT,
        "aggregation_version": "mean-controller-seed-then-turn-v1",
        "candidate_grid": evidence.candidate_grid,
        "candidate_grid_sha256": evidence.candidate_grid_sha256,
        "candidates": evidence.candidates,
        "selection_record": evidence.selection_record,
    }


class DevelopmentPilotStaticSelectionEvidence(_StrictFrozenModel):
    """Frozen selector result and every model-backed development source binding."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["development-pilot-static-selection-evidence-v1"] = (
        "development-pilot-static-selection-evidence-v1"
    )
    dataset_purpose: Literal[DatasetPurpose.DEVELOPMENT] = DatasetPurpose.DEVELOPMENT
    aggregation_version: Literal["mean-controller-seed-then-turn-v1"] = (
        "mean-controller-seed-then-turn-v1"
    )
    candidate_grid: DevelopmentPilotCandidateGrid
    candidate_grid_sha256: Sha256Hex
    candidates: tuple[DevelopmentPilotCandidateEvidence, ...]
    selection_record: StaticSelectionRecord
    evidence_sha256: Sha256Hex

    @field_validator("dataset_purpose", mode="before")
    @classmethod
    def _accept_development_purpose(cls, value: object) -> object:
        return DatasetPurpose(value) if isinstance(value, str) else value

    @field_validator("candidates", mode="before")
    @classmethod
    def _accept_json_candidates(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_selection_evidence(self) -> Self:
        if self.candidate_grid_sha256 != self.candidate_grid.candidate_grid_sha256:
            raise ValueError("candidate grid hash does not match its canonical identity")
        if len(self.candidates) != len(self.candidate_grid.candidate_profiles):
            raise ValueError("model-backed static selection requires the complete candidate grid")
        profile_ids = tuple(candidate.profile.profile_id for candidate in self.candidates)
        if profile_ids != tuple(sorted(set(profile_ids))):
            raise ValueError("pilot candidates must be sorted by unique profile ID")
        profile_values = {
            (
                candidate.profile.temperature,
                candidate.profile.top_p,
                candidate.profile.top_k,
                candidate.profile.presence_penalty,
            )
            for candidate in self.candidates
        }
        if len(profile_values) != len(self.candidates):
            raise ValueError("pilot candidates must use distinct tunable sampling parameters")
        if len({candidate.profile.max_tokens for candidate in self.candidates}) != 1:
            raise ValueError("pilot candidates must share one fixed max_tokens budget")

        first = self.candidates[0]
        first_manifest = first.source_run_manifest
        if tuple(candidate.profile for candidate in self.candidates) != (
            self.candidate_grid.candidate_profiles
        ):
            raise ValueError("pilot candidates differ from the predeclared candidate grid")
        if any(
            candidate.source_run_manifest.candidate_grid_sha256 != self.candidate_grid_sha256
            for candidate in self.candidates
        ):
            raise ValueError("pilot source manifest differs from the candidate-grid identity")
        if (
            self.candidate_grid.dataset_sha256 != first_manifest.dataset_hash
            or self.candidate_grid.dataset_purpose is not DatasetPurpose.DEVELOPMENT
        ):
            raise ValueError("candidate grid differs from the pilot development dataset")
        first_prompt_inputs = {
            pilot_turn_sort_key(turn): prompt_input_sha256(turn) for turn in first.turns
        }
        invariant_manifest_fields = (
            "source_commit",
            "working_tree_clean",
            "dataset_hash",
            "provider_identity",
            "provider_effective_configuration_json",
            "policy_config_hashes",
            "matched_history_policy_sources",
            "metric_versions",
            "seed_schedule",
            "action_bounds",
            "decoding_bounds",
            "decision_rule_version",
            "database_schema_version",
            "run_tier",
        )
        for candidate in self.candidates[1:]:
            manifest = candidate.source_run_manifest
            if any(
                getattr(manifest, field_name) != getattr(first_manifest, field_name)
                for field_name in invariant_manifest_fields
            ):
                raise ValueError("pilot candidate runs differ outside the declared static profile")
            if candidate.development_unit_keys != first.development_unit_keys:
                raise ValueError("pilot candidates do not share exact development unit keys")
            candidate_prompt_inputs = {
                pilot_turn_sort_key(turn): prompt_input_sha256(turn) for turn in candidate.turns
            }
            if candidate_prompt_inputs != first_prompt_inputs:
                raise ValueError("pilot candidate runs do not share exact prompt-side inputs")
        dataset_versions = {
            turn.condition.dataset_version
            for candidate in self.candidates
            for turn in candidate.turns
        }
        if len(dataset_versions) != 1:
            raise ValueError("pilot candidate runs do not share one dataset version")
        if dataset_versions != {self.candidate_grid.dataset_version}:
            raise ValueError("candidate grid differs from the pilot dataset version")
        config_hashes = {
            candidate.source_run_manifest.experiment_config_hash for candidate in self.candidates
        }
        scientific_ids = {
            candidate.source_run_manifest.scientific_identity_sha256
            for candidate in self.candidates
        }
        if len(config_hashes) != len(self.candidates) or len(scientific_ids) != len(
            self.candidates
        ):
            raise ValueError("pilot candidate runs must have distinct frozen identities")

        expected_selection = select_best_static(
            tuple(
                StaticCandidateResult(
                    profile=candidate.profile,
                    unit_scores=candidate.unit_scores,
                )
                for candidate in self.candidates
            ),
            dataset_purpose=DatasetPurpose.DEVELOPMENT,
            dataset_sha256=first_manifest.dataset_hash,
            development_unit_keys=first.development_unit_keys,
        )
        if self.selection_record != expected_selection:
            raise ValueError("static selection record does not derive from bound pilot evidence")
        if (
            self.selection_record.selection_metric != self.candidate_grid.selection_metric
            or self.selection_record.tie_break_rule != self.candidate_grid.tie_break_rule
        ):
            raise ValueError("static selection rules differ from the candidate grid")
        if self.evidence_sha256 != canonical_sha256(_selection_evidence_payload(self)):
            raise ValueError("static selection evidence hash does not match its canonical payload")
        return self


__all__ = [
    "DevelopmentPilotCandidateEvidence",
    "DevelopmentPilotStaticSelectionEvidence",
    "DevelopmentPilotTurnEvidence",
    "aggregate_pilot_unit_scores",
    "pilot_turn_sort_key",
]

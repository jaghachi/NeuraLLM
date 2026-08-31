"""Provider-free derivation and publication of live-pilot static selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from neurallm.control.policy import PolicyTrace
from neurallm.domain.models import ControllerAction
from neurallm.domain.serialization import canonical_json, canonical_json_bytes, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.evaluation.pilot_grid import (
    DevelopmentPilotCandidateGrid,
    load_development_pilot_candidate_grid,
)
from neurallm.evaluation.pilot_selection import (
    DevelopmentPilotCandidateEvidence,
    DevelopmentPilotStaticSelectionEvidence,
    DevelopmentPilotTurnEvidence,
)
from neurallm.evaluation.pilot_selection_builders import (
    build_development_pilot_candidate_evidence,
    build_development_pilot_static_selection_evidence,
)
from neurallm.evaluation.selection import StaticProfile
from neurallm.experiments.dataset import PromptDataset
from neurallm.experiments.runner import DetailedAppliedPolicyTrace
from neurallm.metrics.base import MetricContext
from neurallm.metrics.deterministic import compute_response_metrics
from neurallm.providers.base import GenerationRequest
from neurallm.providers.llama_cpp_evidence import (
    reconstruct_llama_cpp_generation_binding,
)
from neurallm.storage import (
    SQLiteRunStore,
    StoreInvariantError,
    TurnState,
    scientific_result_sha256,
)
from neurallm.storage.models import StoredTurn, TurnInputEvidence


@dataclass(frozen=True, slots=True)
class StaticSelectionPublication:
    """One immutable selector artifact and its incidental publication location."""

    output_path: Path
    evidence: DevelopmentPilotStaticSelectionEvidence
    source_run_directories: tuple[Path, ...]
    candidate_grid_path: Path
    created: bool


def _zero_action() -> ControllerAction:
    return ControllerAction(
        temperature_delta=0.0,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )


def _require_static_trace(turn: StoredTurn) -> None:
    policy_trace_json = turn.policy_trace_json
    if policy_trace_json is None:
        raise StoreInvariantError("committed best_static turn lacks policy trace evidence")
    try:
        payload: object = json.loads(policy_trace_json)
    except json.JSONDecodeError as exc:
        raise StoreInvariantError("best_static policy trace is not valid JSON") from exc
    try:
        trace = DetailedAppliedPolicyTrace.model_validate(payload)
    except ValueError as exc:
        raise StoreInvariantError("best_static policy trace fails its declared schema") from exc
    if canonical_json(trace) != policy_trace_json:
        raise StoreInvariantError("best_static policy trace is not canonical JSON")
    zero = _zero_action()
    application = trace.action_application
    nested_trace = trace.policy_trace
    if (
        trace.policy_id != "best_static"
        or trace.policy_id != turn.condition.policy_id
        or trace.turn_index != turn.condition.turn_index
        or trace.history_access != "none"
        or trace.observation_has_previous_response
        or trace.action != zero
        or application.raw_action != zero
        or application.step_clamped_action != zero
        or application.final_decoding_parameters != turn.request.decoding_parameters
        or application.saturation.any_saturation
        or not isinstance(nested_trace, PolicyTrace)
        or nested_trace.policy_id != turn.condition.policy_id
        or nested_trace.turn_index != turn.condition.turn_index
        or nested_trace.action != zero
    ):
        raise StoreInvariantError("best_static pilot trace is not stateless and exactly zero")


def _profile_from_static_turns(static_turns: tuple[StoredTurn, ...]) -> StaticProfile:
    if not static_turns:
        raise StoreInvariantError("pilot source run contains no best_static turns")
    first = static_turns[0]
    condition = first.condition
    request = first.request
    parameters = request.decoding_parameters
    return StaticProfile(
        profile_id=condition.base_decoding_profile_id,
        temperature=parameters.temperature,
        top_p=parameters.top_p,
        top_k=parameters.top_k,
        presence_penalty=parameters.presence_penalty,
        max_tokens=parameters.max_tokens,
    )


def _candidate_from_run_directory(run_directory: Path) -> DevelopmentPilotCandidateEvidence:
    resolved_directory = run_directory.expanduser().resolve(strict=True)
    if not resolved_directory.is_dir():
        raise ValueError(f"candidate run path is not a directory: {resolved_directory}")
    database_path = (resolved_directory / "run.sqlite3").resolve(strict=True)
    if not database_path.is_file():
        raise ValueError(f"candidate run database is not a file: {database_path}")

    with SQLiteRunStore(database_path) as store:
        store.verify_integrity()
        manifest = store.get_manifest()
        finalization = store.get_finalization()
        if manifest is None or finalization is None:
            raise StoreInvariantError("static selection requires a manifest-bound finalized run")
        stored_turns = store.list_turns()
        if (
            len(stored_turns) != 240
            or any(turn.state is not TurnState.COMMITTED for turn in stored_turns)
            or tuple(sorted(turn.condition_id for turn in stored_turns))
            != finalization.expected_condition_ids
        ):
            raise StoreInvariantError("static selection source is not an exact closed pilot")
        if finalization.scientific_result_sha256 != scientific_result_sha256(stored_turns):
            raise StoreInvariantError(
                "static selection source finalization does not match committed turn evidence"
            )
        static_turns = tuple(
            turn for turn in stored_turns if turn.condition.policy_id == "best_static"
        )
        profile = _profile_from_static_turns(static_turns)
        turn_evidence: list[DevelopmentPilotTurnEvidence] = []
        for turn in static_turns:
            if turn.response is None or turn.metrics is None:
                raise StoreInvariantError("committed best_static turn lacks response metrics")
            _require_static_trace(turn)
            input_evidence = store.get_turn_input(turn.condition_id)
            if input_evidence is None:
                raise StoreInvariantError("best_static turn lacks prompt-side input evidence")
            reconstructed = compute_response_metrics(
                MetricContext(
                    prompt_case_id=input_evidence.prompt_case_id,
                    prompt_family=input_evidence.prompt_family,
                    prompt=turn.request.prompt,
                    response_text=turn.response.text,
                    validator=input_evidence.validator,
                )
            )
            if reconstructed != turn.metrics:
                raise StoreInvariantError("best_static metrics do not reconstruct exactly")
            if (
                turn.response.provider_identity != manifest.provider_identity
                or turn.response.effective_parameters != turn.request.decoding_parameters
            ):
                raise StoreInvariantError("best_static response differs from its provider binding")
            turn_evidence.append(
                DevelopmentPilotTurnEvidence(
                    condition=turn.condition,
                    request_sha256=turn.request_sha256,
                    response_sha256=canonical_sha256(turn.response),
                    generation_metadata=turn.response.raw_metadata,
                    decoding_parameters=turn.request.decoding_parameters,
                    turn_input=input_evidence,
                    task_score=turn.metrics.task_score,
                )
            )
    return build_development_pilot_candidate_evidence(
        source_run_manifest=manifest,
        source_run_finalization=finalization,
        profile=profile,
        turns=tuple(turn_evidence),
    )


def validate_static_selection_evidence_against_dataset(
    evidence: DevelopmentPilotStaticSelectionEvidence,
    dataset: PromptDataset,
) -> None:
    """Bind every selected turn to its declared development prompt and request."""

    if not isinstance(evidence, DevelopmentPilotStaticSelectionEvidence):
        raise TypeError("evidence must be DevelopmentPilotStaticSelectionEvidence")
    if not isinstance(dataset, PromptDataset):
        raise TypeError("dataset must be PromptDataset")
    if dataset.purpose is not DatasetPurpose.DEVELOPMENT:
        raise ValueError("static selection requires a development-purpose dataset")
    if any(
        candidate.source_run_manifest.dataset_hash != dataset.dataset_hash
        for candidate in evidence.candidates
    ):
        raise ValueError("static selection evidence does not match the development dataset hash")
    if (
        evidence.candidate_grid.dataset_version != dataset.version
        or evidence.candidate_grid.dataset_sha256 != dataset.dataset_hash
        or evidence.candidate_grid.dataset_purpose is not dataset.purpose
    ):
        raise ValueError("static selection candidate grid does not match the development dataset")

    sequences = {sequence.sequence_id: sequence for sequence in dataset.sequences}
    expected_coordinates = {
        (sequence.sequence_id, turn_index)
        for sequence in dataset.sequences
        for turn_index in range(len(sequence.cases))
    }
    for candidate in evidence.candidates:
        if {turn.condition.dataset_version for turn in candidate.turns} != {dataset.version}:
            raise ValueError(
                "static selection evidence does not match the development dataset version"
            )
        actual_coordinates = {
            (turn.condition.prompt_sequence_id, turn.condition.turn_index)
            for turn in candidate.turns
        }
        if actual_coordinates != expected_coordinates:
            raise ValueError(
                "static selection evidence does not cover the exact development prompt grid"
            )
        for turn in candidate.turns:
            condition = turn.condition
            sequence = sequences[condition.prompt_sequence_id]
            prompt_case = sequence.cases[condition.turn_index]
            expected_input = TurnInputEvidence(
                condition_id=condition.condition_id,
                prompt_case_id=prompt_case.case_id,
                prompt_family=prompt_case.prompt_family,
                prompt_features=prompt_case.prompt_features,
                validator=prompt_case.validator,
            )
            if turn.turn_input != expected_input:
                raise ValueError(
                    "static selection turn input differs from the development prompt case"
                )
            request = GenerationRequest(
                prompt=prompt_case.prompt,
                decoding_parameters=turn.decoding_parameters,
                condition=condition,
            )
            if turn.request_sha256 != canonical_sha256(request):
                raise ValueError(
                    "static selection request hash differs from the development prompt"
                )
            wire_request, wire_response = reconstruct_llama_cpp_generation_binding(
                condition=condition,
                decoding_parameters=turn.decoding_parameters,
                provider_identity=candidate.source_run_manifest.provider_identity,
                metadata=turn.generation_metadata,
            )
            if wire_request != request or canonical_sha256(wire_response) != turn.response_sha256:
                raise ValueError(
                    "static selection wire evidence differs from the development request/response"
                )


def build_static_selection_evidence(
    candidate_run_directories: tuple[Path, ...],
    candidate_grid: DevelopmentPilotCandidateGrid,
) -> DevelopmentPilotStaticSelectionEvidence:
    """Derive one selection envelope without constructing or contacting a provider."""

    if not isinstance(candidate_run_directories, tuple) or not all(
        isinstance(path, Path) for path in candidate_run_directories
    ):
        raise TypeError("candidate_run_directories must be a tuple of pathlib.Path values")
    if not isinstance(candidate_grid, DevelopmentPilotCandidateGrid):
        raise TypeError("candidate_grid must be a DevelopmentPilotCandidateGrid")
    if len(candidate_run_directories) != len(candidate_grid.candidate_profiles):
        raise ValueError("static selection requires one run for every declared grid profile")
    resolved = tuple(path.expanduser().resolve(strict=True) for path in candidate_run_directories)
    if len(set(resolved)) != len(resolved):
        raise ValueError("candidate run directories must be unique")
    candidates = tuple(_candidate_from_run_directory(path) for path in resolved)
    return build_development_pilot_static_selection_evidence(candidates, candidate_grid)


def load_static_selection_evidence(path: Path) -> DevelopmentPilotStaticSelectionEvidence:
    """Load one exact canonical JSON selection artifact and revalidate all hashes."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    evidence = DevelopmentPilotStaticSelectionEvidence.model_validate_json(raw)
    if raw != canonical_json_bytes(evidence):
        raise ValueError("static selection evidence must use exact canonical JSON bytes")
    return evidence


def _publish_canonical_json(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if path.is_file() and path.read_bytes() == payload:
            return False
        raise FileExistsError(f"refusing to overwrite static selection artifact: {path}")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_file() and path.read_bytes() == payload:
                return False
            raise FileExistsError(
                f"refusing to overwrite static selection artifact: {path}"
            ) from None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def freeze_static_selection(
    candidate_run_directories: tuple[Path, ...],
    candidate_grid_path: Path,
    output_path: Path,
) -> StaticSelectionPublication:
    """Derive, publish, reload, and verify one immutable selection artifact."""

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    candidate_grid = load_development_pilot_candidate_grid(candidate_grid_path)
    evidence = build_static_selection_evidence(candidate_run_directories, candidate_grid)
    resolved_output = output_path.expanduser().resolve()
    created = _publish_canonical_json(resolved_output, canonical_json_bytes(evidence))
    reloaded = load_static_selection_evidence(resolved_output)
    if reloaded != evidence:
        raise RuntimeError("published static selection evidence changed during reload")
    return StaticSelectionPublication(
        output_path=resolved_output,
        evidence=reloaded,
        source_run_directories=tuple(
            path.expanduser().resolve(strict=True) for path in candidate_run_directories
        ),
        candidate_grid_path=candidate_grid_path.expanduser().resolve(strict=True),
        created=created,
    )


__all__ = [
    "StaticSelectionPublication",
    "build_static_selection_evidence",
    "freeze_static_selection",
    "load_static_selection_evidence",
    "validate_static_selection_evidence_against_dataset",
]

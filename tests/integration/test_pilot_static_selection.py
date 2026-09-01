"""Provider-free publication from complete model-backed development pilots."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest

import neurallm.experiments.static_selection as static_selection
from neurallm.cli import main
from neurallm.control.action_space import apply_action
from neurallm.control.policy import PolicyState, PolicyTrace
from neurallm.control.specs import (
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    RandomMatchedPolicySpec,
)
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    DecodingBounds,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
    ProviderIdentity,
    RunManifest,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_json_bytes, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.evaluation.pilot_grid import (
    MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
    DevelopmentPilotCandidateGrid,
    load_development_pilot_candidate_grid,
)
from neurallm.evaluation.pilot_selection import DevelopmentPilotStaticSelectionEvidence
from neurallm.evaluation.pilot_selection_builders import (
    build_development_pilot_candidate_evidence,
    build_development_pilot_static_selection_evidence,
)
from neurallm.evaluation.pilot_selection_turns import DevelopmentPilotTurnEvidence
from neurallm.evaluation.selection import StaticProfile
from neurallm.experiments.dataset import PromptCase, PromptDataset, PromptSequence
from neurallm.experiments.protocol import (
    DEVELOPMENT_PILOT_DECISION_RULE_VERSION,
    MODEL_BACKED_POLICY_IDS,
)
from neurallm.experiments.runner import DetailedAppliedPolicyTrace
from neurallm.experiments.static_selection import (
    _candidate_from_run_directory,
    _profile_from_static_turns,
    _publish_canonical_json,
    _require_static_trace,
    build_static_selection_evidence,
    freeze_static_selection,
    load_static_selection_evidence,
    validate_static_selection_evidence_against_dataset,
)
from neurallm.metrics.base import MetricContext
from neurallm.metrics.deterministic import METRIC_VERSIONS, compute_response_metrics
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.base import GenerationMetadata, GenerationRequest, GenerationResponse
from neurallm.providers.llama_cpp import (
    LlamaCppEffectiveConfiguration,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    DurableExecutionAccounting,
    SQLiteRunStore,
    StoreInvariantError,
    TurnInputEvidence,
    scientific_result_sha256,
)
from tests.integration.pilot_selection_helpers import build_test_static_selection_evidence

_MODEL_SEEDS = (4101, 4102)
_CONTROLLER_SEED = 5101
_SEQUENCE_IDS = tuple(f"pilot-sequence-{index:02d}" for index in range(1, 7))
_DATASET_VERSION = "model-backed-development-pilot-test-v1"
_PILOT_DATASET = PromptDataset(
    schema_version=1,
    dataset_id="model-backed-development-pilot-test",
    version=_DATASET_VERSION,
    purpose=DatasetPurpose.DEVELOPMENT,
    sequences=tuple(
        PromptSequence(
            sequence_id=sequence_id,
            cases=tuple(
                PromptCase(
                    case_id=f"{sequence_id}-turn-{turn_index}",
                    prompt_family="pilot_static_selection",
                    prompt="Return PASS exactly.",
                    prompt_features=PromptFeatures({"constraint_count": 1.0}),
                    validator=ValidatorSpec(kind="exact_match", expected_text="PASS"),
                )
                for turn_index in range(4)
            ),
        )
        for sequence_id in _SEQUENCE_IDS
    ),
)
_DATASET_SHA256 = _PILOT_DATASET.dataset_hash
_PILOT_GRID = DevelopmentPilotCandidateGrid(
    dataset_version=_DATASET_VERSION,
    dataset_purpose=DatasetPurpose.DEVELOPMENT,
    dataset_sha256=_DATASET_SHA256,
    candidate_profiles=MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
)
_ACTION_BOUNDS = ActionBounds()
_DECODING_BOUNDS = DecodingBounds()
_ZERO_ACTION = ControllerAction(
    temperature_delta=0.0,
    top_p_delta=0.0,
    top_k_delta=0,
    presence_penalty_delta=0.0,
)
_POLICY_SPECS = (
    BestStaticPolicySpec(),
    HeuristicAdaptivePolicySpec(),
    NeuralMatchedHistoryStateResetPolicySpec(),
    NeuralPersistentPolicySpec(),
    RandomMatchedPolicySpec(),
)


@dataclass(frozen=True)
class _PilotStores:
    candidate_directories: tuple[Path, Path, Path]
    candidate_grid_path: Path
    trace_tampered_directory: Path
    profile_drift_directory: Path
    provider_identity: ProviderIdentity
    provider_effective_configuration_json: str


def _provider_binding(root: Path) -> tuple[ProviderIdentity, str]:
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    config = LlamaCppProviderConfig(
        base_url="http://127.0.0.1:8080",
        model_alias="pilot-selection-integration",
        model_path=str((root / "pilot-selection.gguf").resolve()),
        model_sha256="a" * 64,
        build_id="pilot-selection-integration-build",
        chat_template_sha256=sha256(chat_template.encode("utf-8")).hexdigest(),
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        write_timeout_seconds=3.0,
        pool_timeout_seconds=4.0,
    )
    effective = LlamaCppEffectiveConfiguration(
        client_config=config,
        model_alias=config.model_alias,
        model_path=config.model_path,
        model_sha256=config.model_sha256,
        build_id=config.build_id,
        chat_template=chat_template,
        chat_template_sha256=config.chat_template_sha256,
        default_generation_settings_json=canonical_json(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "presence_penalty": 0.0,
                "n_predict": 192,
                "seed": _MODEL_SEEDS[0],
            }
        ),
        total_slots=1,
    )
    return llama_cpp_provider_identity(effective), canonical_json(effective)


def _manifest(
    *,
    profile: StaticProfile,
    provider_identity: ProviderIdentity,
    provider_effective_configuration_json: str,
) -> RunManifest:
    return RunManifest(
        source_commit="1" * 40,
        working_tree_clean=True,
        experiment_config_hash=canonical_sha256(
            {"profile_id": profile.profile_id, "kind": "pilot-config"}
        ),
        dataset_hash=_DATASET_SHA256,
        provider_config_hash=provider_identity.provider_config_hash,
        provider_identity=provider_identity,
        provider_effective_configuration_json=provider_effective_configuration_json,
        policy_config_hashes={spec.policy_id: canonical_sha256(spec) for spec in _POLICY_SPECS},
        matched_history_policy_sources={"neural_matched_history_state_reset": "neural_persistent"},
        metric_versions=METRIC_VERSIONS,
        seed_schedule=SeedSchedule(
            model_seeds=_MODEL_SEEDS,
            controller_seeds=(_CONTROLLER_SEED,),
        ),
        action_bounds=_ACTION_BOUNDS,
        decoding_bounds=_DECODING_BOUNDS,
        decision_rule_version=DEVELOPMENT_PILOT_DECISION_RULE_VERSION,
        database_schema_version=CURRENT_SCHEMA_VERSION,
        run_tier="development_pilot",
        scientific_identity_sha256=canonical_sha256(
            {"profile_id": profile.profile_id, "kind": "pilot-scientific-identity"}
        ),
        candidate_grid_sha256=_PILOT_GRID.candidate_grid_sha256,
    )


def _trace(
    request: GenerationRequest,
    *,
    tamper_application: bool,
) -> DetailedAppliedPolicyTrace:
    base_parameters = request.decoding_parameters
    if tamper_application:
        base_parameters = base_parameters.model_copy(
            update={"temperature": base_parameters.temperature + 0.1}
        )
    application = apply_action(
        base_parameters,
        _ZERO_ACTION,
        _ACTION_BOUNDS,
        _DECODING_BOUNDS,
    )
    return DetailedAppliedPolicyTrace(
        policy_id=request.condition.policy_id,
        turn_index=request.condition.turn_index,
        action=_ZERO_ACTION,
        action_application=application,
        history_access="none",
        observation_has_previous_response=False,
        policy_trace=PolicyTrace(
            policy_id=request.condition.policy_id,
            turn_index=request.condition.turn_index,
            action=_ZERO_ACTION,
        ),
    )


def _write_pilot_store(
    root: Path,
    *,
    profile: StaticProfile,
    task_passes: bool,
    provider_identity: ProviderIdentity,
    provider_effective_configuration_json: str,
    tamper_static_trace: bool = False,
) -> Path:
    run_directory = root / profile.profile_id
    run_directory.mkdir(parents=True)
    manifest = _manifest(
        profile=profile,
        provider_identity=provider_identity,
        provider_effective_configuration_json=provider_effective_configuration_json,
    )
    previous_condition_ids: dict[tuple[str, int, str], str] = {}
    condition_ids: list[str] = []
    validator = ValidatorSpec(kind="exact_match", expected_text="PASS")
    with SQLiteRunStore(run_directory / "run.sqlite3", manifest) as store:
        for sequence_id in _SEQUENCE_IDS:
            for model_seed in _MODEL_SEEDS:
                for turn_index in range(4):
                    for policy_id in MODEL_BACKED_POLICY_IDS:
                        condition = ExperimentCondition(
                            experiment_id=f"pilot-{profile.profile_id}",
                            dataset_version=_DATASET_VERSION,
                            prompt_sequence_id=sequence_id,
                            turn_index=turn_index,
                            policy_id=policy_id,
                            model_seed=model_seed,
                            controller_seed=_CONTROLLER_SEED,
                            provider_identity_id=provider_identity.identity_id,
                            base_decoding_profile_id=profile.profile_id,
                        )
                        parameters = DecodingParameters(
                            temperature=profile.temperature,
                            top_p=profile.top_p,
                            top_k=profile.top_k,
                            presence_penalty=profile.presence_penalty,
                            max_tokens=profile.max_tokens,
                            seed=model_seed,
                        )
                        request = GenerationRequest(
                            prompt="Return PASS exactly.",
                            decoding_parameters=parameters,
                            condition=condition,
                        )
                        input_evidence = TurnInputEvidence(
                            condition_id=condition.condition_id,
                            prompt_case_id=f"{sequence_id}-turn-{turn_index}",
                            prompt_family="pilot_static_selection",
                            prompt_features=PromptFeatures({"constraint_count": 1.0}),
                            validator=validator,
                        )
                        history = None
                        if turn_index > 0:
                            source_policy_id = (
                                "neural_persistent"
                                if policy_id == "neural_matched_history_state_reset"
                                else policy_id
                            )
                            history = store.history_binding_for(
                                previous_condition_ids[(sequence_id, model_seed, source_policy_id)]
                            )
                        store.prepare_turn(request, history, input_evidence)
                        store.begin_dispatch(condition.condition_id)
                        response_text = "PASS" if task_passes else "FAIL"
                        provider_request = {
                            "prompt": request.prompt,
                            "model": provider_identity.model_alias,
                            "temperature": parameters.temperature,
                            "top_p": parameters.top_p,
                            "top_k": parameters.top_k,
                            "presence_penalty": parameters.presence_penalty,
                            "n_predict": parameters.max_tokens,
                            "seed": parameters.seed,
                            "stream": False,
                            "cache_prompt": False,
                        }
                        provider_response = {
                            "content": response_text,
                            "stop": True,
                            "model": provider_identity.model_alias,
                            "generation_settings": {
                                "temperature": parameters.temperature,
                                "top_p": parameters.top_p,
                                "top_k": parameters.top_k,
                                "presence_penalty": parameters.presence_penalty,
                                "n_predict": parameters.max_tokens,
                                "max_tokens": parameters.max_tokens,
                                "seed": parameters.seed,
                            },
                        }
                        response = GenerationResponse(
                            text=response_text,
                            provider_identity=provider_identity,
                            effective_parameters=parameters,
                            raw_metadata=GenerationMetadata(
                                request_sha256=canonical_sha256(request),
                                generation_method="llama_cpp_completion_http_v1",
                                provider_request_json=canonical_json(provider_request),
                                provider_request_sha256=canonical_sha256(provider_request),
                                provider_response_json=canonical_json(provider_response),
                                provider_response_sha256=canonical_sha256(provider_response),
                            ),
                        )
                        store.persist_response(condition.condition_id, response)
                        store.persist_metrics(
                            condition.condition_id,
                            compute_response_metrics(
                                MetricContext(
                                    prompt_case_id=input_evidence.prompt_case_id,
                                    prompt_family=input_evidence.prompt_family,
                                    prompt=request.prompt,
                                    response_text=response.text,
                                    validator=validator,
                                )
                            ),
                        )
                        trace_tampered = (
                            tamper_static_trace
                            and policy_id == "best_static"
                            and sequence_id == _SEQUENCE_IDS[0]
                            and model_seed == _MODEL_SEEDS[0]
                            and turn_index == 0
                        )
                        store.commit_turn(
                            condition.condition_id,
                            PolicyState(),
                            _trace(request, tamper_application=trace_tampered),
                        )
                        previous_condition_ids[(sequence_id, model_seed, policy_id)] = (
                            condition.condition_id
                        )
                        condition_ids.append(condition.condition_id)
        aggregate_sha256 = scientific_result_sha256(store.list_turns())
        store.finalize_run(
            tuple(condition_ids),
            aggregate_sha256,
            DurableExecutionAccounting(
                planned_logical_generations=240,
                dispatched_logical_generations=240,
                successful_responses=240,
                uncertain_dispatches=0,
                committed_logical_generations=240,
            ),
        )
        store.verify_integrity()
    return run_directory


@pytest.fixture(scope="module")
def pilot_stores(tmp_path_factory: pytest.TempPathFactory) -> _PilotStores:
    root = tmp_path_factory.mktemp("pilot-static-selection")
    provider_identity, effective_configuration_json = _provider_binding(root)
    winner, conservative, exploratory = MODEL_BACKED_STATIC_CANDIDATE_PROFILES
    candidate_grid_path = root / "candidate-grid.json"
    candidate_grid_path.write_bytes(canonical_json_bytes(_PILOT_GRID))
    tampered = winner.model_copy(
        update={"profile_id": "static-trace-tampered-v1", "temperature": 0.8}
    )
    return _PilotStores(
        candidate_directories=(
            _write_pilot_store(
                root,
                profile=winner,
                task_passes=True,
                provider_identity=provider_identity,
                provider_effective_configuration_json=effective_configuration_json,
            ),
            _write_pilot_store(
                root,
                profile=conservative,
                task_passes=False,
                provider_identity=provider_identity,
                provider_effective_configuration_json=effective_configuration_json,
            ),
            _write_pilot_store(
                root,
                profile=exploratory,
                task_passes=False,
                provider_identity=provider_identity,
                provider_effective_configuration_json=effective_configuration_json,
            ),
        ),
        candidate_grid_path=candidate_grid_path,
        trace_tampered_directory=_write_pilot_store(
            root,
            profile=tampered,
            task_passes=False,
            provider_identity=provider_identity,
            provider_effective_configuration_json=effective_configuration_json,
            tamper_static_trace=True,
        ),
        profile_drift_directory=_write_pilot_store(
            root,
            profile=winner.model_copy(
                update={"profile_id": "static-substituted-v1", "temperature": 0.65}
            ),
            task_passes=False,
            provider_identity=provider_identity,
            provider_effective_configuration_json=effective_configuration_json,
        ),
        provider_identity=provider_identity,
        provider_effective_configuration_json=effective_configuration_json,
    )


@pytest.fixture(scope="module")
def in_memory_selection_evidence(
    pilot_stores: _PilotStores,
) -> DevelopmentPilotStaticSelectionEvidence:
    return build_test_static_selection_evidence(
        development_dataset=_PILOT_DATASET,
        winning_profile=MODEL_BACKED_STATIC_CANDIDATE_PROFILES[0],
        provider_identity=pilot_stores.provider_identity,
        provider_effective_configuration_json=(pilot_stores.provider_effective_configuration_json),
    )


def _replace_candidate(
    evidence: DevelopmentPilotStaticSelectionEvidence,
    candidate_index: int,
    replacement: object,
) -> DevelopmentPilotStaticSelectionEvidence:
    candidates = list(evidence.candidates)
    candidates[candidate_index] = replacement  # type: ignore[assignment]
    return evidence.model_copy(update={"candidates": tuple(candidates)})


def _replace_first_turn(
    evidence: DevelopmentPilotStaticSelectionEvidence,
    replacement: object,
) -> DevelopmentPilotStaticSelectionEvidence:
    candidate = evidence.candidates[0]
    updated = candidate.model_copy(update={"turns": (replacement, *candidate.turns[1:])})
    return _replace_candidate(evidence, 0, updated)


def test_freeze_static_selection_from_complete_declared_pilot_grid(
    tmp_path: Path,
    pilot_stores: _PilotStores,
) -> None:
    output = tmp_path / "evidence" / "development" / "model-backed-static-selection.json"
    publication = freeze_static_selection(
        pilot_stores.candidate_directories,
        pilot_stores.candidate_grid_path,
        output,
    )

    assert publication.created is True
    assert publication.output_path == output.resolve()
    assert len(publication.evidence.candidates) == 3
    assert publication.evidence.candidate_grid == _PILOT_GRID
    assert publication.evidence.selection_record.winning_profile.profile_id == "static-balanced-v1"
    assert all(len(candidate.turns) == 48 for candidate in publication.evidence.candidates)
    assert all(
        len(candidate.development_unit_keys) == 12 for candidate in publication.evidence.candidates
    )
    assert all(
        turn.generation_metadata.generation_method == "llama_cpp_completion_http_v1"
        for candidate in publication.evidence.candidates
        for turn in candidate.turns
    )
    assert output.read_bytes() == canonical_json_bytes(publication.evidence)
    assert load_static_selection_evidence(output) == publication.evidence
    validate_static_selection_evidence_against_dataset(
        publication.evidence,
        _PILOT_DATASET,
    )

    repeated = freeze_static_selection(
        pilot_stores.candidate_directories,
        pilot_stores.candidate_grid_path,
        output,
    )
    assert repeated.created is False
    assert repeated.evidence == publication.evidence


def test_freeze_static_selection_cli_is_provider_free(
    tmp_path: Path,
    pilot_stores: _PilotStores,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "selection.json"
    exit_code = main(
        [
            "freeze-static-selection",
            "--candidate-run-dir",
            str(pilot_stores.candidate_directories[0]),
            "--candidate-run-dir",
            str(pilot_stores.candidate_directories[1]),
            "--candidate-run-dir",
            str(pilot_stores.candidate_directories[2]),
            "--candidate-grid",
            str(pilot_stores.candidate_grid_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    evidence = load_static_selection_evidence(output)
    assert payload["command"] == "freeze-static-selection"
    assert payload["candidate_count"] == 3
    assert payload["candidate_grid_sha256"] == _PILOT_GRID.candidate_grid_sha256
    assert payload["winning_profile"]["profile_id"] == "static-balanced-v1"
    assert payload["static_selection_evidence_sha256"] == evidence.evidence_sha256
    assert payload["source_run_directories"] == [
        str(path.resolve()) for path in pilot_stores.candidate_directories
    ]
    assert payload["provider_constructed"] is False
    assert payload["network_requested"] is False


def test_static_selection_rejects_trace_cross_object_tampering(
    pilot_stores: _PilotStores,
) -> None:
    with pytest.raises(StoreInvariantError, match="stateless and exactly zero"):
        build_static_selection_evidence(
            (
                pilot_stores.candidate_directories[0],
                pilot_stores.candidate_directories[1],
                pilot_stores.trace_tampered_directory,
            ),
            _PILOT_GRID,
        )


def test_static_selection_rejects_artifact_tampering(
    tmp_path: Path,
    pilot_stores: _PilotStores,
) -> None:
    output = tmp_path / "selection.json"
    freeze_static_selection(
        pilot_stores.candidate_directories,
        pilot_stores.candidate_grid_path,
        output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["candidates"][0]["unit_scores"][0] = 0.123
    tampered = tmp_path / "tampered-selection.json"
    tampered.write_text(canonical_json(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scores do not match|analysis hash"):
        load_static_selection_evidence(tampered)


def test_static_selection_rejects_missing_extra_and_substituted_profiles(
    pilot_stores: _PilotStores,
) -> None:
    with pytest.raises(ValueError, match="one run for every declared grid profile"):
        build_static_selection_evidence(pilot_stores.candidate_directories[:2], _PILOT_GRID)
    with pytest.raises(ValueError, match="one run for every declared grid profile"):
        build_static_selection_evidence(
            (*pilot_stores.candidate_directories, pilot_stores.profile_drift_directory),
            _PILOT_GRID,
        )
    with pytest.raises(ValueError, match="differ from the predeclared candidate grid"):
        build_static_selection_evidence(
            (
                pilot_stores.candidate_directories[0],
                pilot_stores.candidate_directories[1],
                pilot_stores.profile_drift_directory,
            ),
            _PILOT_GRID,
        )


def test_candidate_grid_rejects_profile_and_byte_tampering(tmp_path: Path) -> None:
    payload = _PILOT_GRID.model_dump(mode="json")
    payload["candidate_profiles"][0]["max_tokens"] = 256
    tampered_profile = tmp_path / "tampered-profile-grid.json"
    tampered_profile.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact sorted Phase 3 static profile grid"):
        load_development_pilot_candidate_grid(tampered_profile)

    noncanonical = tmp_path / "noncanonical-grid.json"
    noncanonical.write_text(json.dumps(_PILOT_GRID.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="exact canonical JSON bytes"):
        load_development_pilot_candidate_grid(noncanonical)


def test_candidate_grid_rejects_invalid_inputs_and_expected_identity(tmp_path: Path) -> None:
    blank_version = _PILOT_GRID.model_dump(mode="json")
    blank_version["dataset_version"] = "   "
    with pytest.raises(ValueError, match="dataset version must not be blank"):
        DevelopmentPilotCandidateGrid.model_validate(blank_version)

    candidate_grid_path = tmp_path / "candidate-grid.json"
    candidate_grid_path.write_bytes(canonical_json_bytes(_PILOT_GRID))
    with pytest.raises(TypeError, match="path must be a pathlib.Path"):
        load_development_pilot_candidate_grid(str(candidate_grid_path))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lowercase SHA-256 hex digest"):
        load_development_pilot_candidate_grid(candidate_grid_path, expected_sha256="invalid")
    with pytest.raises(ValueError, match="differs from its expected SHA-256"):
        load_development_pilot_candidate_grid(candidate_grid_path, expected_sha256="f" * 64)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("dataset_purpose", "requires a development-purpose dataset"),
        ("manifest_dataset_hash", "does not match the development dataset hash"),
        ("grid_dataset", "candidate grid does not match the development dataset"),
        ("turn_dataset_version", "does not match the development dataset version"),
        ("prompt_coordinates", "does not cover the exact development prompt grid"),
        ("turn_input", "turn input differs from the development prompt case"),
        ("response_sha256", "wire evidence differs from the development request/response"),
    ),
)
def test_static_selection_dataset_binding_rejects_each_cross_object_drift(
    in_memory_selection_evidence: DevelopmentPilotStaticSelectionEvidence,
    tamper: str,
    message: str,
) -> None:
    evidence = in_memory_selection_evidence
    dataset = _PILOT_DATASET
    if tamper == "dataset_purpose":
        dataset = dataset.model_copy(update={"purpose": DatasetPurpose.SYNTHETIC})
    elif tamper == "manifest_dataset_hash":
        candidate = evidence.candidates[0]
        manifest = candidate.source_run_manifest.model_copy(update={"dataset_hash": "f" * 64})
        evidence = _replace_candidate(
            evidence,
            0,
            candidate.model_copy(update={"source_run_manifest": manifest}),
        )
    elif tamper == "grid_dataset":
        evidence = evidence.model_copy(
            update={
                "candidate_grid": evidence.candidate_grid.model_copy(
                    update={"dataset_version": "another-development-dataset"}
                )
            }
        )
    elif tamper == "turn_dataset_version":
        turn = evidence.candidates[0].turns[0]
        condition = turn.condition.model_copy(update={"dataset_version": "another-version"})
        evidence = _replace_first_turn(evidence, turn.model_copy(update={"condition": condition}))
    elif tamper == "prompt_coordinates":
        candidate = evidence.candidates[0]
        omitted = candidate.turns[0].condition
        evidence = _replace_candidate(
            evidence,
            0,
            candidate.model_copy(
                update={
                    "turns": tuple(
                        turn
                        for turn in candidate.turns
                        if (
                            turn.condition.prompt_sequence_id,
                            turn.condition.turn_index,
                        )
                        != (omitted.prompt_sequence_id, omitted.turn_index)
                    )
                }
            ),
        )
    elif tamper == "turn_input":
        turn = evidence.candidates[0].turns[0]
        turn_input = turn.turn_input.model_copy(update={"prompt_case_id": "foreign-case"})
        evidence = _replace_first_turn(evidence, turn.model_copy(update={"turn_input": turn_input}))
    else:
        turn = evidence.candidates[0].turns[0]
        evidence = _replace_first_turn(
            evidence,
            turn.model_copy(update={"response_sha256": "f" * 64}),
        )

    with pytest.raises(ValueError, match=message):
        validate_static_selection_evidence_against_dataset(evidence, dataset)


@pytest.mark.parametrize(
    ("policy_trace_json", "message"),
    (
        (None, "lacks policy trace evidence"),
        ("{", "not valid JSON"),
        ("{}", "fails its declared schema"),
        ("pretty", "not canonical JSON"),
    ),
)
def test_static_trace_rejects_missing_malformed_and_noncanonical_evidence(
    pilot_stores: _PilotStores,
    policy_trace_json: str | None,
    message: str,
) -> None:
    with SQLiteRunStore(pilot_stores.candidate_directories[0] / "run.sqlite3") as store:
        turn = next(
            item for item in store.list_turns() if item.condition.policy_id == "best_static"
        )
    if policy_trace_json == "pretty":
        assert turn.policy_trace_json is not None
        policy_trace_json = json.dumps(json.loads(turn.policy_trace_json), indent=2, sort_keys=True)
    with pytest.raises(StoreInvariantError, match=message):
        _require_static_trace(replace(turn, policy_trace_json=policy_trace_json))


@pytest.mark.parametrize(
    ("scenario", "message"),
    (
        ("missing_finalization", "manifest-bound finalized run"),
        ("turn_count", "not an exact closed pilot"),
        ("finalization_hash", "finalization does not match"),
        ("missing_input", "lacks prompt-side input evidence"),
        ("metric_drift", "metrics do not reconstruct exactly"),
        ("response_binding", "response differs from its provider binding"),
    ),
)
def test_static_selection_source_run_guards(
    pilot_stores: _PilotStores,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    message: str,
) -> None:
    run_directory = pilot_stores.candidate_directories[0]
    with SQLiteRunStore(run_directory / "run.sqlite3") as source:
        manifest = source.get_manifest()
        finalization = source.get_finalization()
        turns = source.list_turns()
        input_evidence = {
            turn.condition_id: source.get_turn_input(turn.condition_id)
            for turn in turns
            if turn.condition.policy_id == "best_static"
        }
    assert manifest is not None
    assert finalization is not None
    missing_input_condition_id: str | None = None
    selected_finalization = finalization
    selected_turns = turns

    if scenario == "missing_finalization":
        selected_finalization = None
    elif scenario == "turn_count":
        selected_turns = turns[:-1]
    elif scenario == "finalization_hash":
        selected_finalization = finalization.model_copy(
            update={"scientific_result_sha256": "f" * 64}
        )
    elif scenario == "missing_input":
        missing_input_condition_id = next(
            turn.condition_id for turn in turns if turn.condition.policy_id == "best_static"
        )
    elif scenario == "metric_drift":
        changed = next(turn for turn in turns if turn.condition.policy_id == "best_static")
        assert changed.metrics is not None
        replacement = replace(
            changed,
            metrics=changed.metrics.model_copy(
                update={
                    "task_score": changed.metrics.task_score.model_copy(
                        update={"value": 0.0 if changed.metrics.task_score.value == 1.0 else 1.0}
                    )
                }
            ),
        )
        selected_turns = tuple(replacement if turn is changed else turn for turn in turns)
        selected_finalization = finalization.model_copy(
            update={"scientific_result_sha256": scientific_result_sha256(selected_turns)}
        )
    else:
        changed = next(turn for turn in turns if turn.condition.policy_id == "best_static")
        assert changed.response is not None
        effective = changed.response.effective_parameters.model_copy(
            update={"top_k": changed.response.effective_parameters.top_k + 1}
        )
        replacement = replace(
            changed,
            response=changed.response.model_copy(update={"effective_parameters": effective}),
        )
        selected_turns = tuple(replacement if turn is changed else turn for turn in turns)
        selected_finalization = finalization.model_copy(
            update={"scientific_result_sha256": scientific_result_sha256(selected_turns)}
        )

    class StubStore:
        def __init__(self, _database_path: Path) -> None:
            pass

        def __enter__(self) -> StubStore:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def verify_integrity(self) -> None:
            return None

        def get_manifest(self) -> object:
            return manifest

        def get_finalization(self) -> object:
            return selected_finalization

        def list_turns(self) -> object:
            return selected_turns

        def get_turn_input(self, condition_id: str) -> object:
            if condition_id == missing_input_condition_id:
                return None
            return input_evidence[condition_id]

    monkeypatch.setattr(static_selection, "SQLiteRunStore", StubStore)
    with pytest.raises(StoreInvariantError, match=message):
        _candidate_from_run_directory(run_directory)


def test_static_selection_public_type_and_path_guards(
    tmp_path: Path,
    pilot_stores: _PilotStores,
    in_memory_selection_evidence: DevelopmentPilotStaticSelectionEvidence,
) -> None:
    with pytest.raises(TypeError, match="evidence must be"):
        validate_static_selection_evidence_against_dataset(object(), _PILOT_DATASET)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dataset must be"):
        validate_static_selection_evidence_against_dataset(
            in_memory_selection_evidence,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="tuple of pathlib.Path"):
        build_static_selection_evidence(
            list(pilot_stores.candidate_directories),  # type: ignore[arg-type]
            _PILOT_GRID,
        )
    with pytest.raises(TypeError, match="DevelopmentPilotCandidateGrid"):
        build_static_selection_evidence((), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="path must be a pathlib.Path"):
        load_static_selection_evidence("selection.json")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="output_path must be a pathlib.Path"):
        freeze_static_selection((), tmp_path / "grid.json", "selection.json")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidate run directories must be unique"):
        build_static_selection_evidence(
            (
                pilot_stores.candidate_directories[0],
                pilot_stores.candidate_directories[1],
                pilot_stores.candidate_directories[1],
            ),
            _PILOT_GRID,
        )

    candidate_file = tmp_path / "candidate-file"
    candidate_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate run path is not a directory"):
        _candidate_from_run_directory(candidate_file)
    directory = tmp_path / "candidate-directory"
    (directory / "run.sqlite3").mkdir(parents=True)
    with pytest.raises(ValueError, match="candidate run database is not a file"):
        _candidate_from_run_directory(directory)
    with pytest.raises(StoreInvariantError, match="contains no best_static turns"):
        _profile_from_static_turns(())


def test_static_selection_canonical_load_and_immutable_publication(
    tmp_path: Path,
    in_memory_selection_evidence: DevelopmentPilotStaticSelectionEvidence,
) -> None:
    noncanonical = tmp_path / "noncanonical-selection.json"
    noncanonical.write_bytes(canonical_json_bytes(in_memory_selection_evidence) + b"\n")
    with pytest.raises(ValueError, match="exact canonical JSON bytes"):
        load_static_selection_evidence(noncanonical)

    occupied = tmp_path / "occupied-selection.json"
    occupied.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish_canonical_json(occupied, b"expected")


@pytest.mark.parametrize("same_payload", (True, False))
def test_static_selection_publication_handles_hard_link_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_payload: bool,
) -> None:
    output = tmp_path / f"race-{same_payload}.json"
    payload = b"canonical evidence"

    def race_link(_source: object, destination: object) -> None:
        Path(destination).write_bytes(payload if same_payload else b"conflict")
        raise FileExistsError

    monkeypatch.setattr(static_selection.os, "link", race_link)
    if same_payload:
        assert _publish_canonical_json(output, payload) is False
    else:
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            _publish_canonical_json(output, payload)


def test_freeze_static_selection_rejects_reload_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_memory_selection_evidence: DevelopmentPilotStaticSelectionEvidence,
) -> None:
    drifted = in_memory_selection_evidence.model_copy(
        update={"candidates": tuple(reversed(in_memory_selection_evidence.candidates))}
    )
    monkeypatch.setattr(
        static_selection,
        "load_development_pilot_candidate_grid",
        lambda _path: _PILOT_GRID,
    )
    monkeypatch.setattr(
        static_selection,
        "build_static_selection_evidence",
        lambda _directories, _grid: in_memory_selection_evidence,
    )
    monkeypatch.setattr(static_selection, "_publish_canonical_json", lambda _path, _raw: True)
    monkeypatch.setattr(static_selection, "load_static_selection_evidence", lambda _path: drifted)

    with pytest.raises(RuntimeError, match="changed during reload"):
        freeze_static_selection((), tmp_path / "grid.json", tmp_path / "selection.json")


def test_static_selection_rejects_cross_candidate_prompt_input_drift(
    pilot_stores: _PilotStores,
) -> None:
    evidence = build_test_static_selection_evidence(
        development_dataset=_PILOT_DATASET,
        winning_profile=MODEL_BACKED_STATIC_CANDIDATE_PROFILES[0],
        provider_identity=pilot_stores.provider_identity,
        provider_effective_configuration_json=(pilot_stores.provider_effective_configuration_json),
    )
    drifted_source = evidence.candidates[1]
    first_turn = drifted_source.turns[0]
    drifted_turn = first_turn.model_copy(
        update={
            "turn_input": first_turn.turn_input.model_copy(
                update={"prompt_family": "foreign_prompt_family"}
            )
        }
    )
    drifted_candidate = build_development_pilot_candidate_evidence(
        source_run_manifest=drifted_source.source_run_manifest,
        source_run_finalization=drifted_source.source_run_finalization,
        profile=drifted_source.profile,
        turns=(drifted_turn, *drifted_source.turns[1:]),
    )

    with pytest.raises(ValueError, match="exact prompt-side inputs"):
        candidates = list(evidence.candidates)
        candidates[1] = drifted_candidate
        build_development_pilot_static_selection_evidence(
            tuple(candidates),
            evidence.candidate_grid,
        )


@pytest.mark.parametrize(
    "tamper",
    ("request_prompt", "request_parameters", "response_content", "response_settings"),
)
def test_static_selection_evidence_rejects_wire_domain_tampering(
    pilot_stores: _PilotStores,
    tamper: str,
) -> None:
    evidence = build_test_static_selection_evidence(
        development_dataset=_PILOT_DATASET,
        winning_profile=MODEL_BACKED_STATIC_CANDIDATE_PROFILES[0],
        provider_identity=pilot_stores.provider_identity,
        provider_effective_configuration_json=(pilot_stores.provider_effective_configuration_json),
    )
    first_turn = evidence.candidates[0].turns[0]
    metadata = first_turn.generation_metadata
    assert metadata.provider_request_json is not None
    assert metadata.provider_response_json is not None
    request_payload = json.loads(metadata.provider_request_json)
    response_payload = json.loads(metadata.provider_response_json)
    if tamper == "request_prompt":
        request_payload["prompt"] = "foreign prompt"
    elif tamper == "request_parameters":
        request_payload["temperature"] = first_turn.decoding_parameters.temperature + 0.1
    elif tamper == "response_content":
        response_payload["content"] = "foreign response"
    else:
        response_payload["generation_settings"]["top_k"] = first_turn.decoding_parameters.top_k + 1
    tampered_metadata = metadata.model_copy(
        update={
            "provider_request_json": canonical_json(request_payload),
            "provider_request_sha256": canonical_sha256(request_payload),
            "provider_response_json": canonical_json(response_payload),
            "provider_response_sha256": canonical_sha256(response_payload),
        }
    )
    payload = evidence.model_dump(mode="python")
    payload["candidates"][0]["turns"][0]["generation_metadata"] = tampered_metadata
    with pytest.raises(ValueError, match="llama.cpp|wire evidence"):
        DevelopmentPilotStaticSelectionEvidence.model_validate(payload)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("policy", "accepts best_static turns only"),
        ("turn_input", "turn input targets another condition"),
        ("seed", "request seed differs from its model seed"),
        ("protocol", "requires request-bound llama_cpp protocol evidence"),
        ("task_score", "requires every task score"),
    ),
)
def test_pilot_turn_evidence_rejects_each_broken_binding(
    in_memory_selection_evidence: DevelopmentPilotStaticSelectionEvidence,
    tamper: str,
    message: str,
) -> None:
    turn = in_memory_selection_evidence.candidates[0].turns[0]
    payload = turn.model_dump(mode="python")
    if tamper == "policy":
        payload["condition"] = turn.condition.model_copy(update={"policy_id": "foreign-policy"})
    elif tamper == "turn_input":
        payload["turn_input"] = turn.turn_input.model_copy(update={"condition_id": "f" * 64})
    elif tamper == "seed":
        payload["decoding_parameters"] = turn.decoding_parameters.model_copy(
            update={"seed": turn.decoding_parameters.seed + 1}
        )
    elif tamper == "protocol":
        payload["generation_metadata"] = turn.generation_metadata.model_copy(
            update={"request_sha256": "f" * 64}
        )
    else:
        payload["task_score"] = turn.task_score.model_copy(
            update={"availability": False, "value": None}
        )
    with pytest.raises(ValueError, match=message):
        DevelopmentPilotTurnEvidence.model_validate(payload)

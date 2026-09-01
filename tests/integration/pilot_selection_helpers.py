"""Typed model-backed pilot evidence builders shared by handoff tests."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from neurallm.control.specs import (
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    PolicySpec,
    RandomMatchedPolicySpec,
)
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
    ProviderIdentity,
    RunManifest,
    SeedSchedule,
    UnitIntervalMetricValue,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.evaluation.pilot_grid import (
    MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
    DevelopmentPilotCandidateGrid,
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
from neurallm.experiments.dataset import PromptCase, PromptDataset
from neurallm.experiments.protocol import MODEL_BACKED_POLICY_IDS
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.base import GenerationMetadata, GenerationRequest, GenerationResponse
from neurallm.providers.llama_cpp import (
    LLAMA_CPP_IMPLEMENTATION_VERSION,
    LlamaCppEffectiveConfiguration,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)
from neurallm.storage import CURRENT_SCHEMA_VERSION
from neurallm.storage.models import (
    DurableExecutionAccounting,
    RunFinalization,
    TurnInputEvidence,
)

_MODEL_SEEDS = (1101, 1102)
_CONTROLLER_SEED = 5101
_DEFAULT_SEQUENCE_IDS = (
    "pilot-sequence-01",
    "pilot-sequence-02",
    "pilot-sequence-03",
    "pilot-sequence-04",
    "pilot-sequence-05",
    "pilot-sequence-06",
)
_DEFAULT_POLICY_SPECS: tuple[PolicySpec, ...] = (
    BestStaticPolicySpec(),
    HeuristicAdaptivePolicySpec(),
    NeuralMatchedHistoryStateResetPolicySpec(),
    NeuralPersistentPolicySpec(),
    RandomMatchedPolicySpec(),
)


def _provider_identity() -> tuple[ProviderIdentity, str]:
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    provider_config = LlamaCppProviderConfig(
        base_url="http://127.0.0.1:8080",
        model_alias="pilot-selection-fixture",
        model_path=str((Path.cwd() / "fixtures" / "pilot-selection.gguf").resolve()),
        model_sha256="a" * 64,
        build_id="pilot-selection-fixture-build",
        chat_template_sha256=sha256(chat_template.encode("utf-8")).hexdigest(),
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        write_timeout_seconds=3.0,
        pool_timeout_seconds=4.0,
    )
    effective = LlamaCppEffectiveConfiguration(
        client_config=provider_config,
        model_alias=provider_config.model_alias,
        model_path=provider_config.model_path,
        model_sha256=provider_config.model_sha256,
        build_id=provider_config.build_id,
        chat_template=chat_template,
        chat_template_sha256=provider_config.chat_template_sha256,
        default_generation_settings_json=canonical_json(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "presence_penalty": 0.0,
                "n_predict": 192,
                "seed": 1101,
            }
        ),
        total_slots=1,
    )
    identity = llama_cpp_provider_identity(effective)
    assert identity.implementation_version == LLAMA_CPP_IMPLEMENTATION_VERSION
    return identity, canonical_json(effective)


def _candidate(
    *,
    dataset_version: str,
    dataset_sha256: str,
    sequence_ids: tuple[str, ...],
    profile: StaticProfile,
    task_score: float,
    provider_identity: ProviderIdentity,
    provider_effective_configuration_json: str,
    policy_config_hashes: Mapping[str, str],
    matched_history_policy_sources: Mapping[str, str],
    action_bounds: ActionBounds,
    decoding_bounds: DecodingBounds,
    metric_versions: Mapping[str, str],
    database_schema_version: int,
    prompt_cases: Mapping[tuple[str, int], PromptCase],
    candidate_grid_sha256: str,
) -> DevelopmentPilotCandidateEvidence:
    manifest = RunManifest(
        source_commit="1" * 40,
        working_tree_clean=True,
        experiment_config_hash=canonical_sha256(
            {"pilot_profile": profile.profile_id, "kind": "experiment-config"}
        ),
        dataset_hash=dataset_sha256,
        provider_config_hash=provider_identity.provider_config_hash,
        provider_identity=provider_identity,
        provider_effective_configuration_json=provider_effective_configuration_json,
        policy_config_hashes=policy_config_hashes,
        matched_history_policy_sources=matched_history_policy_sources,
        metric_versions=metric_versions,
        seed_schedule=SeedSchedule(
            model_seeds=_MODEL_SEEDS,
            controller_seeds=(_CONTROLLER_SEED,),
        ),
        action_bounds=action_bounds,
        decoding_bounds=decoding_bounds,
        decision_rule_version="development-pilot-no-scientific-decision-v1",
        database_schema_version=database_schema_version,
        run_tier="development_pilot",
        scientific_identity_sha256=canonical_sha256(
            {"pilot_profile": profile.profile_id, "kind": "scientific-identity"}
        ),
        candidate_grid_sha256=candidate_grid_sha256,
    )
    turns: list[DevelopmentPilotTurnEvidence] = []
    for sequence_id in sequence_ids:
        for model_seed in _MODEL_SEEDS:
            for turn_index in range(4):
                condition = ExperimentCondition(
                    experiment_id=f"pilot-{profile.profile_id}",
                    dataset_version=dataset_version,
                    prompt_sequence_id=sequence_id,
                    turn_index=turn_index,
                    policy_id="best_static",
                    model_seed=model_seed,
                    controller_seed=_CONTROLLER_SEED,
                    provider_identity_id=provider_identity.identity_id,
                    base_decoding_profile_id=profile.profile_id,
                )
                prompt_case = prompt_cases.get((sequence_id, turn_index))
                parameters = DecodingParameters(
                    temperature=profile.temperature,
                    top_p=profile.top_p,
                    top_k=profile.top_k,
                    presence_penalty=profile.presence_penalty,
                    max_tokens=profile.max_tokens,
                    seed=model_seed,
                )
                prompt = (
                    f"Fixture prompt for {sequence_id} turn {turn_index}."
                    if prompt_case is None
                    else prompt_case.prompt
                )
                request = GenerationRequest(
                    prompt=prompt,
                    decoding_parameters=parameters,
                    condition=condition,
                )
                request_sha256 = canonical_sha256(request)
                provider_request = {
                    "prompt": prompt,
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
                    "content": "fixture response",
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
                metadata = GenerationMetadata(
                    request_sha256=request_sha256,
                    generation_method="llama_cpp_completion_http_v1",
                    provider_request_json=canonical_json(provider_request),
                    provider_request_sha256=canonical_sha256(provider_request),
                    provider_response_json=canonical_json(provider_response),
                    provider_response_sha256=canonical_sha256(provider_response),
                )
                response = GenerationResponse(
                    text="fixture response",
                    provider_identity=provider_identity,
                    effective_parameters=parameters,
                    raw_metadata=metadata,
                )
                turns.append(
                    DevelopmentPilotTurnEvidence(
                        condition=condition,
                        request_sha256=request_sha256,
                        response_sha256=canonical_sha256(response),
                        generation_metadata=metadata,
                        decoding_parameters=parameters,
                        turn_input=TurnInputEvidence(
                            condition_id=condition.condition_id,
                            prompt_case_id=(
                                f"{sequence_id}-turn-{turn_index}"
                                if prompt_case is None
                                else prompt_case.case_id
                            ),
                            prompt_family=(
                                "pilot_selection_fixture"
                                if prompt_case is None
                                else prompt_case.prompt_family
                            ),
                            prompt_features=(
                                PromptFeatures({"constraint_count": 1.0})
                                if prompt_case is None
                                else prompt_case.prompt_features
                            ),
                            validator=(
                                ValidatorSpec(kind="non_empty")
                                if prompt_case is None
                                else prompt_case.validator
                            ),
                        ),
                        task_score=UnitIntervalMetricValue(
                            value=task_score,
                            availability=True,
                            metric_version="validator-v1",
                            input_hash=canonical_sha256(
                                {"condition_id": condition.condition_id, "kind": "task-score"}
                            ),
                        ),
                    )
                )

    expected_condition_ids = {turn.condition.condition_id for turn in turns}
    extra_index = 0
    while len(expected_condition_ids) < 240:
        expected_condition_ids.add(
            canonical_sha256(
                {
                    "pilot_profile": profile.profile_id,
                    "extra_condition_index": extra_index,
                }
            )
        )
        extra_index += 1
    finalization = RunFinalization(
        expected_condition_ids=tuple(sorted(expected_condition_ids)),
        expected_condition_count=240,
        manifest_sha256=canonical_sha256(manifest),
        scientific_result_sha256=canonical_sha256(
            {"pilot_profile": profile.profile_id, "kind": "scientific-result"}
        ),
        execution_accounting=DurableExecutionAccounting(
            planned_logical_generations=240,
            dispatched_logical_generations=240,
            successful_responses=240,
            uncertain_dispatches=0,
            committed_logical_generations=240,
        ),
    )
    return build_development_pilot_candidate_evidence(
        source_run_manifest=manifest,
        source_run_finalization=finalization,
        profile=profile,
        turns=tuple(turns),
    )


def build_test_static_selection_evidence(
    *,
    winning_profile: StaticProfile,
    development_dataset: PromptDataset | None = None,
    dataset_version: str | None = None,
    dataset_sha256: str | None = None,
    sequence_ids: tuple[str, ...] | None = None,
    provider_identity: ProviderIdentity | None = None,
    provider_effective_configuration_json: str | None = None,
    policy_specs: tuple[PolicySpec, ...] = _DEFAULT_POLICY_SPECS,
    action_bounds: ActionBounds | None = None,
    decoding_bounds: DecodingBounds | None = None,
    metric_versions: Mapping[str, str] | None = None,
    database_schema_version: int = CURRENT_SCHEMA_VERSION,
) -> DevelopmentPilotStaticSelectionEvidence:
    """Build valid multi-run evidence for an explicitly supplied winning profile."""

    prompt_cases: dict[tuple[str, int], PromptCase] = {}
    if development_dataset is not None:
        if dataset_version not in (None, development_dataset.version):
            raise ValueError("explicit dataset version differs from development_dataset")
        if dataset_sha256 not in (None, development_dataset.dataset_hash):
            raise ValueError("explicit dataset hash differs from development_dataset")
        declared_sequence_ids = tuple(
            sequence.sequence_id for sequence in development_dataset.sequences
        )
        if sequence_ids not in (None, declared_sequence_ids):
            raise ValueError("explicit sequence IDs differ from development_dataset")
        dataset_version = development_dataset.version
        dataset_sha256 = development_dataset.dataset_hash
        sequence_ids = declared_sequence_ids
        prompt_cases = {
            (sequence.sequence_id, turn_index): prompt_case
            for sequence in development_dataset.sequences
            for turn_index, prompt_case in enumerate(sequence.cases)
        }
    else:
        sequence_ids = sequence_ids or _DEFAULT_SEQUENCE_IDS
    if dataset_version is None or dataset_sha256 is None:
        raise ValueError("test pilot selection requires an explicit development dataset")
    if len(sequence_ids) != 6 or len(set(sequence_ids)) != 6:
        raise ValueError("test pilot selection requires six unique prompt sequences")
    if (provider_identity is None) != (provider_effective_configuration_json is None):
        raise ValueError("test pilot provider identity and effective config must appear together")
    if provider_identity is None:
        provider_identity, provider_effective_configuration_json = _provider_identity()
    assert provider_effective_configuration_json is not None
    resolved_action_bounds = action_bounds or ActionBounds()
    resolved_decoding_bounds = decoding_bounds or DecodingBounds()
    resolved_metric_versions = metric_versions or METRIC_VERSIONS
    policy_config_hashes = {spec.policy_id: canonical_sha256(spec) for spec in policy_specs}
    if set(policy_config_hashes) != set(MODEL_BACKED_POLICY_IDS):
        raise ValueError("test pilot selection requires the exact model-backed policies")
    matched_history_policy_sources = {
        spec.policy_id: history_source_policy_id
        for spec in policy_specs
        if (history_source_policy_id := getattr(spec, "history_source_policy_id", None)) is not None
    }
    if winning_profile not in MODEL_BACKED_STATIC_CANDIDATE_PROFILES:
        raise ValueError("winning_profile must be one exact predeclared pilot profile")
    candidate_grid = DevelopmentPilotCandidateGrid(
        dataset_version=dataset_version,
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=dataset_sha256,
        candidate_profiles=MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
    )
    return build_development_pilot_static_selection_evidence(
        tuple(
            _candidate(
                dataset_version=dataset_version,
                dataset_sha256=dataset_sha256,
                sequence_ids=sequence_ids,
                profile=profile,
                task_score=0.8 if profile == winning_profile else 0.6,
                provider_identity=provider_identity,
                provider_effective_configuration_json=(provider_effective_configuration_json),
                policy_config_hashes=policy_config_hashes,
                matched_history_policy_sources=matched_history_policy_sources,
                action_bounds=resolved_action_bounds,
                decoding_bounds=resolved_decoding_bounds,
                metric_versions=resolved_metric_versions,
                database_schema_version=database_schema_version,
                prompt_cases=prompt_cases,
                candidate_grid_sha256=candidate_grid.candidate_grid_sha256,
            )
            for profile in MODEL_BACKED_STATIC_CANDIDATE_PROFILES
        ),
        candidate_grid,
    )


__all__ = ["build_test_static_selection_evidence"]

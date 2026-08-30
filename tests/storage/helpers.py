"""Typed builders shared by storage contract tests."""

from __future__ import annotations

from neurallm.control.policy import PolicyState, PolicyTrace
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    CountMetricValue,
    DecodingParameters,
    ExperimentCondition,
    ProviderIdentity,
    ResponseMetrics,
    RunManifest,
    SeedSchedule,
    UnitIntervalMetricValue,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.providers.base import GenerationRequest, GenerationResponse
from neurallm.providers.fake import FakeProvider, fake_provider_effective_configuration_json
from neurallm.storage import CommittedHistory, HistoryBinding, SQLiteRunStore


def make_manifest(identity: ProviderIdentity) -> RunManifest:
    """Return a complete manifest bound to the current storage schema."""

    return RunManifest(
        source_commit="0" * 40,
        working_tree_clean=True,
        experiment_config_hash=canonical_sha256("experiment-config"),
        dataset_hash=canonical_sha256("dataset"),
        provider_config_hash=identity.provider_config_hash,
        provider_identity=identity,
        provider_effective_configuration_json=fake_provider_effective_configuration_json(),
        policy_config_hashes={"test-policy": canonical_sha256("test-policy")},
        metric_versions={"test-metrics": "1.0.0"},
        seed_schedule=SeedSchedule(model_seeds=(7,), controller_seeds=(11,)),
        action_bounds=ActionBounds(),
        decision_rule_version="test-v1",
        database_schema_version=1,
    )


def make_request(
    identity: ProviderIdentity,
    *,
    turn_index: int = 0,
    prompt: str | None = None,
    prompt_sequence_id: str = "sequence-a",
    policy_id: str = "test-policy",
    model_seed: int = 7,
    controller_seed: int = 11,
) -> GenerationRequest:
    """Return one deterministic request with a fully bound condition."""

    condition = ExperimentCondition(
        experiment_id="experiment-a",
        dataset_version="dataset-v1",
        prompt_sequence_id=prompt_sequence_id,
        turn_index=turn_index,
        policy_id=policy_id,
        model_seed=model_seed,
        controller_seed=controller_seed,
        provider_identity_id=identity.identity_id,
        base_decoding_profile_id="base-a",
    )
    return GenerationRequest(
        prompt=prompt or f"prompt for turn {turn_index}",
        decoding_parameters=DecodingParameters(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=128,
            seed=model_seed,
        ),
        condition=condition,
    )


def make_metrics(response: GenerationResponse) -> ResponseMetrics:
    """Return complete deterministic metrics bound to one response text."""

    input_hash = canonical_sha256(response.text)

    def unit(value: float) -> UnitIntervalMetricValue:
        return UnitIntervalMetricValue(
            value=value,
            availability=True,
            metric_version="test-v1",
            input_hash=input_hash,
        )

    return ResponseMetrics(
        task_score=unit(1.0),
        instruction_adherence=unit(1.0),
        response_length_tokens=CountMetricValue(
            value=8,
            availability=True,
            metric_version="test-v1",
            input_hash=input_hash,
        ),
        repetition_ratio=unit(0.0),
        repeated_3_gram_ratio=unit(0.0),
        repeated_4_gram_ratio=unit(0.0),
        distinct_2=unit(1.0),
        distinct_3=unit(1.0),
        late_window_repetition_ratio=unit(0.0),
        format_validity=unit(1.0),
        semantic_similarity=UnitIntervalMetricValue(
            value=None,
            availability=False,
            metric_version="test-v1",
            input_hash=input_hash,
        ),
    )


def make_trace(request: GenerationRequest) -> PolicyTrace:
    """Return an in-bounds trace aligned with a request condition."""

    return PolicyTrace(
        policy_id=request.condition.policy_id,
        turn_index=request.condition.turn_index,
        action=ControllerAction(
            temperature_delta=0.0,
            top_p_delta=0.0,
            top_k_delta=0,
            presence_penalty_delta=0.0,
        ),
    )


def complete_request(
    store: SQLiteRunStore,
    provider: FakeProvider,
    request: GenerationRequest,
    history: HistoryBinding | None = None,
    *,
    policy_state: PolicyState | None = None,
) -> CommittedHistory:
    """Drive one request through every durable checkpoint."""

    store.prepare_turn(request, history)
    store.begin_dispatch(request.condition_id)
    response = provider.generate(request)
    store.persist_response(request.condition_id, response)
    store.persist_metrics(request.condition_id, make_metrics(response))
    return store.commit_turn(
        request.condition_id,
        policy_state or PolicyState(),
        make_trace(request),
    )

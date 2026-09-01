from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurallm.control.specs import (
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    RandomMatchedPolicySpec,
)
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    PromptFeatures,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.attribution import AttributionAnalysisSpec
from neurallm.evaluation.confirmatory import ConfirmatoryAnalysisSpec, RecoveryEventSpec
from neurallm.evaluation.models import DatasetPurpose, EvaluationSpec, MatchedUnitKey
from neurallm.evaluation.pilot_grid import (
    MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
    DevelopmentPilotCandidateGrid,
)
from neurallm.evaluation.recovery import RecoveryAnalysisSpec, RecoveryMetricName
from neurallm.evaluation.scientific import EfficacyAnalysisSpec, LimitationDisposition
from neurallm.evaluation.selection import (
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    select_best_static,
)
from neurallm.experiments.config import (
    BaseDecodingProfile,
    DatasetReference,
    DevelopmentSelectionInput,
    ExperimentConfig,
    LoadedExperimentConfig,
    ProviderSelection,
    load_experiment_config,
)
from neurallm.experiments.dataset import (
    DatasetSeal,
    LoadedDataset,
    PromptCase,
    PromptDataset,
    PromptSequence,
)
from neurallm.experiments.plan import ExperimentPlan, build_plan
from neurallm.experiments.protocol import (
    ATTRIBUTION_HISTORY_SOURCE_POLICY_ID,
    ATTRIBUTION_POLICY_ID,
    EFFICACY_POLICY_IDS,
    MODEL_BACKED_POLICY_IDS,
    ExperimentProtocol,
    PreregistrationSeal,
    RunTier,
    ScheduleSpec,
)
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.fake import (
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION
from tests.integration.pilot_selection_helpers import build_test_static_selection_evidence


def _llama_provider_selection() -> ProviderSelection:
    effective = {"endpoint": "http://127.0.0.1:8080", "request_mode": "completion"}
    return ProviderSelection(
        kind="llama_cpp",
        config_path="llama-cpp.local.yaml",
        expected_identity=ProviderIdentity(
            provider_type="llama_cpp",
            implementation_version="llama-cpp-completion-http-v1",
            model_alias="model-backed-test-model",
            build_id="model-backed-test-build",
            provider_config_hash=canonical_sha256(effective),
            model_path="C:/models/model-backed-test.gguf",
            model_sha256="b" * 64,
            chat_template_sha256="c" * 64,
        ),
        expected_effective_configuration_json=canonical_json(effective),
    )


def _dataset(
    purpose: DatasetPurpose,
    *,
    sequence_count: int = 2,
    turns_per_sequence: int = 2,
) -> PromptDataset:
    return PromptDataset(
        schema_version=1,
        dataset_id=f"model-backed-{purpose.value}",
        version=f"model-backed-{purpose.value}-v1",
        purpose=purpose,
        sequences=tuple(
            PromptSequence(
                sequence_id=f"sequence-{sequence_index}",
                cases=tuple(
                    PromptCase(
                        case_id=f"case-{sequence_index}-{turn_index}",
                        prompt_family="model_backed_test",
                        prompt=f"Return response {sequence_index}-{turn_index}.",
                        prompt_features=PromptFeatures(
                            {
                                "constraint_count": 1.0,
                                "target_length": 64.0,
                            }
                        ),
                        validator=ValidatorSpec(kind="non_empty"),
                    )
                    for turn_index in range(turns_per_sequence)
                ),
            )
            for sequence_index in range(sequence_count)
        ),
    )


def _schedule(dataset: PromptDataset, *, model_seed_count: int = 1) -> ScheduleSpec:
    turn_counts = {len(sequence.cases) for sequence in dataset.sequences}
    assert len(turn_counts) == 1
    turns_per_sequence = turn_counts.pop()
    return ScheduleSpec(
        sequence_count=len(dataset.sequences),
        turns_per_sequence=turns_per_sequence,
        model_seed_count=model_seed_count,
        controller_seed_count=1,
        policy_count=5,
        logical_generation_count=(
            len(dataset.sequences) * turns_per_sequence * model_seed_count * 1 * 5
        ),
    )


def _protocol(tier: RunTier, schedule: ScheduleSpec) -> ExperimentProtocol:
    return ExperimentProtocol(run_tier=tier, schedule=schedule)


def _policy_specs() -> tuple[
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    RandomMatchedPolicySpec,
]:
    return (
        BestStaticPolicySpec(),
        HeuristicAdaptivePolicySpec(),
        NeuralMatchedHistoryStateResetPolicySpec(),
        NeuralPersistentPolicySpec(),
        RandomMatchedPolicySpec(),
    )


def _selection_evidence() -> tuple[
    DevelopmentSelectionInput,
    StaticSelectionRecord,
    BaseDecodingProfile,
]:
    development = _dataset(DatasetPurpose.DEVELOPMENT, sequence_count=1)
    unit_keys = (MatchedUnitKey(prompt_sequence_id="sequence-0", model_seed=11),)
    selected = select_best_static(
        (
            StaticCandidateResult(
                profile=StaticProfile(
                    profile_id="static-balanced-v1",
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    presence_penalty=0.0,
                    max_tokens=64,
                ),
                unit_scores=(0.8,),
            ),
            StaticCandidateResult(
                profile=StaticProfile(
                    profile_id="static-conservative-v1",
                    temperature=0.5,
                    top_p=0.8,
                    top_k=20,
                    presence_penalty=0.0,
                    max_tokens=64,
                ),
                unit_scores=(0.7,),
            ),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=development.dataset_hash,
        development_unit_keys=unit_keys,
    )
    development_input = DevelopmentSelectionInput(
        dataset=DatasetReference(
            path="development.yaml",
            version=development.version,
            purpose=DatasetPurpose.DEVELOPMENT,
            expected_dataset_sha256=development.dataset_hash,
        )
    )
    winner = selected.winning_profile
    base = BaseDecodingProfile(
        temperature=winner.temperature,
        top_p=winner.top_p,
        top_k=winner.top_k,
        presence_penalty=winner.presence_penalty,
        max_tokens=winner.max_tokens,
    )
    return development_input, selected, base


def _config(
    dataset: PromptDataset,
    tier: RunTier,
    *,
    preregistration: PreregistrationSeal | None = None,
    schedule: ScheduleSpec | None = None,
) -> ExperimentConfig:
    protocol = _protocol(tier, schedule or _schedule(dataset))
    dataset_seal = (
        None
        if dataset.purpose is not DatasetPurpose.EVALUATION
        else DatasetSeal(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            dataset_sha256=dataset.dataset_hash,
        )
    )
    development_input = None
    selection = None
    evaluation = None
    confirmatory_analysis = None
    candidate_grid = None
    static_selection_evidence = None
    provider = ProviderSelection(
        kind="fake",
        expected_identity=fake_provider_identity(),
        expected_effective_configuration_json=fake_provider_effective_configuration_json(),
    )
    base = BaseDecodingProfile(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=64,
    )
    base_id = "model-backed-base-v1"
    if tier is RunTier.DEVELOPMENT_PILOT:
        selected_profile = MODEL_BACKED_STATIC_CANDIDATE_PROFILES[0]
        candidate_grid = DevelopmentPilotCandidateGrid(
            dataset_version=dataset.version,
            dataset_purpose=DatasetPurpose.DEVELOPMENT,
            dataset_sha256=dataset.dataset_hash,
            candidate_profiles=MODEL_BACKED_STATIC_CANDIDATE_PROFILES,
        )
        base = BaseDecodingProfile(
            temperature=selected_profile.temperature,
            top_p=selected_profile.top_p,
            top_k=selected_profile.top_k,
            presence_penalty=selected_profile.presence_penalty,
            max_tokens=selected_profile.max_tokens,
        )
        base_id = selected_profile.profile_id
    if tier is RunTier.CONFIRMATORY:
        development = _dataset(
            DatasetPurpose.DEVELOPMENT,
            sequence_count=6,
            turns_per_sequence=4,
        )
        static_selection_evidence = build_test_static_selection_evidence(
            development_dataset=development,
            winning_profile=MODEL_BACKED_STATIC_CANDIDATE_PROFILES[0],
        )
        winner = static_selection_evidence.selection_record.winning_profile
        pilot_manifest = static_selection_evidence.candidates[0].source_run_manifest
        development_input = DevelopmentSelectionInput(
            dataset=DatasetReference(
                path="development.yaml",
                version=development.version,
                purpose=DatasetPurpose.DEVELOPMENT,
                expected_dataset_sha256=development.dataset_hash,
            )
        )
        base = BaseDecodingProfile(
            temperature=winner.temperature,
            top_p=winner.top_p,
            top_k=winner.top_k,
            presence_penalty=winner.presence_penalty,
            max_tokens=winner.max_tokens,
        )
        base_id = winner.profile_id
        provider = ProviderSelection(
            kind="llama_cpp",
            config_path="llama-cpp.local.yaml",
            expected_identity=pilot_manifest.provider_identity,
            expected_effective_configuration_json=(
                pilot_manifest.provider_effective_configuration_json
            ),
        )
        evaluation = EvaluationSpec(
            focal_policy_id="neural_persistent",
            required_serious_comparator_ids=("best_static", "heuristic_adaptive"),
            negative_control_policy_ids=("random_matched",),
            bootstrap_seed=101,
            permutation_seed=102,
        )
        confirmatory_analysis = ConfirmatoryAnalysisSpec(
            efficacy=EfficacyAnalysisSpec(
                bootstrap_seed=101,
                permutation_seed=102,
            ),
            recovery=RecoveryAnalysisSpec(
                practical_thresholds={
                    RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: 0.0,
                    RecoveryMetricName.POST_STRESSOR_REPETITION_CHANGE: 0.0,
                    RecoveryMetricName.TIME_TO_RETURN_TO_TARGET_BAND: 0.0,
                },
                bootstrap_seed=103,
            ),
            attribution=AttributionAnalysisSpec(
                practical_effect_threshold=0.0,
                bootstrap_seed=104,
                permutation_seed=105,
            ),
            recovery_events=tuple(
                RecoveryEventSpec(
                    prompt_sequence_id=sequence.sequence_id,
                    stressor_turn_index=len(sequence.cases) - 2,
                    recovery_turn_indexes=(len(sequence.cases) - 1,),
                    minimum_task_score_target=0.8,
                    maximum_repetition_ratio_target=0.2,
                )
                for sequence in dataset.sequences
            ),
            optional_metric_dispositions={
                "semantic_similarity": LimitationDisposition.DISCLOSURE_ONLY,
            },
        )
    return ExperimentConfig(
        schema_version=1,
        experiment_id=f"model-backed-{tier.value}",
        dataset=DatasetReference(
            path="dataset.yaml",
            version=dataset.version,
            purpose=dataset.purpose,
            expected_dataset_sha256=dataset.dataset_hash,
            seal=dataset_seal,
        ),
        provider=provider,
        policy_specs=_policy_specs(),
        protocol=protocol,
        preregistration=preregistration,
        confirmatory_analysis=confirmatory_analysis,
        evaluation=evaluation,
        development_selection_input=development_input,
        candidate_grid=candidate_grid,
        static_selection_record=selection,
        static_selection_evidence=static_selection_evidence,
        model_seeds=(11,),
        controller_seeds=(21,),
        base_decoding_profile_id=base_id,
        base_decoding_profile=base,
        action_bounds=ActionBounds(),
        decoding_bounds=DecodingBounds(),
        metric_versions=METRIC_VERSIONS,
        decision_rule_version=protocol.decision_rule_version,
        database_schema_version=CURRENT_SCHEMA_VERSION,
        artifact_root="run",
    )


def _loaded(
    config: ExperimentConfig, dataset: PromptDataset
) -> tuple[LoadedExperimentConfig, LoadedDataset]:
    return (
        LoadedExperimentConfig(
            config=config,
            source_path=Path("config.yaml"),
            dataset_path=Path("dataset.yaml"),
            provider_config_path=(
                None if config.provider.kind == "fake" else Path("llama-cpp.local.yaml")
            ),
            artifact_root=Path("run"),
            development_selection_dataset_path=(
                None if config.development_selection_input is None else Path("development.yaml")
            ),
        ),
        LoadedDataset(dataset=dataset, source_path=Path("dataset.yaml")),
    )


def test_protocol_requires_exact_roles_and_cross_validates_schedule_product() -> None:
    schedule = ScheduleSpec(
        sequence_count=2,
        turns_per_sequence=2,
        model_seed_count=1,
        controller_seed_count=1,
        logical_generation_count=20,
    )
    protocol = ExperimentProtocol(run_tier=RunTier.ENGINEERING_SMOKE, schedule=schedule)

    assert protocol.policy_ids == MODEL_BACKED_POLICY_IDS
    assert protocol.efficacy_policy_ids == EFFICACY_POLICY_IDS
    assert protocol.attribution.policy_id == ATTRIBUTION_POLICY_ID
    assert protocol.attribution.history_source_policy_id == ATTRIBUTION_HISTORY_SOURCE_POLICY_ID
    assert protocol.decision_rule_version == "engineering-smoke-no-scientific-decision-v1"

    with pytest.raises(ValidationError, match="complete declared schedule product"):
        ScheduleSpec(
            sequence_count=2,
            turns_per_sequence=2,
            model_seed_count=1,
            controller_seed_count=1,
            logical_generation_count=19,
        )
    with pytest.raises(ValidationError, match="exact five"):
        ExperimentProtocol.model_validate(
            {
                **protocol.model_dump(mode="python"),
                "policy_ids": (*MODEL_BACKED_POLICY_IDS[:-1],),
            }
        )


@pytest.mark.parametrize(
    ("sequence_count", "turns_per_sequence", "model_seed_count", "logical_count"),
    (
        (2, 2, 1, 20),
        (6, 4, 2, 240),
        (24, 4, 5, 2_400),
    ),
)
def test_recommended_tier_counts_are_explicit_schedule_data(
    sequence_count: int,
    turns_per_sequence: int,
    model_seed_count: int,
    logical_count: int,
) -> None:
    schedule = ScheduleSpec(
        sequence_count=sequence_count,
        turns_per_sequence=turns_per_sequence,
        model_seed_count=model_seed_count,
        controller_seed_count=1,
        logical_generation_count=logical_count,
    )

    assert schedule.logical_generation_count == logical_count


def test_smoke_config_and_plan_materialize_exact_explicit_twenty_turn_grid() -> None:
    dataset = _dataset(DatasetPurpose.DEVELOPMENT)
    config = _config(dataset, RunTier.ENGINEERING_SMOKE)
    plan = build_plan(*_loaded(config, dataset))

    assert plan.protocol is not None
    assert plan.protocol.schedule.logical_generation_count == 20
    assert len(plan.turns) == 20
    assert {turn.condition.policy_id for turn in plan.turns} == set(MODEL_BACKED_POLICY_IDS)
    for offset in range(0, len(plan.turns), 5):
        assert plan.turns[offset + 4].condition.policy_id == "neural_matched_history_state_reset"


def test_config_rejects_policy_seed_tier_and_preregistration_drift() -> None:
    dataset = _dataset(DatasetPurpose.DEVELOPMENT)
    valid = _config(dataset, RunTier.DEVELOPMENT_PILOT)
    payload = valid.model_dump(mode="python")
    assert valid.protocol is not None

    with pytest.raises(ValidationError, match="exact five policy specs"):
        ExperimentConfig.model_validate({**payload, "policy_specs": _policy_specs()[:-1]})

    wrong_schedule = valid.protocol.model_copy(
        update={
            "schedule": valid.protocol.schedule.model_copy(
                update={"model_seed_count": 2, "logical_generation_count": 40}
            )
        }
    )
    with pytest.raises(ValidationError, match="model_seed_count"):
        ExperimentConfig.model_validate({**payload, "protocol": wrong_schedule})

    seal = PreregistrationSeal(
        experiment_id=valid.experiment_id,
        scientific_identity_sha256="a" * 64,
    )
    with pytest.raises(ValidationError, match="only confirmatory"):
        ExperimentConfig.model_validate({**payload, "preregistration": seal})

    evaluation_dataset = _dataset(DatasetPurpose.EVALUATION)
    with pytest.raises(ValidationError, match="development-purpose"):
        _config(evaluation_dataset, RunTier.ENGINEERING_SMOKE)


def test_model_backed_config_rejects_role_topology_and_schedule_drift() -> None:
    dataset = _dataset(DatasetPurpose.DEVELOPMENT)
    valid = _config(dataset, RunTier.ENGINEERING_SMOKE)
    payload = valid.model_dump(mode="python")
    assert valid.protocol is not None
    assert valid.policy_specs is not None

    with pytest.raises(ValidationError, match="requires typed policy_specs"):
        ExperimentConfig.model_validate(
            {
                **payload,
                "policy_ids": MODEL_BACKED_POLICY_IDS,
                "policy_specs": None,
            }
        )

    reset = next(
        spec
        for spec in valid.policy_specs
        if spec.policy_id == "neural_matched_history_state_reset"
    )
    wrong_edge_specs = tuple(
        reset.model_copy(update={"history_source_policy_id": "best_static"})
        if spec is reset
        else spec
        for spec in valid.policy_specs
    )
    with pytest.raises(ValidationError, match="exact attribution edge"):
        ExperimentConfig.model_validate({**payload, "policy_specs": wrong_edge_specs})

    persistent = next(spec for spec in valid.policy_specs if spec.policy_id == "neural_persistent")
    one_neural_specs = tuple(
        BestStaticPolicySpec().model_copy(update={"policy_id": "neural_persistent"})
        if spec is persistent
        else spec
        for spec in valid.policy_specs
    )
    with pytest.raises(ValidationError, match="both declared neural policies"):
        ExperimentConfig.model_validate({**payload, "policy_specs": one_neural_specs})

    controller_schedule = ScheduleSpec(
        sequence_count=2,
        turns_per_sequence=2,
        model_seed_count=1,
        controller_seed_count=2,
        logical_generation_count=40,
    )
    with pytest.raises(ValidationError, match="controller_seed_count"):
        ExperimentConfig.model_validate(
            {
                **payload,
                "protocol": valid.protocol.model_copy(update={"schedule": controller_schedule}),
            }
        )

    wrong_policy_count = valid.protocol.schedule.model_copy(
        update={"policy_count": 4, "logical_generation_count": 16}
    )
    with pytest.raises(ValidationError, match="policy_count"):
        ExperimentConfig.model_validate(
            {
                **payload,
                "protocol": valid.protocol.model_copy(update={"schedule": wrong_policy_count}),
            }
        )

    with pytest.raises(ValidationError, match="decision_rule_version"):
        ExperimentConfig.model_validate(
            {**payload, "decision_rule_version": "confirmatory-scientific-decision-v1"}
        )


def test_development_tiers_reject_confirmatory_claim_inputs() -> None:
    development = _dataset(DatasetPurpose.DEVELOPMENT)
    smoke = _config(development, RunTier.ENGINEERING_SMOKE)
    smoke_payload = smoke.model_dump(mode="python")
    evaluation_dataset = _dataset(
        DatasetPurpose.EVALUATION,
        sequence_count=1,
        turns_per_sequence=2,
    )
    confirmatory = _config(evaluation_dataset, RunTier.CONFIRMATORY)
    assert confirmatory.evaluation is not None
    assert confirmatory.confirmatory_analysis is not None

    with pytest.raises(ValidationError, match="cannot produce confirmatory evaluation"):
        ExperimentConfig.model_validate({**smoke_payload, "evaluation": confirmatory.evaluation})
    with pytest.raises(ValidationError, match="cannot carry confirmatory analysis"):
        ExperimentConfig.model_validate(
            {
                **smoke_payload,
                "confirmatory_analysis": confirmatory.confirmatory_analysis,
            }
        )


def test_plan_rejects_dataset_schedule_and_cross_arm_input_drift() -> None:
    three_turn_dataset = _dataset(
        DatasetPurpose.DEVELOPMENT,
        turns_per_sequence=3,
    )
    declared_two_turn_schedule = ScheduleSpec(
        sequence_count=2,
        turns_per_sequence=2,
        model_seed_count=1,
        controller_seed_count=1,
        logical_generation_count=20,
    )
    config = _config(
        three_turn_dataset,
        RunTier.ENGINEERING_SMOKE,
        schedule=declared_two_turn_schedule,
    )
    with pytest.raises(ValidationError, match="turns_per_sequence"):
        build_plan(*_loaded(config, three_turn_dataset))

    dataset = _dataset(DatasetPurpose.DEVELOPMENT)
    plan = build_plan(*_loaded(_config(dataset, RunTier.ENGINEERING_SMOKE), dataset))
    turns = list(plan.turns)
    turns[0] = turns[0].model_copy(update={"prompt": "cross-arm drift"})
    with pytest.raises(ValidationError, match="share exact current inputs"):
        ExperimentPlan.model_validate({**plan.model_dump(mode="python"), "turns": tuple(turns)})


def test_model_backed_plan_rejects_manifest_and_schedule_identity_drift() -> None:
    dataset = _dataset(DatasetPurpose.DEVELOPMENT)
    plan = build_plan(*_loaded(_config(dataset, RunTier.ENGINEERING_SMOKE), dataset))
    payload = plan.model_dump(mode="python")
    assert plan.protocol is not None

    with pytest.raises(ValidationError, match="at least one turn"):
        ExperimentPlan.model_validate({**payload, "turns": ()})
    with pytest.raises(ValidationError, match="duplicate conditions"):
        ExperimentPlan.model_validate({**payload, "turns": (*plan.turns, plan.turns[0])})
    with pytest.raises(ValidationError, match="exact five policy arms"):
        ExperimentPlan.model_validate(
            {
                **payload,
                "turns": tuple(
                    turn for turn in plan.turns if turn.condition.policy_id != "random_matched"
                ),
            }
        )
    with pytest.raises(ValidationError, match="wrong tier decision rule"):
        ExperimentPlan.model_validate(
            {**payload, "decision_rule_version": "development-pilot-no-scientific-decision-v1"}
        )
    with pytest.raises(ValidationError, match="current database schema"):
        ExperimentPlan.model_validate({**payload, "database_schema_version": 1})
    with pytest.raises(ValidationError, match="turn identity differs"):
        ExperimentPlan.model_validate({**payload, "experiment_id": "another-experiment"})

    schedule_cases = (
        ({"sequence_count": 3}, "sequence_count"),
        ({"model_seed_count": 2}, "model_seed_count"),
        ({"controller_seed_count": 2}, "controller_seed_count"),
        ({"logical_generation_count": 21}, "logical_generation_count"),
    )
    for changes, message in schedule_cases:
        schedule = plan.protocol.schedule.model_copy(update=changes)
        protocol = plan.protocol.model_copy(update={"schedule": schedule})
        with pytest.raises(ValidationError, match=message):
            ExperimentPlan.model_validate({**payload, "protocol": protocol})


def test_development_plan_rejects_cross_arm_profile_and_confirmatory_evidence() -> None:
    dataset = _dataset(DatasetPurpose.DEVELOPMENT)
    plan = build_plan(*_loaded(_config(dataset, RunTier.DEVELOPMENT_PILOT), dataset))
    payload = plan.model_dump(mode="python")

    turns = list(plan.turns)
    turns[0] = turns[0].model_copy(
        update={
            "condition": turns[0].condition.model_copy(
                update={"base_decoding_profile_id": "different-profile"}
            )
        }
    )
    with pytest.raises(ValidationError, match="one base decoding profile"):
        ExperimentPlan.model_validate({**payload, "turns": tuple(turns)})

    with pytest.raises(ValidationError, match="development-purpose data"):
        ExperimentPlan.model_validate({**payload, "dataset_purpose": DatasetPurpose.EVALUATION})

    seal = PreregistrationSeal(
        experiment_id=plan.experiment_id,
        scientific_identity_sha256="a" * 64,
    )
    with pytest.raises(ValidationError, match="cannot contain confirmatory evidence"):
        ExperimentPlan.model_validate({**payload, "preregistration": seal})


def test_confirmatory_candidate_can_be_sealed_and_any_scientific_drift_fails() -> None:
    dataset = _dataset(
        DatasetPurpose.EVALUATION,
        sequence_count=1,
        turns_per_sequence=2,
    )
    draft_config = _config(dataset, RunTier.CONFIRMATORY)

    with pytest.raises(ValueError, match="frozen preregistration seal"):
        build_plan(*_loaded(draft_config, dataset))
    candidate = build_plan(
        *_loaded(draft_config, dataset),
        require_frozen_preregistration=False,
    )
    seal = PreregistrationSeal(
        experiment_id=draft_config.experiment_id,
        scientific_identity_sha256=candidate.scientific_identity_sha256,
    )
    sealed_config = ExperimentConfig.model_validate(
        {**draft_config.model_dump(mode="python"), "preregistration": seal}
    )
    sealed = build_plan(*_loaded(sealed_config, dataset))

    assert sealed.scientific_identity_sha256 == candidate.scientific_identity_sha256
    assert len(seal.seal_sha256) == 64
    assert sealed.protocol is not None
    assert sealed.protocol.efficacy_policy_ids == EFFICACY_POLICY_IDS
    assert sealed.protocol.attribution.policy_id not in sealed.protocol.efficacy_policy_ids

    with pytest.raises(ValidationError, match="development-pilot evidence"):
        ExperimentConfig.model_validate(
            {
                **sealed_config.model_dump(mode="python"),
                "action_bounds": ActionBounds(temperature_delta=(-0.05, 0.05)),
            }
        )


def test_legacy_protocol_fields_are_absent_and_phase4_hash_is_unchanged() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root / "configs" / "experiments" / "phase4-neural-causal-smoke.yaml"
    ).config
    serialized = canonical_json(config)

    assert config.protocol is None
    assert config.preregistration is None
    assert config.confirmatory_analysis is None
    assert '"protocol"' not in serialized
    assert '"preregistration"' not in serialized
    assert '"confirmatory_analysis"' not in serialized
    assert config.experiment_config_hash == (
        "c8c5c39ce1c8cf01255535509a64647531b0fb05c50119f2144779fe05b16074"
    )

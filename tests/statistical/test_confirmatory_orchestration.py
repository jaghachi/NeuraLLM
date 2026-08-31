"""Provider-free tests for confirmatory scientific evaluation orchestration."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

import neurallm.experiments.scientific_analysis as scientific_analysis
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
    RunManifest,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.aggregation import aggregate_matched_units
from neurallm.evaluation.attribution import AttributionAnalysisSpec
from neurallm.evaluation.confirmatory import (
    ConfirmatoryAnalysisSpec,
    ConfirmatoryEvaluationResult,
    RecoveryEventSpec,
    confirmatory_result_sha256,
)
from neurallm.evaluation.models import (
    CoverageResult,
    DatasetPurpose,
    EvaluationSpec,
    MatchedUnitKey,
    TurnEvaluationRecord,
)
from neurallm.evaluation.recovery import RecoveryAnalysisSpec, RecoveryMetricName
from neurallm.evaluation.scientific import (
    EfficacyAnalysisSpec,
    LimitationDisposition,
    LimitationKind,
    ScientificDecisionState,
    ScientificEvidenceStatus,
)
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
    MODEL_BACKED_POLICY_IDS,
    ExperimentProtocol,
    PreregistrationSeal,
    RunTier,
    ScheduleSpec,
)
from neurallm.experiments.runner import (
    GitProvenance,
    build_policy_runtimes,
    build_run_manifest,
)
from neurallm.experiments.scientific_analysis import (
    ConfirmatoryAnalysisContext,
    analyze_closed_confirmatory_run,
    analyze_confirmatory_records,
    build_confirmatory_analysis_contract_sha256,
    confirmatory_analysis_contract_sha256,
)
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.fake import (
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.providers.llama_cpp import (
    LlamaCppEffectiveConfiguration,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)
from neurallm.storage import StoreInvariantError
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION
from neurallm.storage.models import DurableExecutionAccounting, RunFinalization

SEQUENCE_COUNT = 8
TURNS_PER_SEQUENCE = 4
MODEL_SEED = 11
CONTROLLER_SEED = 21


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


def _provider_selection(
    provider_kind: Literal["fake", "llama_cpp"],
) -> ProviderSelection:
    if provider_kind == "fake":
        return ProviderSelection(
            kind="fake",
            expected_identity=fake_provider_identity(),
            expected_effective_configuration_json=(fake_provider_effective_configuration_json()),
        )
    model_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "llama_cpp_model_stub.txt"
    ).resolve()
    model_sha256 = sha256(model_path.read_bytes()).hexdigest()
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    provider_config = LlamaCppProviderConfig(
        base_url="http://127.0.0.1:8080",
        model_alias="orchestration-test-model",
        model_path=str(model_path),
        model_sha256=model_sha256,
        build_id="orchestration-test-build",
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
                "n_predict": 64,
                "seed": 11,
            }
        ),
        total_slots=1,
    )
    return ProviderSelection(
        kind="llama_cpp",
        config_path="llama-cpp.yaml",
        expected_identity=llama_cpp_provider_identity(effective),
        expected_effective_configuration_json=canonical_json(effective),
    )


def _dataset(purpose: DatasetPurpose, *, sequence_count: int = SEQUENCE_COUNT) -> PromptDataset:
    return PromptDataset(
        schema_version=1,
        dataset_id=f"confirmatory-orchestration-{purpose.value}",
        version=f"confirmatory-orchestration-{purpose.value}-v1",
        purpose=purpose,
        sequences=tuple(
            PromptSequence(
                sequence_id=f"sequence-{sequence_index:02d}",
                cases=tuple(
                    PromptCase(
                        case_id=f"case-{sequence_index:02d}-{turn_index}",
                        prompt_family=(
                            "family_a" if sequence_index < sequence_count // 2 else "family_b"
                        ),
                        prompt=f"Return response {sequence_index}-{turn_index}.",
                        prompt_features=PromptFeatures(
                            {"constraint_count": 1.0, "target_length": 100.0}
                        ),
                        validator=ValidatorSpec(kind="non_empty"),
                    )
                    for turn_index in range(TURNS_PER_SEQUENCE)
                ),
            )
            for sequence_index in range(sequence_count)
        ),
    )


def _selection_evidence() -> tuple[
    DevelopmentSelectionInput,
    StaticSelectionRecord,
    BaseDecodingProfile,
]:
    development = _dataset(DatasetPurpose.DEVELOPMENT, sequence_count=1)
    selection = select_best_static(
        (
            StaticCandidateResult(
                profile=StaticProfile(
                    profile_id="selected-static-v1",
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    presence_penalty=0.0,
                    max_tokens=128,
                ),
                unit_scores=(0.8,),
            ),
            StaticCandidateResult(
                profile=StaticProfile(
                    profile_id="unselected-static-v1",
                    temperature=0.5,
                    top_p=0.8,
                    top_k=20,
                    presence_penalty=0.0,
                    max_tokens=128,
                ),
                unit_scores=(0.7,),
            ),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=development.dataset_hash,
        development_unit_keys=(
            MatchedUnitKey(prompt_sequence_id="sequence-00", model_seed=MODEL_SEED),
        ),
    )
    winner = selection.winning_profile
    return (
        DevelopmentSelectionInput(
            dataset=DatasetReference(
                path="development.yaml",
                version=development.version,
                purpose=DatasetPurpose.DEVELOPMENT,
                expected_dataset_sha256=development.dataset_hash,
            )
        ),
        selection,
        BaseDecodingProfile(
            temperature=winner.temperature,
            top_p=winner.top_p,
            top_k=winner.top_k,
            presence_penalty=winner.presence_penalty,
            max_tokens=winner.max_tokens,
        ),
    )


def _loaded(
    config: ExperimentConfig,
    dataset: PromptDataset,
) -> tuple[LoadedExperimentConfig, LoadedDataset]:
    return (
        LoadedExperimentConfig(
            config=config,
            source_path=Path("confirmatory.yaml"),
            dataset_path=Path("evaluation.yaml"),
            provider_config_path=(
                None if config.provider.config_path is None else Path("llama-cpp.yaml")
            ),
            artifact_root=Path("run"),
            development_selection_dataset_path=Path("development.yaml"),
        ),
        LoadedDataset(dataset=dataset, source_path=Path("evaluation.yaml")),
    )


def _draft_config(
    dataset: PromptDataset,
    *,
    optional_disposition: LimitationDisposition,
    provider_kind: Literal["fake", "llama_cpp"] = "fake",
) -> ExperimentConfig:
    development_input, selection, base = _selection_evidence()
    evaluation = EvaluationSpec(
        focal_policy_id="neural_persistent",
        required_serious_comparator_ids=("best_static", "heuristic_adaptive"),
        negative_control_policy_ids=("random_matched",),
        bootstrap_resamples=512,
        bootstrap_seed=101,
        permutation_resamples=512,
        permutation_seed=102,
        practical_effect_threshold=0.02,
    )
    analysis = ConfirmatoryAnalysisSpec(
        efficacy=EfficacyAnalysisSpec(
            practical_effect_threshold=evaluation.practical_effect_threshold,
            bootstrap_resamples=evaluation.bootstrap_resamples,
            confidence_level=evaluation.confidence_level,
            bootstrap_seed=evaluation.bootstrap_seed,
            permutation_resamples=evaluation.permutation_resamples,
            permutation_seed=evaluation.permutation_seed,
        ),
        recovery=RecoveryAnalysisSpec(
            practical_thresholds={
                RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: 0.05,
                RecoveryMetricName.POST_STRESSOR_REPETITION_CHANGE: 0.05,
                RecoveryMetricName.TIME_TO_RETURN_TO_TARGET_BAND: 0.5,
            },
            bootstrap_resamples=512,
            bootstrap_seed=103,
        ),
        attribution=AttributionAnalysisSpec(
            practical_effect_threshold=0.02,
            bootstrap_resamples=512,
            bootstrap_seed=104,
            permutation_resamples=512,
            permutation_seed=105,
        ),
        recovery_events=tuple(
            RecoveryEventSpec(
                prompt_sequence_id=sequence.sequence_id,
                stressor_turn_index=2,
                recovery_turn_indexes=(3,),
                minimum_task_score_target=0.8,
                maximum_repetition_ratio_target=0.2,
            )
            for sequence in dataset.sequences
        ),
        optional_metric_dispositions={"semantic_similarity": optional_disposition},
    )
    protocol = ExperimentProtocol(
        run_tier=RunTier.CONFIRMATORY,
        schedule=ScheduleSpec(
            sequence_count=SEQUENCE_COUNT,
            turns_per_sequence=TURNS_PER_SEQUENCE,
            model_seed_count=1,
            controller_seed_count=1,
            policy_count=5,
            logical_generation_count=(SEQUENCE_COUNT * TURNS_PER_SEQUENCE * 5),
        ),
    )
    return ExperimentConfig(
        schema_version=1,
        experiment_id="confirmatory-orchestration-test",
        dataset=DatasetReference(
            path="evaluation.yaml",
            version=dataset.version,
            purpose=DatasetPurpose.EVALUATION,
            expected_dataset_sha256=dataset.dataset_hash,
            seal=DatasetSeal(
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                dataset_sha256=dataset.dataset_hash,
            ),
        ),
        provider=_provider_selection(provider_kind),
        policy_specs=_policy_specs(),
        protocol=protocol,
        evaluation=evaluation,
        confirmatory_analysis=analysis,
        development_selection_input=development_input,
        static_selection_record=selection,
        model_seeds=(MODEL_SEED,),
        controller_seeds=(CONTROLLER_SEED,),
        base_decoding_profile_id=selection.winning_profile.profile_id,
        base_decoding_profile=base,
        action_bounds=ActionBounds(),
        decoding_bounds=DecodingBounds(),
        metric_versions=METRIC_VERSIONS,
        decision_rule_version=protocol.decision_rule_version,
        database_schema_version=CURRENT_SCHEMA_VERSION,
        artifact_root="run",
    )


def _plan(
    *,
    optional_disposition: LimitationDisposition = LimitationDisposition.DISCLOSURE_ONLY,
    provider_kind: Literal["fake", "llama_cpp"] = "llama_cpp",
) -> ExperimentPlan:
    dataset = _dataset(DatasetPurpose.EVALUATION)
    draft = _draft_config(
        dataset,
        optional_disposition=optional_disposition,
        provider_kind=provider_kind,
    )
    candidate = build_plan(
        *_loaded(draft, dataset),
        require_frozen_preregistration=False,
    )
    seal = PreregistrationSeal(
        experiment_id=draft.experiment_id,
        scientific_identity_sha256=candidate.scientific_identity_sha256,
    )
    sealed = ExperimentConfig.model_validate(
        {**draft.model_dump(mode="python"), "preregistration": seal}
    )
    return build_plan(*_loaded(sealed, dataset))


_METRICS: dict[str, tuple[tuple[float, float], ...]] = {
    "best_static": ((0.5, 0.4), (0.55, 0.35), (0.3, 0.7), (0.55, 0.35)),
    "heuristic_adaptive": ((0.5, 0.4), (0.6, 0.3), (0.3, 0.7), (0.6, 0.3)),
    "neural_persistent": ((0.5, 0.4), (0.8, 0.15), (0.3, 0.7), (0.9, 0.1)),
    "random_matched": ((0.5, 0.4), (0.4, 0.45), (0.3, 0.7), (0.4, 0.45)),
    "neural_matched_history_state_reset": (
        (0.5, 0.4),
        (0.55, 0.35),
        (0.25, 0.75),
        (0.6, 0.3),
    ),
}
_ACTION_MAGNITUDES = {
    "best_static": 0.02,
    "heuristic_adaptive": 0.06,
    "neural_persistent": 0.12,
    "random_matched": 0.0,
    "neural_matched_history_state_reset": 0.08,
}


def _records(
    plan: ExperimentPlan,
    *,
    focal_nonreturn_sequence: str | None = None,
    subgroup_conflict: bool = False,
) -> tuple[TurnEvaluationRecord, ...]:
    records: list[TurnEvaluationRecord] = []
    family_by_sequence = {
        turn.condition.prompt_sequence_id: turn.prompt_family for turn in plan.turns
    }
    for planned in plan.turns:
        condition = planned.condition
        task_score, repetition_ratio = _METRICS[condition.policy_id][condition.turn_index]
        if (
            focal_nonreturn_sequence == condition.prompt_sequence_id
            and condition.policy_id == "neural_persistent"
            and condition.turn_index == 3
        ):
            task_score, repetition_ratio = 0.7, 0.3
        if (
            subgroup_conflict
            and family_by_sequence[condition.prompt_sequence_id] == "family_b"
            and condition.policy_id in {"best_static", "heuristic_adaptive"}
            and condition.turn_index > 0
        ):
            task_score = 0.95
        turn_zero = condition.turn_index == 0
        records.append(
            TurnEvaluationRecord(
                dataset_sha256=plan.dataset_hash,
                prompt_sequence_id=condition.prompt_sequence_id,
                turn_index=condition.turn_index,
                policy_id=condition.policy_id,
                model_seed=condition.model_seed,
                controller_seed=condition.controller_seed,
                provider_identity_id=condition.provider_identity_id,
                has_previous_response=not turn_zero,
                previous_history_commitment_sha256=(None if turn_zero else "a" * 64),
                task_score=task_score,
                instruction_adherence=0.95,
                response_length_tokens=100,
                repetition_ratio=repetition_ratio,
                action_magnitude=(0.0 if turn_zero else _ACTION_MAGNITUDES[condition.policy_id]),
                action_within_bounds=True,
                action_saturated=False,
            )
        )
    return tuple(records)


def _analyze(
    plan: ExperimentPlan,
    records: tuple[TurnEvaluationRecord, ...],
) -> tuple[ConfirmatoryEvaluationResult, ConfirmatoryAnalysisContext]:
    return analyze_confirmatory_records(
        plan,
        records,
        optional_metric_availability={"semantic_similarity": (0, len(records))},
        causal_mechanism_validated=True,
    )


def _closed_run_evidence(plan: ExperimentPlan) -> tuple[RunManifest, RunFinalization]:
    runtimes = build_policy_runtimes(plan, _policy_specs())
    manifest = build_run_manifest(
        plan,
        plan.provider_identity,
        runtimes,
        GitProvenance(source_commit="a" * 40, working_tree_clean=True),
    )
    condition_ids = tuple(sorted(turn.condition.condition_id for turn in plan.turns))
    finalization = RunFinalization(
        expected_condition_ids=condition_ids,
        expected_condition_count=len(condition_ids),
        manifest_sha256=canonical_sha256(manifest),
        scientific_result_sha256="d" * 64,
        execution_accounting=DurableExecutionAccounting(
            planned_logical_generations=len(condition_ids),
            dispatched_logical_generations=len(condition_ids),
            successful_responses=len(condition_ids),
            uncertain_dispatches=0,
            committed_logical_generations=len(condition_ids),
        ),
    )
    return manifest, finalization


def test_full_five_arm_orchestration_is_positive_and_keeps_reset_attribution_only() -> None:
    plan = _plan()
    records = _records(plan)

    result, context = _analyze(plan, records)

    assert result.coverage.exact
    assert result.coverage.expected_count == SEQUENCE_COUNT * TURNS_PER_SEQUENCE * 5
    assert len(result.unit_outcomes) == SEQUENCE_COUNT * 4
    assert {outcome.policy_id for outcome in result.unit_outcomes} == set(
        MODEL_BACKED_POLICY_IDS
    ) - {"neural_matched_history_state_reset"}
    focal_unit = next(
        outcome
        for outcome in result.unit_outcomes
        if outcome.unit_key.prompt_sequence_id == "sequence-00"
        and outcome.policy_id == "neural_persistent"
    )
    assert focal_unit.guardrail_clean_task_score.raw_task_score == pytest.approx(0.625)
    assert focal_unit.guardrail_clean_task_score.gated_value == pytest.approx(0.625)
    assert tuple(item.comparator_policy_id for item in result.efficacy_comparisons) == (
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    )
    assert tuple(item.included_in_holm_family for item in result.efficacy_comparisons) == (
        True,
        True,
        False,
    )
    assert result.attribution.comparator_policy_id == "neural_matched_history_state_reset"
    assert result.attribution.turn_zero_excluded_from_effect
    assert result.attribution.unit_count == SEQUENCE_COUNT
    assert result.recovery.right_censored_focal_units == 0
    assert result.recovery.right_censored_comparator_units == SEQUENCE_COUNT * 2
    assert all(item.unit_count == SEQUENCE_COUNT for item in result.recovery.metric_results)
    assert result.recovery.status is ScientificEvidenceStatus.PASS
    assert result.decision.decision is ScientificDecisionState.VALIDATED_POSITIVE
    assert result.statistics_call_count == 22
    assert context.claim_eligible is False
    assert context.causal_mechanism_validated is True
    assert canonical_sha256(result.model_dump(mode="json", exclude={"result_sha256"})) == (
        result.result_sha256
    )


def test_recovery_uses_the_frozen_worst_serious_comparator_margin() -> None:
    plan = _plan()
    records = tuple(
        record.model_copy(update={"task_score": 1.0, "repetition_ratio": 0.0})
        if record.policy_id == "best_static" and record.turn_index == 3
        else record
        for record in _records(plan)
    )

    result, _ = _analyze(plan, records)

    assert plan.confirmatory_analysis is not None
    recovery_spec = plan.confirmatory_analysis.recovery
    assert recovery_spec.serious_comparator_ids == (
        "best_static",
        "heuristic_adaptive",
    )
    assert (
        recovery_spec.comparator_reduction_version
        == "per-unit-minimum-serious-comparator-margin-v1"
    )
    by_metric = {item.metric_name: item for item in result.recovery.metric_results}
    assert by_metric[RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE].estimate == pytest.approx(
        -0.1
    )
    assert by_metric[RecoveryMetricName.POST_STRESSOR_REPETITION_CHANGE].estimate == pytest.approx(
        -0.1
    )
    assert by_metric[RecoveryMetricName.TIME_TO_RETURN_TO_TARGET_BAND].estimate == 0.0
    assert result.recovery.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE
    assert result.decision.decision is ScientificDecisionState.VALIDATED_NEGATIVE


def test_confirmatory_envelope_rejects_analysis_and_decision_rebinding() -> None:
    plan = _plan()
    result, _ = _analyze(plan, _records(plan))
    assert plan.confirmatory_analysis is not None
    spec = plan.confirmatory_analysis
    assert len(result.subgroup_effects) == 4
    assert ConfirmatoryAnalysisSpec.model_validate_json(spec.model_dump_json()) == spec

    invalid_events = (
        ({"recovery_turn_indexes": ()}, "at least one"),
        ({"recovery_turn_indexes": (3, 3)}, "sorted and unique"),
        ({"recovery_turn_indexes": (2,)}, "strictly after"),
    )
    base_event = spec.recovery_events[0]
    for update, message in invalid_events:
        with pytest.raises(ValueError, match=message):
            RecoveryEventSpec.model_validate({**base_event.model_dump(mode="python"), **update})

    spec_payload = spec.model_dump(mode="python")
    with pytest.raises(ValueError, match="preregistered recovery events"):
        ConfirmatoryAnalysisSpec.model_validate({**spec_payload, "recovery_events": ()})
    with pytest.raises(ValueError, match="unique prompt sequences"):
        ConfirmatoryAnalysisSpec.model_validate(
            {**spec_payload, "recovery_events": (base_event, base_event)}
        )
    with pytest.raises(ValueError, match="missingness dispositions"):
        ConfirmatoryAnalysisSpec.model_validate(
            {**spec_payload, "optional_metric_dispositions": {}}
        )
    with pytest.raises(ValueError, match="subgroup fields"):
        ConfirmatoryAnalysisSpec.model_validate({**spec_payload, "subgroup_fields": ()})

    result_payload = result.model_dump(mode="python")
    with pytest.raises(ValueError, match="exactly three efficacy comparisons"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **result_payload,
                "efficacy_comparisons": result.efficacy_comparisons[:-1],
            }
        )
    with pytest.raises(ValueError, match="auditable unit outcomes"):
        ConfirmatoryEvaluationResult.model_validate({**result_payload, "unit_outcomes": ()})
    nonexact_coverage = CoverageResult(
        exact=False,
        expected_count=result.coverage.expected_count,
        observed_count=result.coverage.observed_count - 1,
    )
    with pytest.raises(ValueError, match="exact condition coverage"):
        ConfirmatoryEvaluationResult.model_validate(
            {**result_payload, "coverage": nonexact_coverage}
        )
    availability_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "optional_metric_availability": {
            "semantic_similarity": (0, result.coverage.observed_count - 1)
        },
    }
    with pytest.raises(ValueError, match="totals must match exact coverage"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **availability_payload,
                "result_sha256": confirmatory_result_sha256(availability_payload),
            }
        )
    decoy_guardrails = tuple(
        guardrail.model_copy(update={"scope": "decoy:scope"})
        if guardrail.name == "provider_identity_stability"
        else guardrail
        for guardrail in result.guardrails
    )
    decoy_guardrail_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "guardrails": decoy_guardrails,
    }
    with pytest.raises(ValueError, match="exact frozen scope set"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **decoy_guardrail_payload,
                "result_sha256": confirmatory_result_sha256(decoy_guardrail_payload),
            }
        )
    foreign_recovery = (
        result.recovery_unit_outcomes[0].model_copy(
            update={
                "unit_key": MatchedUnitKey(
                    prompt_sequence_id="foreign-recovery",
                    model_seed=999,
                )
            }
        ),
        *result.recovery_unit_outcomes[1:],
    )
    foreign_recovery_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "recovery_unit_outcomes": foreign_recovery,
    }
    with pytest.raises(ValueError, match="exact event/seed keys"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **foreign_recovery_payload,
                "result_sha256": confirmatory_result_sha256(foreign_recovery_payload),
            }
        )
    arbitrary_limitation_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "limitations": (
            *result.limitations,
            result.limitations[0].model_copy(update={"code": "arbitrary_unbound_limitation"}),
        ),
    }
    with pytest.raises(ValueError, match="limitations do not reconstruct"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **arbitrary_limitation_payload,
                "result_sha256": confirmatory_result_sha256(arbitrary_limitation_payload),
            }
        )
    first_comparison = result.efficacy_comparisons[0]
    assert first_comparison.negative_side_evidence is not None
    tampered_negative = first_comparison.negative_side_evidence.model_copy(
        update={
            "bootstrap": first_comparison.negative_side_evidence.bootstrap.model_copy(
                update={"resamples": 1, "seed": 999_999}
            )
        }
    )
    tampered_comparisons = (
        first_comparison.model_copy(update={"negative_side_evidence": tampered_negative}),
        *result.efficacy_comparisons[1:],
    )
    tampered_evidence_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "efficacy_comparisons": tampered_comparisons,
    }
    with pytest.raises(ValueError, match="bootstrap evidence does not match the analysis spec"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **tampered_evidence_payload,
                "result_sha256": confirmatory_result_sha256(tampered_evidence_payload),
            }
        )
    assert first_comparison.permutation is not None
    tampered_permutation = first_comparison.permutation.model_copy(
        update={"exact": False, "performed_permutations": 1}
    )
    permutation_comparisons = (
        first_comparison.model_copy(update={"permutation": tampered_permutation}),
        *result.efficacy_comparisons[1:],
    )
    permutation_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "efficacy_comparisons": permutation_comparisons,
    }
    with pytest.raises(ValueError, match="evidence parameters do not match"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **permutation_payload,
                "result_sha256": confirmatory_result_sha256(permutation_payload),
            }
        )
    tampered_count_payload = {
        **result.model_dump(mode="python", exclude={"result_sha256"}),
        "statistics_call_count": 0,
    }
    with pytest.raises(ValueError, match="statistics call count"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **tampered_count_payload,
                "result_sha256": confirmatory_result_sha256(tampered_count_payload),
            }
        )
    with pytest.raises(ValueError, match="does not hash the enclosed evidence"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **result_payload,
                "decision": result.decision.model_copy(update={"decision_input_sha256": "f" * 64}),
            }
        )
    with pytest.raises(ValueError, match="does not match the enclosed evidence"):
        ConfirmatoryEvaluationResult.model_validate(
            {
                **result_payload,
                "decision": result.decision.model_copy(update={"limitations": ()}),
            }
        )
    with pytest.raises(ValueError, match="result hash"):
        ConfirmatoryEvaluationResult.model_validate({**result_payload, "result_sha256": "f" * 64})


def test_focal_no_return_is_counted_and_forces_a_validated_negative() -> None:
    plan = _plan()
    records = _records(plan, focal_nonreturn_sequence="sequence-00")

    result, _ = _analyze(plan, records)

    assert result.recovery.right_censored_focal_units == 1
    assert result.recovery.right_censored_comparator_units == SEQUENCE_COUNT * 2
    assert all(item.unit_count == SEQUENCE_COUNT for item in result.recovery.metric_results)
    assert result.recovery.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE
    assert result.decision.decision is ScientificDecisionState.VALIDATED_NEGATIVE
    censoring = next(item for item in result.limitations if item.code == "recovery_right_censoring")
    assert "focal=1" in censoring.detail
    assert f"serious_comparator={SEQUENCE_COUNT * 2}" in censoring.detail


def test_optional_missingness_and_prompt_family_conflicts_follow_frozen_dispositions() -> None:
    missing_plan = _plan(optional_disposition=LimitationDisposition.INCONCLUSIVE)
    missing_result, _ = _analyze(missing_plan, _records(missing_plan))
    optional = next(
        item
        for item in missing_result.limitations
        if item.kind is LimitationKind.OPTIONAL_METRIC_UNAVAILABLE
    )
    assert optional.disposition is LimitationDisposition.INCONCLUSIVE
    assert missing_result.decision.decision is ScientificDecisionState.INCONCLUSIVE

    conflict_plan = _plan()
    conflict_result, _ = _analyze(
        conflict_plan,
        _records(conflict_plan, subgroup_conflict=True),
    )
    conflicts = tuple(
        item
        for item in conflict_result.limitations
        if item.kind is LimitationKind.SUBGROUP_CONFLICT
    )
    assert {item.code for item in conflicts} == {
        "prompt_family_conflict_best_static",
        "prompt_family_conflict_heuristic_adaptive",
    }
    assert all(item.disposition is LimitationDisposition.INCONCLUSIVE for item in conflicts)
    assert conflict_result.decision.decision is ScientificDecisionState.INCONCLUSIVE


def test_exact_five_arm_coverage_and_causal_validation_fail_closed() -> None:
    plan = _plan()
    records = _records(plan)
    missing_reset = tuple(
        record
        for record in records
        if not (
            record.policy_id == "neural_matched_history_state_reset"
            and record.prompt_sequence_id == "sequence-00"
            and record.turn_index == 3
        )
    )

    with pytest.raises(ValueError, match="exact full five-arm coverage"):
        _analyze(plan, missing_reset)
    with pytest.raises(ValueError, match="validated causal evidence"):
        analyze_confirmatory_records(
            plan,
            records,
            optional_metric_availability={"semantic_similarity": (0, len(records))},
            causal_mechanism_validated=False,
        )

    invalid_provider = list(records)
    invalid_provider[0] = invalid_provider[0].model_copy(update={"provider_identity_id": "f" * 64})
    invalid_result, _ = _analyze(plan, tuple(invalid_provider))
    assert invalid_result.decision.decision is ScientificDecisionState.INVALID_RUN
    assert any(
        outcome.guardrail_clean_task_score.gated_value is None
        for outcome in invalid_result.unit_outcomes
    )

    saturated = tuple(
        record.model_copy(update={"action_saturated": True})
        if record.policy_id == "neural_persistent"
        else record
        for record in records
    )
    saturated_result, _ = _analyze(plan, saturated)
    assert saturated_result.decision.decision is ScientificDecisionState.VALIDATED_NEGATIVE
    assert any(
        outcome.policy_id == "neural_persistent"
        and outcome.guardrail_clean_task_score.gated_value is None
        for outcome in saturated_result.unit_outcomes
    )


def test_orchestration_helpers_reject_foreign_contracts_and_malformed_evidence() -> None:
    plan = _plan()
    records = _records(plan)
    assert plan.confirmatory_analysis is not None

    with pytest.raises(TypeError, match="ExperimentPlan"):
        scientific_analysis._require_confirmatory_plan(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="five-arm protocol"):
        scientific_analysis._require_confirmatory_plan(plan.model_copy(update={"protocol": None}))
    with pytest.raises(ValueError, match="model-artifact SHA-256"):
        scientific_analysis._require_confirmatory_plan(
            plan.model_copy(
                update={
                    "provider_identity": plan.provider_identity.model_copy(
                        update={"model_sha256": None}
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="frozen analysis evidence"):
        scientific_analysis._require_confirmatory_plan(plan.model_copy(update={"evaluation": None}))
    with pytest.raises(ValueError, match="sealed evaluation dataset"):
        scientific_analysis._require_confirmatory_plan(
            plan.model_copy(update={"dataset_seal": None})
        )
    with pytest.raises(ValueError, match="published preregistration"):
        scientific_analysis._require_confirmatory_plan(
            plan.model_copy(update={"preregistration": None})
        )

    def reseal(candidate: ExperimentPlan) -> ExperimentPlan:
        unsealed = candidate.model_copy(update={"preregistration": None})
        seal = PreregistrationSeal(
            experiment_id=unsealed.experiment_id,
            scientific_identity_sha256=unsealed.scientific_identity_sha256,
        )
        return unsealed.model_copy(update={"preregistration": seal})

    missing_arm = reseal(
        plan.model_copy(
            update={
                "turns": tuple(
                    turn for turn in plan.turns if turn.condition.policy_id != "random_matched"
                )
            }
        )
    )
    with pytest.raises(ValueError, match="exact five policy arms"):
        scientific_analysis._require_confirmatory_plan(missing_arm)

    wrong_subgroup = reseal(
        plan.model_copy(
            update={
                "confirmatory_analysis": plan.confirmatory_analysis.model_copy(
                    update={"subgroup_fields": ("topic",)}
                )
            }
        )
    )
    with pytest.raises(ValueError, match="prompt_family subgroup"):
        scientific_analysis._require_confirmatory_plan(wrong_subgroup)

    efficacy_records = tuple(
        record for record in records if record.policy_id != "neural_matched_history_state_reset"
    )
    outcomes = aggregate_matched_units(efficacy_records)
    with pytest.raises(ValueError, match="duplicate matched unit"):
        scientific_analysis._outcomes_by_policy((outcomes[0], outcomes[0]))
    by_policy = scientific_analysis._outcomes_by_policy(outcomes)
    missing_key = {policy_id: dict(values) for policy_id, values in by_policy.items()}
    missing_key["best_static"].pop(next(iter(missing_key["best_static"])))
    with pytest.raises(ValueError, match="exact matched-unit keys"):
        scientific_analysis._paired_task_differences(missing_key, "best_static")
    with pytest.raises(ValueError, match="empty group"):
        scientific_analysis._mean(())
    with pytest.raises(ValueError, match="complete task and repetition"):
        scientific_analysis._turn_metric_means(
            (records[0].model_copy(update={"task_score": None}),)
        )

    drifted_turns = list(plan.turns)
    drifted_turns[0] = drifted_turns[0].model_copy(update={"prompt_family": "drift"})
    with pytest.raises(ValueError, match="one family per sequence"):
        scientific_analysis._prompt_family_by_sequence(
            plan.model_copy(update={"turns": tuple(drifted_turns)})
        )

    with pytest.raises(ValueError, match="disposition set"):
        scientific_analysis._optional_metric_limitations(plan.confirmatory_analysis, {})
    with pytest.raises(ValueError, match="counts are invalid"):
        scientific_analysis._optional_metric_limitations(
            plan.confirmatory_analysis,
            {"semantic_similarity": (True, len(records))},
        )
    assert (
        scientific_analysis._optional_metric_limitations(
            plan.confirmatory_analysis,
            {"semantic_similarity": (len(records), len(records))},
        )
        == ()
    )
    availability = scientific_analysis._optional_metric_availability_from_turns(
        (
            SimpleNamespace(
                metrics=SimpleNamespace(semantic_similarity=SimpleNamespace(availability=True))
            ),
            SimpleNamespace(metrics=None),
        ),  # type: ignore[arg-type]
        plan.confirmatory_analysis,
    )
    assert availability == {"semantic_similarity": (1, 2)}

    returning_comparators = tuple(
        record.model_copy(update={"task_score": 0.9, "repetition_ratio": 0.1})
        if record.policy_id in {"best_static", "heuristic_adaptive"} and record.turn_index == 3
        else record
        for record in records
    )
    _, _, focal_censored, comparator_censored = scientific_analysis._recovery_evidence(
        returning_comparators,
        plan.confirmatory_analysis,
    )
    assert focal_censored == 0
    assert comparator_censored == 0


def test_contract_is_recomputable_from_manifest_fields_and_real_path_rejects_fake() -> None:
    plan = _plan()
    assert plan.confirmatory_analysis is not None
    assert plan.preregistration is not None
    assert plan.dataset_seal is not None
    assert plan.dataset_purpose is DatasetPurpose.EVALUATION
    assert plan.evaluation is not None
    assert plan.evaluation_spec_sha256 is not None
    spec_hash = canonical_sha256(plan.confirmatory_analysis)
    prompt_family_by_sequence = scientific_analysis._prompt_family_by_sequence(plan)
    prompt_family_design_sha256 = canonical_sha256(prompt_family_by_sequence)

    rebuilt = confirmatory_analysis_contract_sha256(
        scientific_identity_sha256=plan.scientific_identity_sha256,
        preregistration_sha256=plan.preregistration.seal_sha256,
        confirmatory_analysis_spec=plan.confirmatory_analysis,
        confirmatory_analysis_spec_sha256=spec_hash,
        evaluation_spec=plan.evaluation,
        evaluation_spec_sha256=plan.evaluation_spec_sha256,
        prompt_family_by_sequence=prompt_family_by_sequence,
        prompt_family_design_sha256=prompt_family_design_sha256,
        dataset_sha256=plan.dataset_hash,
        dataset_purpose=plan.dataset_purpose,
        dataset_seal_sha256=plan.dataset_seal.seal_sha256,
    )
    assert rebuilt == build_confirmatory_analysis_contract_sha256(plan)
    assert len(rebuilt) == 64
    with pytest.raises(ValueError, match="spec hash"):
        confirmatory_analysis_contract_sha256(
            scientific_identity_sha256=plan.scientific_identity_sha256,
            preregistration_sha256=plan.preregistration.seal_sha256,
            confirmatory_analysis_spec=plan.confirmatory_analysis,
            confirmatory_analysis_spec_sha256="f" * 64,
            evaluation_spec=plan.evaluation,
            evaluation_spec_sha256=plan.evaluation_spec_sha256,
            prompt_family_by_sequence=prompt_family_by_sequence,
            prompt_family_design_sha256=prompt_family_design_sha256,
            dataset_sha256=plan.dataset_hash,
            dataset_purpose=plan.dataset_purpose,
            dataset_seal_sha256=plan.dataset_seal.seal_sha256,
        )
    drifted_evaluation = plan.evaluation.model_copy(update={"maximum_action_saturation_rate": 0.99})
    with pytest.raises(ValueError, match="evaluation spec hash"):
        confirmatory_analysis_contract_sha256(
            scientific_identity_sha256=plan.scientific_identity_sha256,
            preregistration_sha256=plan.preregistration.seal_sha256,
            confirmatory_analysis_spec=plan.confirmatory_analysis,
            confirmatory_analysis_spec_sha256=spec_hash,
            evaluation_spec=drifted_evaluation,
            evaluation_spec_sha256=plan.evaluation_spec_sha256,
            prompt_family_by_sequence=prompt_family_by_sequence,
            prompt_family_design_sha256=prompt_family_design_sha256,
            dataset_sha256=plan.dataset_hash,
            dataset_purpose=plan.dataset_purpose,
            dataset_seal_sha256=plan.dataset_seal.seal_sha256,
        )
    fake_candidate = plan.model_copy(
        update={
            "provider_identity": fake_provider_identity(),
            "provider_effective_configuration_json": (fake_provider_effective_configuration_json()),
            "preregistration": None,
        }
    )
    fake_plan = fake_candidate.model_copy(
        update={
            "preregistration": PreregistrationSeal(
                experiment_id=fake_candidate.experiment_id,
                scientific_identity_sha256=fake_candidate.scientific_identity_sha256,
            )
        }
    )
    with pytest.raises(ValueError, match="requires llama_cpp"):
        analyze_closed_confirmatory_run(fake_plan, Path("never-opened.sqlite3"))
    with pytest.raises(TypeError, match="pathlib.Path"):
        analyze_closed_confirmatory_run(plan, "not-a-path")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evaluation-purpose"):
        confirmatory_analysis_contract_sha256(
            scientific_identity_sha256=plan.scientific_identity_sha256,
            preregistration_sha256=plan.preregistration.seal_sha256,
            confirmatory_analysis_spec=plan.confirmatory_analysis,
            confirmatory_analysis_spec_sha256=spec_hash,
            evaluation_spec=plan.evaluation,
            evaluation_spec_sha256=plan.evaluation_spec_sha256,
            prompt_family_by_sequence=prompt_family_by_sequence,
            prompt_family_design_sha256=prompt_family_design_sha256,
            dataset_sha256=plan.dataset_hash,
            dataset_purpose=DatasetPurpose.SYNTHETIC,
            dataset_seal_sha256=plan.dataset_seal.seal_sha256,
        )


def test_real_closed_run_path_validates_exact_llama_manifest_and_causal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neurallm.reporting import artifacts as reporting_artifacts

    plan = _plan(provider_kind="llama_cpp")
    records = _records(plan)
    manifest, finalization = _closed_run_evidence(plan)
    scientific_analysis._validate_manifest(plan, manifest)
    scientific_analysis._validate_finalization(plan, manifest, finalization)

    with pytest.raises(StoreInvariantError, match="manifest does not exactly match"):
        scientific_analysis._validate_manifest(
            plan,
            manifest.model_copy(update={"working_tree_clean": False}),
        )
    with pytest.raises(StoreInvariantError, match="does not close"):
        scientific_analysis._validate_finalization(
            plan,
            manifest,
            finalization.model_copy(update={"manifest_sha256": "f" * 64}),
        )

    class ClosedStore:
        def __init__(self, database_path: Path) -> None:
            assert database_path == Path("closed.sqlite3")

        def __enter__(self) -> ClosedStore:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def verify_integrity(self) -> None:
            return None

        def get_manifest(self) -> RunManifest | None:
            return manifest

        def get_finalization(self) -> RunFinalization | None:
            return finalization

        def list_turns(self) -> tuple[object, ...]:
            return ()

        def list_turn_inputs(self) -> tuple[object, ...]:
            return ()

    monkeypatch.setattr(scientific_analysis, "SQLiteRunStore", ClosedStore)
    monkeypatch.setattr(
        scientific_analysis,
        "scientific_result_sha256",
        lambda _turns: "0" * 64,
    )
    with pytest.raises(StoreInvariantError, match="does not match the committed"):
        analyze_closed_confirmatory_run(plan, Path("closed.sqlite3"))

    monkeypatch.setattr(
        scientific_analysis,
        "scientific_result_sha256",
        lambda _turns: finalization.scientific_result_sha256,
    )
    monkeypatch.setattr(
        scientific_analysis,
        "evaluation_records_from_store",
        lambda _plan, _store: records,
    )
    monkeypatch.setattr(
        scientific_analysis,
        "_optional_metric_availability_from_turns",
        lambda _turns, _spec: {"semantic_similarity": (0, len(records))},
    )

    def causal_failure(*_args: object) -> None:
        raise ValueError("causal evidence drift")

    monkeypatch.setattr(
        reporting_artifacts,
        "_validate_phase4_mechanism_evidence",
        causal_failure,
    )
    with pytest.raises(StoreInvariantError, match="causal mechanism evidence is invalid"):
        analyze_closed_confirmatory_run(plan, Path("closed.sqlite3"))

    monkeypatch.setattr(
        reporting_artifacts,
        "_validate_phase4_mechanism_evidence",
        lambda *_args: None,
    )
    result, context = analyze_closed_confirmatory_run(plan, Path("closed.sqlite3"))
    assert result.decision.decision is ScientificDecisionState.VALIDATED_POSITIVE
    assert context.claim_eligible is True
    assert context.run_manifest_sha256 == canonical_sha256(manifest)
    assert context.run_finalization_sha256 == canonical_sha256(finalization)

    class UnboundStore(ClosedStore):
        def get_manifest(self) -> None:
            return None

    monkeypatch.setattr(scientific_analysis, "SQLiteRunStore", UnboundStore)
    with pytest.raises(StoreInvariantError, match="manifest-bound closed run"):
        analyze_closed_confirmatory_run(plan, Path("closed.sqlite3"))

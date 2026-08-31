"""Atomic schema-v2 persistence for confirmatory scientific evidence."""

from __future__ import annotations

import csv
import json
import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest

from neurallm.control.action_space import apply_action
from neurallm.control.policy import PolicyState
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    PromptFeatures,
    ProviderIdentity,
    RunManifest,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import CoverageResult, DatasetPurpose, EvaluationSpec, MatchedUnitKey
from neurallm.evaluation.attribution import (
    AttributionAnalysisSpec,
    evaluate_persistent_state_attribution,
)
from neurallm.evaluation.confirmatory import (
    AttributionUnitOutcome,
    ConfirmatoryAnalysisSpec,
    ConfirmatoryEvaluationResult,
    RecoveryEventSpec,
    RecoveryUnitOutcome,
    ScientificUnitOutcome,
    confirmatory_result_sha256,
)
from neurallm.evaluation.recovery import (
    RECOVERY_METRIC_NAMES,
    RecoveryAnalysisSpec,
    evaluate_recovery,
)
from neurallm.evaluation.scientific import (
    DEFAULT_VALIDATED_NEGATIVE_MULTIPLICITY,
    EfficacyAnalysisSpec,
    ExperimentTier,
    GuardrailCleanTaskScore,
    LimitationDisposition,
    LimitationKind,
    ScientificDecisionInput,
    ScientificDecisionRecord,
    ScientificDecisionState,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
    ScientificLimitation,
    ScientificReasonCode,
    decide_scientific_outcome,
    evaluate_efficacy_comparisons,
)
from neurallm.experiments.protocol import CONFIRMATORY_DECISION_RULE_VERSION
from neurallm.experiments.runner import DetailedAppliedPolicyTrace
from neurallm.experiments.scientific_analysis import (
    ConfirmatoryAnalysisContext,
    confirmatory_analysis_contract_sha256,
)
from neurallm.metrics import MetricContext, ValidatorSpec, compute_response_metrics
from neurallm.providers.base import GenerationMetadata, GenerationResponse
from neurallm.providers.fake import FakeProvider
from neurallm.providers.llama_cpp import (
    LlamaCppEffectiveConfiguration,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)
from neurallm.reporting import artifacts as reporting_artifacts
from neurallm.reporting import export_closed_run, scientific_result_sha256
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    DurableExecutionAccounting,
    RunFinalization,
    ScientificAnalysisManifest,
    SQLiteRunStore,
    StoreCorruptionError,
    StoreInvariantError,
    TurnInputEvidence,
)
from tests.storage.helpers import make_metrics, make_request, make_trace

_SCIENTIFIC_POLICY_IDS = (
    "best_static",
    "heuristic_adaptive",
    "neural_matched_history_state_reset",
    "neural_persistent",
    "random_matched",
)


def _llama_identity_and_effective_json() -> tuple[ProviderIdentity, str]:
    model_path = Path(__file__).resolve().parents[1] / "fixtures" / "llama_cpp_model_stub.txt"
    model_sha256 = sha256(model_path.read_bytes()).hexdigest()
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    provider_config = LlamaCppProviderConfig(
        base_url="http://127.0.0.1:8080",
        model_alias="test-model",
        model_path=str(model_path.resolve()),
        model_sha256=model_sha256,
        build_id="test-build",
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
    return llama_cpp_provider_identity(effective), canonical_json(effective)


def _turn_inputs(
    identity: ProviderIdentity,
    *,
    prompt_family: str = "family-a",
    prompt_features: PromptFeatures | None = None,
    validator: ValidatorSpec | None = None,
) -> tuple[TurnInputEvidence, ...]:
    features = prompt_features or PromptFeatures({})
    frozen_validator = validator or ValidatorSpec(kind="non_empty")
    return tuple(
        sorted(
            (
                TurnInputEvidence(
                    condition_id=make_request(
                        identity,
                        policy_id=policy_id,
                        turn_index=turn_index,
                    ).condition_id,
                    prompt_case_id=f"case-{turn_index}",
                    prompt_family=prompt_family,
                    prompt_features=features,
                    validator=frozen_validator,
                )
                for turn_index in (0, 1)
                for policy_id in _SCIENTIFIC_POLICY_IDS
            ),
            key=lambda item: item.condition_id,
        )
    )


def _run_manifest(
    *,
    identity: ProviderIdentity | None = None,
    effective_json: str | None = None,
    decision_rule_version: str = CONFIRMATORY_DECISION_RULE_VERSION,
    run_tier: str = "confirmatory",
    preregistration_sha256: str | None = None,
    prompt_family: str = "family-a",
    prompt_features: PromptFeatures | None = None,
    validator: ValidatorSpec | None = None,
) -> RunManifest:
    if identity is None or effective_json is None:
        identity, effective_json = _llama_identity_and_effective_json()
    if preregistration_sha256 is None and run_tier == "confirmatory":
        preregistration_sha256 = canonical_sha256("preregistration")
    scientific_identity_sha256 = canonical_sha256("confirmatory-plan")
    dataset_sha256 = canonical_sha256("confirmatory-dataset")
    dataset_seal_sha256 = canonical_sha256("confirmatory-dataset-seal")
    turn_inputs = _turn_inputs(
        identity,
        prompt_family=prompt_family,
        prompt_features=prompt_features,
        validator=validator,
    )
    turn_input_evidence_sha256 = canonical_sha256(turn_inputs)
    confirmatory_contract_sha256 = None
    if run_tier == "confirmatory":
        assert preregistration_sha256 is not None
        spec = _confirmatory_spec()
        prompt_family_by_sequence = {"sequence-a": prompt_family}
        confirmatory_contract_sha256 = confirmatory_analysis_contract_sha256(
            scientific_identity_sha256=scientific_identity_sha256,
            preregistration_sha256=preregistration_sha256,
            confirmatory_analysis_spec=spec,
            confirmatory_analysis_spec_sha256=canonical_sha256(spec),
            evaluation_spec=_evaluation_spec(),
            evaluation_spec_sha256=canonical_sha256(_evaluation_spec()),
            turn_input_evidence_sha256=turn_input_evidence_sha256,
            prompt_family_by_sequence=prompt_family_by_sequence,
            prompt_family_design_sha256=canonical_sha256(prompt_family_by_sequence),
            dataset_sha256=dataset_sha256,
            dataset_purpose=DatasetPurpose.EVALUATION,
            dataset_seal_sha256=dataset_seal_sha256,
        )
    return RunManifest(
        source_commit="0" * 40,
        working_tree_clean=True,
        experiment_config_hash=canonical_sha256("confirmatory-config"),
        dataset_hash=dataset_sha256,
        provider_config_hash=identity.provider_config_hash,
        provider_identity=identity,
        provider_effective_configuration_json=effective_json,
        policy_config_hashes={
            policy_id: canonical_sha256({"policy_id": policy_id})
            for policy_id in _SCIENTIFIC_POLICY_IDS
        },
        matched_history_policy_sources={"neural_matched_history_state_reset": "neural_persistent"},
        metric_versions={"test-metrics": "1.0.0"},
        seed_schedule=SeedSchedule(model_seeds=(7,), controller_seeds=(11,)),
        action_bounds=ActionBounds(),
        decision_rule_version=decision_rule_version,
        database_schema_version=CURRENT_SCHEMA_VERSION,
        evaluation_spec_json=canonical_json(_evaluation_spec()),
        evaluation_spec_sha256=canonical_sha256(_evaluation_spec()),
        turn_input_evidence_sha256=(
            turn_input_evidence_sha256 if run_tier == "confirmatory" else None
        ),
        run_tier=run_tier,
        scientific_identity_sha256=scientific_identity_sha256,
        preregistration_sha256=preregistration_sha256,
        confirmatory_analysis_contract_sha256=confirmatory_contract_sha256,
    )


def _evaluation_spec() -> EvaluationSpec:
    return EvaluationSpec(
        focal_policy_id="neural_persistent",
        required_serious_comparator_ids=("best_static", "heuristic_adaptive"),
        negative_control_policy_ids=("random_matched",),
        bootstrap_seed=101,
        permutation_seed=202,
    )


def _confirmatory_spec() -> ConfirmatoryAnalysisSpec:
    return ConfirmatoryAnalysisSpec(
        efficacy=EfficacyAnalysisSpec(
            bootstrap_resamples=16,
            bootstrap_seed=101,
            permutation_resamples=16,
            permutation_seed=202,
        ),
        recovery=RecoveryAnalysisSpec(
            practical_thresholds={metric_name: 0.01 for metric_name in RECOVERY_METRIC_NAMES},
            bootstrap_resamples=16,
            bootstrap_seed=303,
        ),
        attribution=AttributionAnalysisSpec(
            bootstrap_resamples=16,
            bootstrap_seed=404,
            permutation_resamples=16,
            permutation_seed=505,
        ),
        recovery_events=(
            RecoveryEventSpec(
                prompt_sequence_id="sequence-a",
                stressor_turn_index=0,
                recovery_turn_indexes=(1,),
                minimum_task_score_target=0.5,
                maximum_repetition_ratio_target=0.5,
            ),
        ),
        optional_metric_dispositions={"semantic_similarity": LimitationDisposition.DISCLOSURE_ONLY},
    )


def _guardrails() -> tuple[ScientificGuardrailResult, ...]:
    guardrails = [
        ScientificGuardrailResult(
            name="action_bound_compliance",
            status=ScientificGuardrailStatus.PASS,
            scope="efficacy:global",
            detail="every recorded action is within the frozen bounds",
            observed_value=10.0,
            threshold=10.0,
        ),
        ScientificGuardrailResult(
            name="matched_condition_coverage",
            status=ScientificGuardrailStatus.PASS,
            scope="efficacy:global",
            detail="observed keys exactly match the frozen condition grid",
            observed_value=1.0,
            threshold=1.0,
        ),
        ScientificGuardrailResult(
            name="metric_availability",
            status=ScientificGuardrailStatus.PASS,
            scope="efficacy:global",
            detail="all required evaluator metrics are available",
            observed_value=10.0,
            threshold=10.0,
        ),
        ScientificGuardrailResult(
            name="provider_identity_stability",
            status=ScientificGuardrailStatus.PASS,
            scope="efficacy:global",
            detail="every record matches the frozen provider identity",
            observed_value=1.0,
            threshold=1.0,
        ),
        ScientificGuardrailResult(
            name="turn_zero_equivalence",
            status=ScientificGuardrailStatus.PASS,
            scope="efficacy:global",
            detail="all turn-zero conditions have explicit null/false history",
            observed_value=5.0,
            threshold=5.0,
        ),
        ScientificGuardrailResult(
            name="action_saturation_rate",
            status=ScientificGuardrailStatus.PASS,
            scope="efficacy:policy:neural_persistent",
            detail="focal action saturation is within the preregistered limit",
            observed_value=0.0,
            threshold=0.05,
        ),
    ]
    for comparator_id in ("best_static", "heuristic_adaptive", "random_matched"):
        guardrails.extend(
            (
                ScientificGuardrailResult(
                    name="behavioral_alias_detection",
                    status=ScientificGuardrailStatus.FAIL,
                    scope=f"efficacy:pair:neural_persistent:{comparator_id}",
                    detail="focal and comparator behavior are aliased within tolerance",
                    observed_value=0.0,
                    threshold=0.0,
                ),
                ScientificGuardrailResult(
                    name="instruction_adherence_non_regression",
                    status=ScientificGuardrailStatus.PASS,
                    scope=f"efficacy:pair:neural_persistent:{comparator_id}",
                    detail="matched adherence is within the non-regression margin",
                    observed_value=0.0,
                    threshold=-0.01,
                ),
                ScientificGuardrailResult(
                    name="response_length_confound",
                    status=ScientificGuardrailStatus.PASS,
                    scope=f"efficacy:pair:neural_persistent:{comparator_id}",
                    detail="no repetition-improving matched unit exceeds the shortening limit",
                    observed_value=0.0,
                    threshold=0.05,
                ),
            )
        )
    guardrails.extend(
        (
            ScientificGuardrailResult(
                name="causal_mechanism_validation",
                status=ScientificGuardrailStatus.PASS,
                scope="attribution:causal",
                detail="stored persistent/reset mechanism evidence passed the causal validator",
            ),
            ScientificGuardrailResult(
                name="intervention_turn_only_attribution",
                status=ScientificGuardrailStatus.PASS,
                scope="attribution:causal",
                detail=(
                    "turn zero is excluded and every attribution observation has turn_index > 0"
                ),
                observed_value=2.0,
                threshold=2.0,
            ),
            ScientificGuardrailResult(
                name="behavioral_alias_detection",
                status=ScientificGuardrailStatus.FAIL,
                scope=("attribution:pair:neural_persistent:neural_matched_history_state_reset"),
                detail="focal and comparator behavior are aliased within tolerance",
                observed_value=0.0,
                threshold=0.0,
            ),
            ScientificGuardrailResult(
                name="instruction_adherence_non_regression",
                status=ScientificGuardrailStatus.PASS,
                scope=("attribution:pair:neural_persistent:neural_matched_history_state_reset"),
                detail="matched adherence is within the non-regression margin",
                observed_value=0.0,
                threshold=-0.01,
            ),
            ScientificGuardrailResult(
                name="response_length_confound",
                status=ScientificGuardrailStatus.PASS,
                scope=("attribution:pair:neural_persistent:neural_matched_history_state_reset"),
                detail="no repetition-improving matched unit exceeds the shortening limit",
                observed_value=0.0,
                threshold=0.05,
            ),
        )
    )
    return tuple(sorted(guardrails, key=lambda guardrail: guardrail.evidence_key))


def _unit_gate_names(
    guardrails: tuple[ScientificGuardrailResult, ...],
    policy_id: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                guardrail.name
                for guardrail in guardrails
                if guardrail.scope == "efficacy:global"
                or guardrail.scope == f"efficacy:policy:{policy_id}"
                or (
                    guardrail.scope.startswith("efficacy:pair:")
                    and policy_id in guardrail.scope.split(":")[2:]
                )
            }
        )
    )


def _confirmatory_result() -> ConfirmatoryEvaluationResult:
    spec = _confirmatory_spec()
    guardrails = _guardrails()
    comparison_guardrails = {
        comparator_id: tuple(
            guardrail
            for guardrail in guardrails
            if guardrail.scope == "efficacy:global"
            or guardrail.scope == "efficacy:policy:neural_persistent"
            or guardrail.scope == f"efficacy:pair:neural_persistent:{comparator_id}"
        )
        for comparator_id in ("best_static", "heuristic_adaptive", "random_matched")
    }
    comparisons = evaluate_efficacy_comparisons(
        {
            "best_static": (0.0,),
            "heuristic_adaptive": (0.0,),
            "random_matched": (0.0,),
        },
        spec=spec.efficacy,
        negative_multiplicity=spec.validated_negative_multiplicity,
        guardrails_by_comparator=comparison_guardrails,
        behavioral_alias_by_comparator={
            comparator_id: True
            for comparator_id in ("best_static", "heuristic_adaptive", "random_matched")
        },
    )
    recovery_unit_outcomes = (
        RecoveryUnitOutcome(
            unit_key=MatchedUnitKey(prompt_sequence_id="sequence-a", model_seed=7),
            post_stressor_task_score_change=0.0,
            post_stressor_repetition_change=0.0,
            time_to_return_to_target_band=0.0,
            focal_right_censored=False,
            comparator_right_censored_count=0,
        ),
    )
    recovery = evaluate_recovery(
        {metric_name: (0.0,) for metric_name in RECOVERY_METRIC_NAMES},
        spec=spec.recovery,
        negative_multiplicity=spec.validated_negative_multiplicity,
    )
    attribution_unit_outcomes = (
        AttributionUnitOutcome(
            unit_key=MatchedUnitKey(prompt_sequence_id="sequence-a", model_seed=7),
            persistent_minus_reset_task_score=0.0,
        ),
    )
    attribution_guardrails = tuple(
        guardrail for guardrail in guardrails if guardrail.scope.startswith("attribution:")
    )
    attribution = evaluate_persistent_state_attribution(
        (0.0,),
        spec=spec.attribution,
        negative_multiplicity=spec.validated_negative_multiplicity,
        causal_guardrails=attribution_guardrails,
        behavioral_alias=True,
    )
    limitations = (
        ScientificLimitation(
            kind=LimitationKind.OPTIONAL_METRIC_UNAVAILABLE,
            code="optional_metric_unavailable_semantic_similarity",
            detail="semantic_similarity available for 0 of 10 committed turns",
            disposition=LimitationDisposition.DISCLOSURE_ONLY,
        ),
    )
    decision_input = ScientificDecisionInput(
        tier=ExperimentTier.CONFIRMATORY,
        efficacy_comparisons=comparisons,
        recovery=recovery.decision_gate,
        attribution=attribution.decision_gate,
        guardrails=guardrails,
        limitations=limitations,
    )
    decision = decide_scientific_outcome(decision_input)
    assert decision.decision is ScientificDecisionState.VALIDATED_NEGATIVE
    payload = {
        "schema_version": 2,
        "implementation_version": "confirmatory-evaluation-v2",
        "claim_scope": "confirmatory-model-backed-scientific-decision",
        "analysis_contract_sha256": canonical_sha256("provider-free-analysis-contract"),
        "confirmatory_analysis_spec": spec,
        "confirmatory_analysis_spec_sha256": canonical_sha256(spec),
        "prompt_family_by_sequence": {"sequence-a": "family-a"},
        "prompt_family_design_sha256": canonical_sha256({"sequence-a": "family-a"}),
        "validated_negative_multiplicity_sha256": canonical_sha256(
            DEFAULT_VALIDATED_NEGATIVE_MULTIPLICITY
        ),
        "causal_mechanism_validated": True,
        "claim_eligible": False,
        "run_manifest_sha256": None,
        "run_finalization_sha256": None,
        "input_sha256": canonical_sha256("confirmatory-input"),
        "coverage": CoverageResult(exact=True, expected_count=10, observed_count=10),
        "optional_metric_availability": {"semantic_similarity": (0, 10)},
        "unit_outcomes": tuple(
            ScientificUnitOutcome(
                unit_key=MatchedUnitKey(prompt_sequence_id="sequence-a", model_seed=7),
                prompt_family="family-a",
                policy_id=policy_id,
                guardrail_clean_task_score=GuardrailCleanTaskScore(
                    raw_task_score=1.0,
                    gate_status=ScientificGuardrailStatus.FAIL,
                    gate_names=_unit_gate_names(guardrails, policy_id),
                ),
                instruction_adherence=1.0,
                repetition_ratio=0.0,
                response_length_tokens=8.0,
            )
            for policy_id in (
                "best_static",
                "heuristic_adaptive",
                "neural_persistent",
                "random_matched",
            )
        ),
        "recovery_unit_outcomes": recovery_unit_outcomes,
        "attribution_unit_outcomes": attribution_unit_outcomes,
        "efficacy_comparisons": comparisons,
        "recovery": recovery,
        "attribution": attribution,
        "subgroup_effects": (),
        "guardrails": guardrails,
        "limitations": limitations,
        "decision": decision,
        "statistics_call_count": 18,
    }
    return ConfirmatoryEvaluationResult.model_validate(
        {**payload, "result_sha256": confirmatory_result_sha256(payload)}
    )


def _validated_negative_result() -> ConfirmatoryEvaluationResult:
    """Build complete adjusted negative evidence for export/persistence coverage."""

    return _confirmatory_result()


def _claim_bound_result(
    run_manifest: RunManifest,
    run_finalization: RunFinalization,
    *,
    base: ConfirmatoryEvaluationResult | None = None,
) -> tuple[ConfirmatoryEvaluationResult, ConfirmatoryAnalysisContext]:
    """Bind the compact result fixture to one exact closed-run authority context."""

    analysis_contract_sha256 = run_manifest.confirmatory_analysis_contract_sha256
    assert analysis_contract_sha256 is not None
    run_manifest_sha256 = canonical_sha256(run_manifest)
    run_finalization_sha256 = canonical_sha256(run_finalization)
    input_sha256 = canonical_sha256(
        {
            "implementation_version": "claim-bound-storage-fixture-v1",
            "analysis_contract_sha256": analysis_contract_sha256,
            "run_manifest_sha256": run_manifest_sha256,
            "run_finalization_sha256": run_finalization_sha256,
        }
    )
    source = _confirmatory_result() if base is None else base
    payload = source.model_dump(mode="python", exclude={"result_sha256"})
    payload.update(
        {
            "analysis_contract_sha256": analysis_contract_sha256,
            "claim_eligible": True,
            "run_manifest_sha256": run_manifest_sha256,
            "run_finalization_sha256": run_finalization_sha256,
            "input_sha256": input_sha256,
        }
    )
    result = ConfirmatoryEvaluationResult.model_validate(
        {**payload, "result_sha256": confirmatory_result_sha256(payload)}
    )
    context = ConfirmatoryAnalysisContext(
        analysis_contract_sha256=analysis_contract_sha256,
        evaluation_input_sha256=input_sha256,
        causal_mechanism_validated=True,
        claim_eligible=True,
        run_manifest_sha256=run_manifest_sha256,
        run_finalization_sha256=run_finalization_sha256,
    )
    return result, context


def _provider_free_context(result: ConfirmatoryEvaluationResult) -> ConfirmatoryAnalysisContext:
    return ConfirmatoryAnalysisContext(
        analysis_contract_sha256=result.analysis_contract_sha256,
        evaluation_input_sha256=result.input_sha256,
        causal_mechanism_validated=True,
        claim_eligible=False,
        run_manifest_sha256=None,
        run_finalization_sha256=None,
    )


def _complete_llama_requests(
    store: SQLiteRunStore,
    identity: ProviderIdentity,
    *,
    prompt_family: str | None = "family-a",
    prompt_features: PromptFeatures | None = None,
    validator: ValidatorSpec | None = None,
    metric_validator: ValidatorSpec | None = None,
) -> tuple[str, ...]:
    condition_ids: list[str] = []
    features = prompt_features or PromptFeatures({})
    frozen_validator = validator or ValidatorSpec(kind="non_empty")
    frozen_metric_validator = metric_validator or frozen_validator
    previous_condition_ids: dict[str, str] = {}
    for turn_index in (0, 1):
        for policy_id in _SCIENTIFIC_POLICY_IDS:
            request = make_request(identity, policy_id=policy_id, turn_index=turn_index)
            history = None
            if turn_index > 0:
                source_policy_id = (
                    "neural_persistent"
                    if policy_id == "neural_matched_history_state_reset"
                    else policy_id
                )
                history = store.history_binding_for(previous_condition_ids[source_policy_id])
            input_evidence = (
                None
                if prompt_family is None
                else TurnInputEvidence(
                    condition_id=request.condition_id,
                    prompt_case_id=f"case-{turn_index}",
                    prompt_family=prompt_family,
                    prompt_features=features,
                    validator=frozen_validator,
                )
            )
            store.prepare_turn(request, history, input_evidence)
            store.begin_dispatch(request.condition_id)
            response = GenerationResponse(
                text="one two three four five six seven eight",
                provider_identity=identity,
                effective_parameters=request.decoding_parameters,
                raw_metadata=GenerationMetadata(request_sha256=canonical_sha256(request)),
            )
            store.persist_response(request.condition_id, response)
            metric_prompt_family = prompt_family or "family-a"
            store.persist_metrics(
                request.condition_id,
                compute_response_metrics(
                    MetricContext(
                        prompt_case_id=f"case-{turn_index}",
                        prompt_family=metric_prompt_family,
                        prompt=request.prompt,
                        response_text=response.text,
                        validator=frozen_metric_validator,
                    )
                ),
            )
            policy_trace = make_trace(request)
            store.commit_turn(
                request.condition_id,
                PolicyState(),
                DetailedAppliedPolicyTrace(
                    policy_id=policy_id,
                    turn_index=turn_index,
                    action=policy_trace.action,
                    action_application=apply_action(
                        request.decoding_parameters,
                        policy_trace.action,
                        ActionBounds(),
                        DecodingBounds(),
                    ),
                    history_access=("none" if turn_index == 0 else "own_previous_response"),
                    observation_has_previous_response=turn_index > 0,
                    policy_trace=policy_trace,
                ),
            )
            condition_ids.append(request.condition_id)
            previous_condition_ids[policy_id] = request.condition_id
    return tuple(sorted(condition_ids))


def _complete_accounting(condition_count: int) -> DurableExecutionAccounting:
    return DurableExecutionAccounting(
        planned_logical_generations=condition_count,
        dispatched_logical_generations=condition_count,
        successful_responses=condition_count,
        uncertain_dispatches=0,
        committed_logical_generations=condition_count,
    )


def _scientific_manifest(
    run_manifest: RunManifest,
    run_finalization: RunFinalization,
    result: ConfirmatoryEvaluationResult,
) -> ScientificAnalysisManifest:
    spec = _confirmatory_spec()
    assert run_manifest.scientific_identity_sha256 is not None
    assert run_manifest.preregistration_sha256 is not None
    assert run_manifest.confirmatory_analysis_contract_sha256 is not None
    return ScientificAnalysisManifest(
        run_manifest_sha256=canonical_sha256(run_manifest),
        run_finalization_sha256=canonical_sha256(run_finalization),
        scientific_result_sha256=run_finalization.scientific_result_sha256,
        scientific_identity_sha256=run_manifest.scientific_identity_sha256,
        preregistration_sha256=run_manifest.preregistration_sha256,
        confirmatory_analysis_contract_sha256=(run_manifest.confirmatory_analysis_contract_sha256),
        confirmatory_analysis_spec=spec,
        confirmatory_analysis_spec_sha256=canonical_sha256(spec),
        prompt_family_by_sequence=result.prompt_family_by_sequence,
        prompt_family_design_sha256=result.prompt_family_design_sha256,
        dataset_sha256=run_manifest.dataset_hash,
        dataset_purpose=DatasetPurpose.EVALUATION,
        dataset_seal_sha256=canonical_sha256("confirmatory-dataset-seal"),
        evaluation_input_sha256=result.input_sha256,
    )


def test_scientific_analysis_is_atomic_typed_idempotent_and_reopenable(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    database = tmp_path / "run.sqlite3"

    with SQLiteRunStore(database, run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
        manifest = _scientific_manifest(run_manifest, run_finalization, result)

        first = store.persist_scientific_analysis(manifest, result, context=context)
        assert store.persist_scientific_analysis(manifest, result, context=context) == first
        assert store.get_analysis() is None
        stored = store.get_scientific_analysis()
        assert stored is not None
        assert stored.manifest == manifest
        assert stored.result == result
        assert stored.efficacy_comparisons == result.efficacy_comparisons
        assert stored.attribution == result.attribution
        assert stored.guardrails == _guardrails()
        assert stored.finalization == first
        assert first.comparison_count == 4
        assert first.efficacy_comparison_count == 3
        assert first.attribution_comparison_count == 1
        assert first.guardrail_count == 20

        comparison_rows = store._connection.execute(
            "SELECT result_json FROM comparison_results"
        ).fetchall()
        comparison_kinds = sorted(json.loads(row[0])["comparison_kind"] for row in comparison_rows)
        assert comparison_kinds == [
            "efficacy",
            "efficacy",
            "efficacy",
            "persistent_state_attribution",
        ]
        decision_json = store._connection.execute(
            "SELECT decision_json FROM analysis_decision WHERE singleton_id = 1"
        ).fetchone()[0]
        assert '"recovery"' in decision_json
        assert all("recovery" not in json.loads(row[0]) for row in comparison_rows)
        store.verify_integrity()

    with SQLiteRunStore(database) as reopened:
        assert reopened.get_analysis() is None
        assert reopened.get_scientific_analysis() == stored
        reopened.verify_integrity()


@pytest.mark.parametrize(
    ("prompt_family", "message"),
    (
        (None, "exact prompt-side input evidence coverage"),
        ("family-b", "frozen run identity"),
    ),
)
def test_scientific_analysis_requires_reconstructable_prompt_family_evidence(
    tmp_path: Path,
    prompt_family: str | None,
    message: str,
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(
            store,
            run_manifest.provider_identity,
            prompt_family=prompt_family,
        )
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
        manifest = _scientific_manifest(run_manifest, run_finalization, result)

        with pytest.raises(StoreInvariantError, match=message):
            store.persist_scientific_analysis(manifest, result, context=context)

        assert store.get_scientific_analysis() is None


def test_scientific_analysis_rejects_prompt_features_outside_frozen_identity(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(
            store,
            run_manifest.provider_identity,
            prompt_features=PromptFeatures({"unbound_feature": 1.0}),
        )
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)

        with pytest.raises(StoreInvariantError, match="frozen run identity"):
            store.persist_scientific_analysis(
                _scientific_manifest(run_manifest, run_finalization, result),
                result,
                context=context,
            )


def test_scientific_analysis_recomputes_metrics_from_frozen_validator(
    tmp_path: Path,
) -> None:
    validator = ValidatorSpec(kind="exact_match", expected_text="never matches")
    run_manifest = _run_manifest(validator=validator)
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(
            store,
            run_manifest.provider_identity,
            validator=validator,
            metric_validator=ValidatorSpec(kind="non_empty"),
        )
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)

        with pytest.raises(StoreInvariantError, match="metrics do not reconstruct"):
            store.persist_scientific_analysis(
                _scientific_manifest(run_manifest, run_finalization, result),
                result,
                context=context,
            )


def test_confirmatory_result_rejects_forged_positive_decision() -> None:
    valid = _confirmatory_result()
    forged_decision = ScientificDecisionRecord(
        decision=ScientificDecisionState.VALIDATED_POSITIVE,
        reason_codes=(ScientificReasonCode.ALL_POSITIVE_GATES_PASSED,),
        decision_input_sha256=valid.decision.decision_input_sha256,
    )
    payload = valid.model_dump(mode="python", exclude={"result_sha256"})
    payload["decision"] = forged_decision

    with pytest.raises(ValueError, match="does not match the enclosed evidence"):
        ConfirmatoryEvaluationResult.model_validate(
            {**payload, "result_sha256": confirmatory_result_sha256(payload)}
        )


def test_scientific_analysis_rejects_provider_free_authority_before_write(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    provider_free_result = _confirmatory_result()
    provider_free_context = _provider_free_context(provider_free_result)
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        manifest = _scientific_manifest(run_manifest, finalization, provider_free_result)

        with pytest.raises(StoreInvariantError, match="claim-eligible closed-run evidence"):
            store.persist_scientific_analysis(
                manifest,
                provider_free_result,
                context=provider_free_context,
            )

        assert store.get_analysis() is None
        assert store.get_scientific_analysis() is None


def test_scientific_analysis_rejects_wrong_finalized_aggregate_before_write(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            canonical_sha256("wrong-aggregate"),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, finalization)
        manifest = _scientific_manifest(run_manifest, finalization, result)

        with pytest.raises(StoreInvariantError, match="does not match the committed result"):
            store.persist_scientific_analysis(manifest, result, context=context)

        assert store.get_analysis() is None
        assert store.get_scientific_analysis() is None


@pytest.mark.parametrize(
    "context_mutation",
    (
        {"claim_eligible": False},
        {"causal_mechanism_validated": False},
        {"analysis_contract_sha256": "1" * 64},
        {"evaluation_input_sha256": "2" * 64},
        {"run_manifest_sha256": None},
        {"run_finalization_sha256": "3" * 64},
    ),
)
def test_scientific_analysis_rejects_drifted_claim_context_before_write(
    tmp_path: Path,
    context_mutation: dict[str, object],
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, finalization)
        manifest = _scientific_manifest(run_manifest, finalization, result)

        with pytest.raises(StoreInvariantError, match="claim-eligible run binding"):
            store.persist_scientific_analysis(
                manifest,
                result,
                context=context.model_copy(update=context_mutation),
            )

        assert store.get_analysis() is None
        assert store.get_scientific_analysis() is None


def test_scientific_export_has_exact_typed_comparisons_and_final_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise derived artifacts over already validated causal/store evidence."""

    run_manifest = _run_manifest()
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database, run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(
            run_manifest,
            run_finalization,
            base=_validated_negative_result(),
        )
        store.persist_scientific_analysis(
            _scientific_manifest(run_manifest, run_finalization, result),
            result,
            context=context,
        )

    causal_validation_calls: list[int] = []

    def record_causal_validation(
        _manifest: object,
        turns: tuple[object, ...],
        _inputs: tuple[object, ...],
    ) -> None:
        causal_validation_calls.append(len(turns))

    monkeypatch.setattr(
        "neurallm.reporting.artifacts._validate_phase4_mechanism_evidence",
        record_causal_validation,
    )
    first = export_closed_run(tmp_path)
    first_contents = {
        name: (tmp_path / name).read_bytes()
        for name in first.artifact_names
        if name != "run.sqlite3"
    }
    repeated = export_closed_run(tmp_path)

    assert first == repeated
    assert causal_validation_calls == [10, 10]
    assert first.implementation_phase == 5
    assert first.scientific_decision == "VALIDATED_NEGATIVE"
    assert first.phase3_baseline_evaluator_verdict is None
    assert set(first.artifact_names) == {
        "run.sqlite3",
        "manifest.json",
        "results.csv",
        "comparisons.csv",
        "decision.json",
        "report.md",
    }
    assert first_contents == {name: (tmp_path / name).read_bytes() for name in first_contents}

    with (tmp_path / "comparisons.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        comparison_rows = tuple(reader)
        comparison_fields = tuple(reader.fieldnames or ())
    assert comparison_fields == (
        "comparison_id",
        "comparison_kind",
        "focal_policy_id",
        "comparator_policy_id",
        "comparator_role",
        "attribution_only",
        "included_in_efficacy",
        "included_in_holm_family",
        "primary_metric",
        "unit_count",
        "mean_difference",
        "bootstrap_lower",
        "bootstrap_upper",
        "bootstrap_resamples",
        "bootstrap_seed",
        "negative_multiplicity_method",
        "negative_familywise_alpha",
        "negative_family_size",
        "negative_confidence_level",
        "negative_bootstrap_lower",
        "negative_bootstrap_upper",
        "negative_bootstrap_resamples",
        "negative_bootstrap_seed",
        "negative_decisive",
        "permutation_p_value",
        "permutation_exact",
        "permutation_count",
        "permutation_seed",
        "holm_adjusted_p_value",
        "practical_effect_threshold",
        "behavioral_alias",
        "guardrail_statuses",
        "status",
        "detail",
    )
    negative_fields = tuple(field for field in comparison_fields if field.startswith("negative_"))
    assert negative_fields == (
        "negative_multiplicity_method",
        "negative_familywise_alpha",
        "negative_family_size",
        "negative_confidence_level",
        "negative_bootstrap_lower",
        "negative_bootstrap_upper",
        "negative_bootstrap_resamples",
        "negative_bootstrap_seed",
        "negative_decisive",
    )
    assert not set(negative_fields).intersection(reporting_artifacts._PHASE2_COMPARISON_FIELDS)
    assert not set(negative_fields).intersection(reporting_artifacts._PHASE3_COMPARISON_FIELDS)
    assert len(comparison_rows) == 4
    assert [row["comparison_kind"] for row in comparison_rows] == [
        "efficacy",
        "efficacy",
        "efficacy",
        "persistent_state_attribution",
    ]
    assert [row["included_in_efficacy"] for row in comparison_rows] == [
        "True",
        "True",
        "True",
        "False",
    ]
    assert [row["negative_multiplicity_method"] for row in comparison_rows] == [
        "bonferroni-simultaneous-bootstrap-v1",
        "bonferroni-simultaneous-bootstrap-v1",
        "bonferroni-simultaneous-bootstrap-v1",
        "bonferroni-simultaneous-bootstrap-v1",
    ]
    assert [row["negative_family_size"] for row in comparison_rows] == ["7"] * 4
    assert [row["negative_decisive"] for row in comparison_rows] == [
        "True",
        "True",
        "True",
        "False",
    ]

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["schema_version"] == 2
    assert decision["scientific_decision"] == "VALIDATED_NEGATIVE"
    assert decision["confirmatory_analysis_spec_sha256"] == canonical_sha256(
        result.confirmatory_analysis_spec
    )
    assert decision["confirmatory_analysis_spec"] == result.confirmatory_analysis_spec.model_dump(
        mode="json"
    )
    assert decision["subgroup_effects"] == []
    assert decision["statistics_call_count"] == 18
    assert decision["reason_codes"] == [
        "attribution_failed",
        "guardrail_failed",
        "negative_control_sanity_failed",
        "recovery_failed",
        "required_comparator_failed",
    ]
    assert decision["execution_accounting"] == {
        "planned_logical_generations": 10,
        "dispatched_logical_generations": 10,
        "successful_responses": 10,
        "uncertain_dispatches": 0,
        "committed_logical_generations": 10,
    }

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    headings = tuple(line for line in report.splitlines() if line.startswith("## "))
    assert headings == (
        "## Engineering validity",
        "## Controller activity",
        "## End-to-end efficacy",
        "## Persistent-state attribution",
        "## Guardrail outcomes",
        "## Limitations",
        "## Final decision",
    )
    assert report.count("adjusted negative-side evidence") == 7
    assert report.count("`bonferroni-simultaneous-bootstrap-v1`") == 7
    assert report.count("familywise alpha `0.050000` across `7` gates") == 7
    assert report.count("adjusted two-sided confidence `0.992857`") == 7
    assert report.count("decisive negative `true`") == 6
    assert "adjusted negative-side evidence `efficacy:best_static`" in report
    assert "adjusted negative-side evidence `recovery:post_stressor_task_score_change`" in report
    assert (
        "adjusted negative-side evidence `attribution:neural_matched_history_state_reset`" in report
    )
    assert "## Final decision\n\n`VALIDATED_NEGATIVE`. Reason codes:" in report


def test_scientific_export_invokes_causal_gate_and_fails_closed(
    tmp_path: Path,
) -> None:
    """A persisted decision cannot bypass incomplete causal mechanism evidence."""

    run_manifest = _run_manifest()
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database, run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
        store.persist_scientific_analysis(
            _scientific_manifest(run_manifest, run_finalization, result),
            result,
            context=context,
        )
        store._connection.execute("DELETE FROM turn_inputs")
        store._connection.commit()

    with pytest.raises(StoreCorruptionError, match="prompt-side input evidence coverage"):
        export_closed_run(tmp_path)

    assert not (tmp_path / "manifest.json").exists()


def test_scientific_analysis_rejects_foreign_bindings_before_any_write(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
        manifest = _scientific_manifest(run_manifest, run_finalization, result)
        drifted_spec = manifest.confirmatory_analysis_spec.model_copy(
            update={"subgroup_fields": ("unexpected_subgroup",)}
        )
        mutations = (
            {"run_manifest_sha256": "1" * 64},
            {"run_finalization_sha256": "2" * 64},
            {"scientific_result_sha256": "3" * 64},
            {"scientific_identity_sha256": "4" * 64},
            {"preregistration_sha256": "5" * 64},
            {"confirmatory_analysis_contract_sha256": "8" * 64},
            {"confirmatory_analysis_spec_sha256": "9" * 64},
            {
                "confirmatory_analysis_spec": drifted_spec,
                "confirmatory_analysis_spec_sha256": canonical_sha256(drifted_spec),
            },
            {"dataset_sha256": "6" * 64},
            {"dataset_purpose": DatasetPurpose.DEVELOPMENT},
            {"dataset_seal_sha256": "a" * 64},
            {"evaluation_input_sha256": "7" * 64},
        )
        for mutation in mutations:
            with pytest.raises(StoreInvariantError):
                store.persist_scientific_analysis(
                    manifest.model_copy(update=mutation),
                    result,
                    context=context,
                )
            assert store.get_analysis() is None
            assert store.get_scientific_analysis() is None

        drifted_result_spec = result.confirmatory_analysis_spec.model_copy(
            update={
                "optional_metric_dispositions": {
                    "foreign_metric": LimitationDisposition.DISCLOSURE_ONLY
                }
            }
        )
        foreign_limitation = ScientificLimitation(
            kind=LimitationKind.OPTIONAL_METRIC_UNAVAILABLE,
            code="optional_metric_unavailable_foreign_metric",
            detail="foreign_metric available for 0 of 10 committed turns",
            disposition=LimitationDisposition.DISCLOSURE_ONLY,
        )
        drifted_decision = decide_scientific_outcome(
            ScientificDecisionInput(
                tier=ExperimentTier.CONFIRMATORY,
                efficacy_comparisons=result.efficacy_comparisons,
                recovery=result.recovery.decision_gate,
                attribution=result.attribution.decision_gate,
                guardrails=result.guardrails,
                limitations=(foreign_limitation,),
            )
        )
        drifted_result_payload = {
            **result.model_dump(mode="python", exclude={"result_sha256"}),
            "confirmatory_analysis_spec": drifted_result_spec,
            "confirmatory_analysis_spec_sha256": canonical_sha256(drifted_result_spec),
            "optional_metric_availability": {"foreign_metric": (0, 10)},
            "limitations": (foreign_limitation,),
            "decision": drifted_decision,
        }
        drifted_result = ConfirmatoryEvaluationResult.model_validate(
            {
                **drifted_result_payload,
                "result_sha256": confirmatory_result_sha256(drifted_result_payload),
            }
        )
        with pytest.raises(StoreInvariantError, match="preregistered analysis design"):
            store.persist_scientific_analysis(
                manifest,
                drifted_result,
                context=context,
            )
        assert store.get_scientific_analysis() is None


def test_scientific_analysis_recomputes_guardrails_from_committed_source(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
        forged_guardrails = tuple(
            guardrail.model_copy(update={"status": ScientificGuardrailStatus.FAIL})
            if guardrail.name == "action_saturation_rate"
            else guardrail
            for guardrail in result.guardrails
        )
        comparison_guardrails = {
            comparator_id: tuple(
                guardrail
                for guardrail in forged_guardrails
                if guardrail.scope == "efficacy:global"
                or guardrail.scope == "efficacy:policy:neural_persistent"
                or guardrail.scope == f"efficacy:pair:neural_persistent:{comparator_id}"
            )
            for comparator_id in ("best_static", "heuristic_adaptive", "random_matched")
        }
        spec = result.confirmatory_analysis_spec
        forged_comparisons = evaluate_efficacy_comparisons(
            {
                "best_static": (0.0,),
                "heuristic_adaptive": (0.0,),
                "random_matched": (0.0,),
            },
            spec=spec.efficacy,
            negative_multiplicity=spec.validated_negative_multiplicity,
            guardrails_by_comparator=comparison_guardrails,
            behavioral_alias_by_comparator={
                "best_static": True,
                "heuristic_adaptive": True,
                "random_matched": True,
            },
        )
        forged_decision = decide_scientific_outcome(
            ScientificDecisionInput(
                tier=ExperimentTier.CONFIRMATORY,
                efficacy_comparisons=forged_comparisons,
                recovery=result.recovery.decision_gate,
                attribution=result.attribution.decision_gate,
                guardrails=forged_guardrails,
                limitations=result.limitations,
            )
        )
        forged_payload = {
            **result.model_dump(mode="python", exclude={"result_sha256"}),
            "efficacy_comparisons": forged_comparisons,
            "guardrails": forged_guardrails,
            "decision": forged_decision,
        }
        forged = ConfirmatoryEvaluationResult.model_validate(
            {
                **forged_payload,
                "result_sha256": confirmatory_result_sha256(forged_payload),
            }
        )
        with pytest.raises(StoreInvariantError, match="committed source evidence"):
            store.persist_scientific_analysis(
                _scientific_manifest(run_manifest, run_finalization, forged),
                forged,
                context=context,
            )
        assert store.get_scientific_analysis() is None


def test_scientific_analysis_requires_finalized_confirmatory_llama_run(
    tmp_path: Path,
) -> None:
    valid_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "open.sqlite3", valid_manifest) as store:
        condition_ids = _complete_llama_requests(store, valid_manifest.provider_identity)
        placeholder_finalization = RunFinalization(
            expected_condition_ids=condition_ids,
            expected_condition_count=len(condition_ids),
            manifest_sha256=canonical_sha256(valid_manifest),
            scientific_result_sha256=canonical_sha256("not-finalized"),
        )
        result, context = _claim_bound_result(valid_manifest, placeholder_finalization)
        scientific_manifest = _scientific_manifest(
            valid_manifest,
            placeholder_finalization,
            result,
        )
        with pytest.raises(StoreInvariantError, match="finalized confirmatory run"):
            store.persist_scientific_analysis(
                scientific_manifest,
                result,
                context=context,
            )

    with SQLiteRunStore(tmp_path / "unaccounted.sqlite3", valid_manifest) as store:
        condition_ids = _complete_llama_requests(store, valid_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
        )
        result, context = _claim_bound_result(valid_manifest, finalization)
        with pytest.raises(StoreInvariantError, match="durable execution accounting"):
            store.persist_scientific_analysis(
                _scientific_manifest(valid_manifest, finalization, result),
                result,
                context=context,
            )

    dirty_manifest = valid_manifest.model_copy(update={"working_tree_clean": False})
    with SQLiteRunStore(tmp_path / "dirty.sqlite3", dirty_manifest) as store:
        condition_ids = _complete_llama_requests(store, dirty_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(dirty_manifest, finalization)
        with pytest.raises(StoreInvariantError, match="confirmatory run manifest"):
            store.persist_scientific_analysis(
                _scientific_manifest(dirty_manifest, finalization, result),
                result,
                context=context,
            )

    wrong_contract_manifest = valid_manifest.model_copy(
        update={"confirmatory_analysis_contract_sha256": "b" * 64}
    )
    with SQLiteRunStore(
        tmp_path / "wrong-contract.sqlite3",
        wrong_contract_manifest,
    ) as store:
        condition_ids = _complete_llama_requests(
            store,
            wrong_contract_manifest.provider_identity,
        )
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(wrong_contract_manifest, finalization)
        with pytest.raises(StoreInvariantError, match="pre-execution confirmatory"):
            store.persist_scientific_analysis(
                _scientific_manifest(wrong_contract_manifest, finalization, result),
                result,
                context=context,
            )

    fake = FakeProvider()
    fake_manifest = _run_manifest(
        identity=fake.provider_identity,
        effective_json=fake.effective_configuration_json,
    )
    with SQLiteRunStore(tmp_path / "fake.sqlite3", fake_manifest) as store:
        request = make_request(fake.provider_identity, policy_id="neural_persistent")
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        response = fake.generate(request)
        store.persist_response(request.condition_id, response)
        store.persist_metrics(request.condition_id, make_metrics(response))
        store.commit_turn(request.condition_id, PolicyState(), make_trace(request))
        finalization = store.finalize_run(
            (request.condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
        )
        result, context = _claim_bound_result(fake_manifest, finalization)
        with pytest.raises(StoreInvariantError, match="llama_cpp"):
            store.persist_scientific_analysis(
                _scientific_manifest(fake_manifest, finalization, result),
                result,
                context=context,
            )

    digestless_identity = valid_manifest.provider_identity.model_copy(update={"model_sha256": None})
    digestless_manifest = _run_manifest(
        identity=digestless_identity,
        effective_json=valid_manifest.provider_effective_configuration_json,
    )
    with SQLiteRunStore(tmp_path / "digestless.sqlite3", digestless_manifest) as store:
        condition_ids = _complete_llama_requests(store, digestless_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(digestless_manifest, finalization)
        with pytest.raises(StoreInvariantError, match="digest-bound llama_cpp"):
            store.persist_scientific_analysis(
                _scientific_manifest(digestless_manifest, finalization, result),
                result,
                context=context,
            )

    drifted_digest_identity = valid_manifest.provider_identity.model_copy(
        update={"model_sha256": "f" * 64}
    )
    drifted_digest_manifest = _run_manifest(
        identity=drifted_digest_identity,
        effective_json=valid_manifest.provider_effective_configuration_json,
    )
    with SQLiteRunStore(tmp_path / "drifted-digest.sqlite3", drifted_digest_manifest) as store:
        condition_ids = _complete_llama_requests(store, drifted_digest_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(drifted_digest_manifest, finalization)
        with pytest.raises(StoreInvariantError, match="internally consistent digest-bound"):
            store.persist_scientific_analysis(
                _scientific_manifest(drifted_digest_manifest, finalization, result),
                result,
                context=context,
            )

    pilot_manifest = _run_manifest(
        decision_rule_version="development-pilot-no-scientific-decision-v1",
        run_tier="development_pilot",
        preregistration_sha256=None,
    )
    with SQLiteRunStore(tmp_path / "pilot.sqlite3", pilot_manifest) as store:
        condition_ids = _complete_llama_requests(store, pilot_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        promoted_pilot_manifest = pilot_manifest.model_copy(
            update={
                "preregistration_sha256": canonical_sha256("foreign-seal"),
                "confirmatory_analysis_contract_sha256": canonical_sha256("foreign-contract"),
            }
        )
        result, context = _claim_bound_result(promoted_pilot_manifest, finalization)
        scientific_manifest = _scientific_manifest(
            promoted_pilot_manifest,
            finalization,
            result,
        )
        with pytest.raises(StoreInvariantError, match="confirmatory run manifest"):
            store.persist_scientific_analysis(
                scientific_manifest,
                result,
                context=context,
            )


def test_unknown_scientific_manifest_discriminant_fails_integrity(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database, run_manifest) as store:
        condition_ids = _complete_llama_requests(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            condition_ids,
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(len(condition_ids)),
        )
        result, context = _claim_bound_result(run_manifest, finalization)
        store.persist_scientific_analysis(
            _scientific_manifest(run_manifest, finalization, result),
            result,
            context=context,
        )

    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            "SELECT manifest_json FROM analysis_manifest WHERE singleton_id = 1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["implementation_version"] = "unknown-analysis-storage-v99"
        connection.execute(
            "UPDATE analysis_manifest SET manifest_json = ?, manifest_sha256 = ?",
            (canonical_json(payload), canonical_sha256(payload)),
        )

    with SQLiteRunStore(database) as reopened:
        with pytest.raises(StoreCorruptionError, match="unknown implementation_version"):
            reopened.get_analysis()
        with pytest.raises(StoreCorruptionError, match="unknown implementation_version"):
            reopened.verify_integrity()

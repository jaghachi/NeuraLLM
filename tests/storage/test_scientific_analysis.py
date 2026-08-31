"""Atomic schema-v2 persistence for confirmatory scientific evidence."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Literal

import pytest

from neurallm.control.policy import PolicyState
from neurallm.domain.models import (
    ActionBounds,
    ProviderIdentity,
    RunManifest,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import CoverageResult, DatasetPurpose, MatchedUnitKey
from neurallm.evaluation.attribution import (
    AttributionAnalysisSpec,
    PersistentStateAttributionResult,
)
from neurallm.evaluation.confirmatory import (
    ConfirmatoryAnalysisSpec,
    ConfirmatoryEvaluationResult,
    RecoveryEventSpec,
    ScientificUnitOutcome,
    confirmatory_result_sha256,
)
from neurallm.evaluation.recovery import (
    RECOVERY_METRIC_NAMES,
    RecoveryAnalysisSpec,
    RecoveryEvaluationResult,
)
from neurallm.evaluation.scientific import (
    REQUIRED_SCIENTIFIC_GUARDRAILS,
    ComparatorRole,
    EfficacyAnalysisSpec,
    EfficacyComparisonResult,
    ExperimentTier,
    GuardrailCleanTaskScore,
    LimitationDisposition,
    ScientificDecisionInput,
    ScientificDecisionRecord,
    ScientificDecisionState,
    ScientificEvidenceStatus,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
    ScientificReasonCode,
    decide_scientific_outcome,
)
from neurallm.experiments.scientific_analysis import (
    ConfirmatoryAnalysisContext,
    confirmatory_analysis_contract_sha256,
)
from neurallm.providers.base import GenerationMetadata, GenerationResponse
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import export_closed_run, scientific_result_sha256
from neurallm.storage import (
    CURRENT_SCHEMA_VERSION,
    DurableExecutionAccounting,
    RunFinalization,
    ScientificAnalysisManifest,
    SQLiteRunStore,
    StoreCorruptionError,
    StoreInvariantError,
)
from tests.storage.helpers import make_metrics, make_request, make_trace


def _llama_identity_and_effective_json() -> tuple[ProviderIdentity, str]:
    effective = {
        "build_id": "test-build",
        "model_alias": "test-model",
        "model_path": "C:/models/test.gguf",
        "provider_type": "llama_cpp",
    }
    effective_json = canonical_json(effective)
    return (
        ProviderIdentity(
            provider_type="llama_cpp",
            implementation_version="llama-cpp-completion-http-v1",
            model_alias="test-model",
            build_id="test-build",
            provider_config_hash=canonical_sha256(effective),
            model_path="C:/models/test.gguf",
            chat_template_sha256="c" * 64,
        ),
        effective_json,
    )


def _run_manifest(
    *,
    identity: ProviderIdentity | None = None,
    effective_json: str | None = None,
    decision_rule_version: str = "confirmatory-scientific-decision-v1",
    run_tier: str = "confirmatory",
    preregistration_sha256: str | None = None,
) -> RunManifest:
    if identity is None or effective_json is None:
        identity, effective_json = _llama_identity_and_effective_json()
    if preregistration_sha256 is None and run_tier == "confirmatory":
        preregistration_sha256 = canonical_sha256("preregistration")
    scientific_identity_sha256 = canonical_sha256("confirmatory-plan")
    dataset_sha256 = canonical_sha256("confirmatory-dataset")
    dataset_seal_sha256 = canonical_sha256("confirmatory-dataset-seal")
    confirmatory_contract_sha256 = None
    if run_tier == "confirmatory":
        assert preregistration_sha256 is not None
        spec = _confirmatory_spec()
        confirmatory_contract_sha256 = confirmatory_analysis_contract_sha256(
            scientific_identity_sha256=scientific_identity_sha256,
            preregistration_sha256=preregistration_sha256,
            confirmatory_analysis_spec=spec,
            confirmatory_analysis_spec_sha256=canonical_sha256(spec),
            dataset_sha256=dataset_sha256,
            dataset_purpose=DatasetPurpose.EVALUATION,
            dataset_seal_sha256=dataset_seal_sha256,
        )
    policy_ids = (
        "best_static",
        "heuristic_adaptive",
        "neural_matched_history_state_reset",
        "neural_persistent",
        "random_matched",
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
            policy_id: canonical_sha256({"policy_id": policy_id}) for policy_id in policy_ids
        },
        matched_history_policy_sources={"neural_matched_history_state_reset": "neural_persistent"},
        metric_versions={"test-metrics": "1.0.0"},
        seed_schedule=SeedSchedule(model_seeds=(7,), controller_seeds=(11,)),
        action_bounds=ActionBounds(),
        decision_rule_version=decision_rule_version,
        database_schema_version=CURRENT_SCHEMA_VERSION,
        run_tier=run_tier,
        scientific_identity_sha256=scientific_identity_sha256,
        preregistration_sha256=preregistration_sha256,
        confirmatory_analysis_contract_sha256=confirmatory_contract_sha256,
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
    return tuple(
        ScientificGuardrailResult(
            name=name,
            status=ScientificGuardrailStatus.PASS,
            scope="confirmatory_run",
            detail=f"{name} passed",
        )
        for name in REQUIRED_SCIENTIFIC_GUARDRAILS
    )


def _confirmatory_result() -> ConfirmatoryEvaluationResult:
    guardrails = _guardrails()
    comparator_ids: tuple[Literal["best_static", "heuristic_adaptive", "random_matched"], ...] = (
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    )
    comparisons = tuple(
        EfficacyComparisonResult(
            comparator_policy_id=comparator_id,
            comparator_role=(
                ComparatorRole.NEGATIVE_CONTROL
                if comparator_id == "random_matched"
                else ComparatorRole.SERIOUS
            ),
            included_in_holm_family=comparator_id != "random_matched",
            unit_count=0,
            practical_effect_threshold=0.02,
            guardrails=(guardrails[0],),
            status=ScientificEvidenceStatus.INVALID,
            detail="intentionally invalid storage fixture",
        )
        for comparator_id in comparator_ids
    )
    recovery = RecoveryEvaluationResult(
        status=ScientificEvidenceStatus.INVALID,
        detail="intentionally invalid recovery fixture",
    )
    attribution = PersistentStateAttributionResult(
        unit_count=0,
        practical_effect_threshold=0.0,
        causal_guardrails=(guardrails[0],),
        status=ScientificEvidenceStatus.INVALID,
        detail="intentionally invalid attribution fixture",
    )
    decision_input = ScientificDecisionInput(
        tier=ExperimentTier.CONFIRMATORY,
        efficacy_comparisons=comparisons,
        recovery=recovery.decision_gate,
        attribution=attribution.decision_gate,
        guardrails=guardrails,
    )
    decision = decide_scientific_outcome(decision_input)
    assert decision.decision is ScientificDecisionState.INVALID_RUN
    payload = {
        "schema_version": 1,
        "implementation_version": "confirmatory-evaluation-v1",
        "claim_scope": "confirmatory-model-backed-scientific-decision",
        "analysis_contract_sha256": canonical_sha256("provider-free-analysis-contract"),
        "causal_mechanism_validated": True,
        "claim_eligible": False,
        "run_manifest_sha256": None,
        "run_finalization_sha256": None,
        "input_sha256": canonical_sha256("confirmatory-input"),
        "coverage": CoverageResult(exact=True, expected_count=1, observed_count=1),
        "unit_outcomes": (
            ScientificUnitOutcome(
                unit_key=MatchedUnitKey(prompt_sequence_id="sequence-a", model_seed=7),
                policy_id="neural_persistent",
                guardrail_clean_task_score=GuardrailCleanTaskScore(
                    raw_task_score=0.5,
                    gate_status=ScientificGuardrailStatus.PASS,
                    gate_names=("instruction_adherence_non_regression",),
                ),
                instruction_adherence=1.0,
                repetition_ratio=0.0,
                response_length_tokens=8.0,
            ),
        ),
        "efficacy_comparisons": comparisons,
        "recovery": recovery,
        "attribution": attribution,
        "guardrails": guardrails,
        "limitations": (),
        "decision": decision,
        "statistics_call_count": 0,
    }
    return ConfirmatoryEvaluationResult.model_validate(
        {**payload, "result_sha256": confirmatory_result_sha256(payload)}
    )


def _claim_bound_result(
    run_manifest: RunManifest,
    run_finalization: RunFinalization,
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
    base = _confirmatory_result()
    payload = base.model_dump(mode="python", exclude={"result_sha256"})
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


def _complete_llama_request(store: SQLiteRunStore, identity: ProviderIdentity) -> str:
    request = make_request(identity, policy_id="neural_persistent")
    store.prepare_turn(request)
    store.begin_dispatch(request.condition_id)
    response = GenerationResponse(
        text="deterministic stored response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=GenerationMetadata(request_sha256=canonical_sha256(request)),
    )
    store.persist_response(request.condition_id, response)
    store.persist_metrics(request.condition_id, make_metrics(response))
    store.commit_turn(request.condition_id, PolicyState(), make_trace(request))
    return request.condition_id


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
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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
        assert first.guardrail_count == len(REQUIRED_SCIENTIFIC_GUARDRAILS)

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
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
            canonical_sha256("wrong-aggregate"),
            execution_accounting=_complete_accounting(1),
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
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
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
    assert causal_validation_calls == [1, 1]
    assert first.implementation_phase == 5
    assert first.scientific_decision == "INVALID_RUN"
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
        comparison_rows = tuple(csv.DictReader(handle))
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

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["scientific_decision"] == "INVALID_RUN"
    assert decision["reason_codes"] == [
        "attribution_invalid",
        "efficacy_invalid",
        "recovery_invalid",
    ]
    assert decision["execution_accounting"] == {
        "planned_logical_generations": 1,
        "dispatched_logical_generations": 1,
        "successful_responses": 1,
        "uncertain_dispatches": 0,
        "committed_logical_generations": 1,
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
    assert "## Final decision\n\n`INVALID_RUN`. Reason codes:" in report


def test_scientific_export_invokes_causal_gate_and_fails_closed(
    tmp_path: Path,
) -> None:
    """A persisted decision cannot bypass incomplete causal mechanism evidence."""

    run_manifest = _run_manifest()
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database, run_manifest) as store:
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
        )
        result, context = _claim_bound_result(run_manifest, run_finalization)
        store.persist_scientific_analysis(
            _scientific_manifest(run_manifest, run_finalization, result),
            result,
            context=context,
        )

    with pytest.raises(ValueError, match="exactly the two neural policies"):
        export_closed_run(tmp_path)

    assert not (tmp_path / "manifest.json").exists()


def test_scientific_analysis_rejects_foreign_bindings_before_any_write(
    tmp_path: Path,
) -> None:
    run_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        run_finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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


def test_scientific_analysis_requires_finalized_confirmatory_llama_run(
    tmp_path: Path,
) -> None:
    valid_manifest = _run_manifest()
    with SQLiteRunStore(tmp_path / "open.sqlite3", valid_manifest) as store:
        condition_id = _complete_llama_request(store, valid_manifest.provider_identity)
        placeholder_finalization = RunFinalization(
            expected_condition_ids=(condition_id,),
            expected_condition_count=1,
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
        condition_id = _complete_llama_request(store, valid_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
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
        condition_id = _complete_llama_request(store, dirty_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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
        condition_id = _complete_llama_request(
            store,
            wrong_contract_manifest.provider_identity,
        )
        finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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

    pilot_manifest = _run_manifest(
        decision_rule_version="development-pilot-no-scientific-decision-v1",
        run_tier="development_pilot",
        preregistration_sha256=None,
    )
    with SQLiteRunStore(tmp_path / "pilot.sqlite3", pilot_manifest) as store:
        condition_id = _complete_llama_request(store, pilot_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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
        condition_id = _complete_llama_request(store, run_manifest.provider_identity)
        finalization = store.finalize_run(
            (condition_id,),
            scientific_result_sha256(store.list_turns()),
            execution_accounting=_complete_accounting(1),
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

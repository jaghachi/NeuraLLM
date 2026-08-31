"""Pre-execution provenance binding for Phase 3 analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurallm.control.specs import BestStaticPolicySpec
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    PromptFeatures,
    RunManifest,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    MatchedUnitKey,
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    select_best_static,
)
from neurallm.experiments.analysis import (
    analyze_closed_run,
    build_phase3_analysis_contract_sha256,
)
from neurallm.experiments.matching import materialize_matched_coverage
from neurallm.experiments.plan import (
    PHASE3_DECISION_RULE_VERSION,
    ExperimentPlan,
    PlannedTurn,
)
from neurallm.experiments.runner import (
    GitProvenance,
    build_policy_runtimes,
    build_run_manifest,
)
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
)
from tests.storage.helpers import make_manifest, make_request


def _selection_record() -> StaticSelectionRecord:
    first = StaticProfile(
        profile_id="first",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=128,
    )
    second = first.model_copy(update={"profile_id": "second", "temperature": 0.8})
    return select_best_static(
        (
            StaticCandidateResult(profile=first, unit_scores=(0.8,)),
            StaticCandidateResult(profile=second, unit_scores=(0.6,)),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=canonical_sha256("development"),
        development_unit_keys=(MatchedUnitKey(prompt_sequence_id="development-a", model_seed=7),),
    )


def _plan() -> ExperimentPlan:
    provider = FakeProvider()
    request = make_request(provider.provider_identity, policy_id="best_static")
    planned = PlannedTurn(
        condition=request.condition,
        prompt_case_id="case-a",
        prompt_family="synthetic",
        prompt_features=PromptFeatures({}),
        prompt=request.prompt,
        validator=ValidatorSpec(kind="non_empty"),
        decoding_parameters=request.decoding_parameters,
    )
    spec = EvaluationSpec(
        focal_policy_id="best_static",
        required_serious_comparator_ids=("baseline-policy",),
        bootstrap_resamples=4,
        bootstrap_seed=17,
        permutation_resamples=4,
        permutation_seed=19,
    )
    selection = _selection_record()
    matched_units = materialize_matched_coverage(
        (request.condition,),
        experiment_id=request.condition.experiment_id,
        dataset_version=request.condition.dataset_version,
        sequence_turn_indexes={request.condition.prompt_sequence_id: (0,)},
        policy_ids=(request.condition.policy_id,),
        model_seeds=(request.condition.model_seed,),
        controller_seeds=(request.condition.controller_seed,),
    )
    return ExperimentPlan(
        experiment_id=request.condition.experiment_id,
        dataset_version=request.condition.dataset_version,
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        experiment_config_hash=canonical_sha256("experiment-config"),
        dataset_hash=canonical_sha256("dataset"),
        provider_identity=provider.provider_identity,
        provider_effective_configuration_json=fake_provider_effective_configuration_json(),
        action_bounds=ActionBounds(),
        decoding_bounds=DecodingBounds(),
        metric_versions={"test-metrics": "1.0.0"},
        decision_rule_version=PHASE3_DECISION_RULE_VERSION,
        database_schema_version=2,
        evaluation=spec,
        evaluation_spec_sha256=canonical_sha256(spec),
        static_selection_record=selection,
        static_selection_result_sha256=selection.selection_result_sha256,
        matched_units=matched_units,
        turns=(planned,),
    )


def _foreign_selection(plan: ExperimentPlan) -> StaticSelectionRecord:
    selected = plan.static_selection_record
    assert selected is not None
    foreign_results = tuple(
        result.model_copy(
            update={"unit_scores": tuple(1.0 - score for score in result.unit_scores)}
        )
        for result in selected.candidate_results
    )
    return select_best_static(
        foreign_results,
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=selected.development_dataset_sha256,
        development_unit_keys=selected.development_unit_keys,
    )


def test_run_manifest_freezes_the_complete_phase3_analysis_contract() -> None:
    plan = _plan()
    manifest = build_run_manifest(
        plan,
        plan.provider_identity,
        build_policy_runtimes(plan, (BestStaticPolicySpec(),)),
        GitProvenance(source_commit="0" * 40, working_tree_clean=False),
    )

    assert manifest.phase3_analysis_contract_sha256 == (build_phase3_analysis_contract_sha256(plan))
    assert manifest.phase3_analysis_contract_sha256 is not None


def test_phase2_manifest_canonical_identity_omits_the_optional_phase3_contract() -> None:
    legacy = make_manifest(FakeProvider().provider_identity).model_copy(
        update={"database_schema_version": 1}
    )
    serialized = canonical_json(legacy)

    assert "phase3_analysis_contract_sha256" not in serialized
    assert "matched_history_policy_sources" not in serialized
    restored = RunManifest.model_validate_json(serialized)
    assert restored.phase3_analysis_contract_sha256 is None
    assert canonical_sha256(restored) == canonical_sha256(legacy)


def test_run_manifest_requires_the_exact_phase4_rule_and_matched_history_edge() -> None:
    persistent = "neural_persistent"
    reset = "neural_matched_history_state_reset"
    base = make_manifest(FakeProvider().provider_identity).model_dump(mode="python")
    base["policy_config_hashes"] = {
        persistent: canonical_sha256(persistent),
        reset: canonical_sha256(reset),
    }
    base["decision_rule_version"] = "phase4-neural-mechanism-only-v1"
    base["matched_history_policy_sources"] = {reset: persistent}

    phase4 = RunManifest.model_validate(base)

    assert phase4.decision_rule_version == "phase4-neural-mechanism-only-v1"
    assert dict(phase4.matched_history_policy_sources) == {reset: persistent}

    rule_without_edge = dict(base)
    rule_without_edge["matched_history_policy_sources"] = {}
    with pytest.raises(ValueError, match="must appear together"):
        RunManifest.model_validate(rule_without_edge)

    edge_without_rule = dict(base)
    edge_without_rule["decision_rule_version"] = "test-v1"
    with pytest.raises(ValueError, match="must appear together"):
        RunManifest.model_validate(edge_without_rule)

    wrong_edge = dict(base)
    wrong_edge["matched_history_policy_sources"] = {persistent: reset}
    with pytest.raises(ValueError, match="only the Phase 4 matched-history policy edge"):
        RunManifest.model_validate(wrong_edge)


def test_plan_serialization_omits_only_absent_phase3_selection_evidence() -> None:
    phase3 = _plan()
    serialized_phase3 = canonical_json(phase3)
    assert '"static_selection_record"' in serialized_phase3
    assert '"static_selection_result_sha256"' in serialized_phase3
    restored_phase3 = ExperimentPlan.model_validate_json(serialized_phase3)
    assert canonical_sha256(restored_phase3) == canonical_sha256(phase3)

    phase2 = phase3.model_copy(
        update={
            "evaluation": None,
            "evaluation_spec_sha256": None,
            "static_selection_record": None,
            "static_selection_result_sha256": None,
            "matched_units": (),
        }
    )
    serialized_phase2 = canonical_json(phase2)

    assert '"static_selection_record"' not in serialized_phase2
    assert '"static_selection_result_sha256"' not in serialized_phase2
    restored_phase2 = ExperimentPlan.model_validate_json(serialized_phase2)
    assert restored_phase2.static_selection_record is None
    assert restored_phase2.static_selection_result_sha256 is None
    assert canonical_sha256(restored_phase2) == canonical_sha256(phase2)


def test_analysis_rejects_a_foreign_selection_before_opening_the_store(
    tmp_path: Path,
) -> None:
    plan = _plan()
    foreign = _foreign_selection(plan)
    assert foreign.selection_result_sha256 != plan.static_selection_result_sha256

    with pytest.raises(ValueError, match="does not match the frozen plan"):
        analyze_closed_run(plan, foreign, tmp_path / "missing.sqlite3")
    assert not (tmp_path / "missing.sqlite3").exists()

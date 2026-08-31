"""Atomic, hash-verified persistence for finalized Phase 3 evaluation evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    MatchedUnitKey,
    Phase3Verdict,
    SequenceExpectation,
    StaticCandidateResult,
    StaticProfile,
    evaluate_phase3,
    select_best_static,
)
from neurallm.evaluation.contract import phase3_analysis_contract_sha256
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import scientific_result_sha256
from neurallm.storage import (
    AnalysisManifest,
    SQLiteRunStore,
    StoreCorruptionError,
    StoreInvariantError,
)
from tests.storage.helpers import complete_request, make_manifest, make_request


def _evaluation_result(provider: FakeProvider, dataset_sha256: str):
    spec = EvaluationSpec(
        focal_policy_id="focal-policy",
        required_serious_comparator_ids=("test-policy",),
        bootstrap_resamples=10,
        bootstrap_seed=17,
        permutation_resamples=10,
        permutation_seed=19,
    )
    design = ExpectedEvaluationDesign(
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        dataset_sha256=dataset_sha256,
        provider_identity_id=provider.provider_identity.identity_id,
        sequences=(SequenceExpectation(prompt_sequence_id="sequence-a", turn_count=1),),
        model_seeds=(7,),
        controller_seeds=(11,),
        policy_ids=("focal-policy", "test-policy"),
    )
    result = evaluate_phase3((), design=design, spec=spec)
    assert result.verdict is Phase3Verdict.INVALID
    assert not result.statistics_computed
    return result, spec, design


def _selection_record():
    alternative = StaticProfile(
        profile_id="alternative",
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=64,
    )
    winner = alternative.model_copy(update={"profile_id": "winner", "temperature": 0.7})
    return select_best_static(
        (
            StaticCandidateResult(profile=alternative, unit_scores=(0.4,)),
            StaticCandidateResult(profile=winner, unit_scores=(0.8,)),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=canonical_sha256("development-dataset"),
        development_unit_keys=(MatchedUnitKey(prompt_sequence_id="development-a", model_seed=7),),
    )


def _foreign_selection_record():
    selected = _selection_record()
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


def _bind_phase3_contract(run_manifest, spec, design):
    selection = _selection_record()
    contract_sha256 = phase3_analysis_contract_sha256(
        experiment_plan_sha256=canonical_sha256("phase3-plan"),
        evaluation_spec=spec,
        evaluation_spec_sha256=canonical_sha256(spec),
        static_selection_record=selection,
        static_selection_result_sha256=selection.selection_result_sha256,
        evaluation_design=design,
        dataset_sha256=run_manifest.dataset_hash,
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        dataset_seal_sha256=None,
    )
    return run_manifest.model_copy(
        update={
            "decision_rule_version": "phase3-baseline-evaluator-v1",
            "phase3_analysis_contract_sha256": contract_sha256,
        }
    )


def _analysis_manifest(
    *,
    run_manifest,
    run_finalization,
    result,
    spec,
    design,
) -> AnalysisManifest:
    selection = _selection_record()
    return AnalysisManifest(
        run_manifest_sha256=canonical_sha256(run_manifest),
        run_finalization_sha256=canonical_sha256(run_finalization),
        scientific_result_sha256=run_finalization.scientific_result_sha256,
        experiment_plan_sha256=canonical_sha256("phase3-plan"),
        evaluation_spec=spec,
        evaluation_spec_sha256=canonical_sha256(spec),
        static_selection_record=selection,
        static_selection_result_sha256=selection.selection_result_sha256,
        evaluation_design=design,
        dataset_sha256=run_manifest.dataset_hash,
        dataset_purpose=DatasetPurpose.SYNTHETIC,
        evaluation_input_sha256=result.input_sha256,
    )


def test_phase3_analysis_is_atomic_hash_validated_and_idempotent(tmp_path: Path) -> None:
    provider = FakeProvider()
    run_manifest = make_manifest(provider.provider_identity)
    result, spec, design = _evaluation_result(provider, run_manifest.dataset_hash)
    run_manifest = _bind_phase3_contract(run_manifest, spec, design)
    request = make_request(provider.provider_identity)
    database = tmp_path / "run.sqlite3"

    with SQLiteRunStore(database, run_manifest) as store:
        complete_request(store, provider, request)
        result_sha256 = scientific_result_sha256(store.list_turns())
        run_finalization = store.finalize_run((request.condition_id,), result_sha256)
        analysis_manifest = _analysis_manifest(
            run_manifest=run_manifest,
            run_finalization=run_finalization,
            result=result,
            spec=spec,
            design=design,
        )

        first = store.persist_analysis(analysis_manifest, result)
        second = store.persist_analysis(analysis_manifest, result)
        assert second == first
        stored = store.get_analysis()
        assert stored is not None
        assert stored.manifest == analysis_manifest
        assert stored.result == result
        assert stored.comparisons == ()
        assert stored.guardrails == result.global_guardrails
        assert stored.finalization == first
        store.verify_integrity()

    with SQLiteRunStore(database) as reopened:
        assert reopened.get_analysis() == stored


def test_phase3_analysis_requires_closed_phase3_run(tmp_path: Path) -> None:
    provider = FakeProvider()
    run_manifest = make_manifest(provider.provider_identity)
    request = make_request(provider.provider_identity)
    result, spec, design = _evaluation_result(provider, run_manifest.dataset_hash)

    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        complete_request(store, provider, request)
        fake_finalization = {
            "expected_condition_ids": (request.condition_id,),
            "expected_condition_count": 1,
            "manifest_sha256": canonical_sha256(run_manifest),
            "scientific_result_sha256": scientific_result_sha256(store.list_turns()),
        }
        selection = _selection_record()
        analysis_manifest = AnalysisManifest(
            run_manifest_sha256=canonical_sha256(run_manifest),
            run_finalization_sha256=canonical_sha256(fake_finalization),
            scientific_result_sha256=fake_finalization["scientific_result_sha256"],
            experiment_plan_sha256=canonical_sha256("phase3-plan"),
            evaluation_spec=spec,
            evaluation_spec_sha256=canonical_sha256(spec),
            static_selection_record=selection,
            static_selection_result_sha256=selection.selection_result_sha256,
            evaluation_design=design,
            dataset_sha256=run_manifest.dataset_hash,
            dataset_purpose=DatasetPurpose.SYNTHETIC,
            evaluation_input_sha256=result.input_sha256,
        )
        with pytest.raises(StoreInvariantError, match="finalized run"):
            store.persist_analysis(analysis_manifest, result)


def test_phase3_analysis_tampering_fails_closed(tmp_path: Path) -> None:
    provider = FakeProvider()
    run_manifest = make_manifest(provider.provider_identity)
    result, spec, design = _evaluation_result(provider, run_manifest.dataset_hash)
    run_manifest = _bind_phase3_contract(run_manifest, spec, design)
    request = make_request(provider.provider_identity)
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database, run_manifest) as store:
        complete_request(store, provider, request)
        run_finalization = store.finalize_run(
            (request.condition_id,), scientific_result_sha256(store.list_turns())
        )
        store.persist_analysis(
            _analysis_manifest(
                run_manifest=run_manifest,
                run_finalization=run_finalization,
                result=result,
                spec=spec,
                design=design,
            ),
            result,
        )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE analysis_decision SET decision_json = ? WHERE singleton_id = 1",
            ('{"tampered":true}',),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="analysis decision"):
            store.verify_integrity()


def test_foreign_selection_is_rejected_before_first_analysis_persistence(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    run_manifest = make_manifest(provider.provider_identity)
    result, spec, design = _evaluation_result(provider, run_manifest.dataset_hash)
    run_manifest = _bind_phase3_contract(run_manifest, spec, design)
    request = make_request(provider.provider_identity)

    with SQLiteRunStore(tmp_path / "run.sqlite3", run_manifest) as store:
        complete_request(store, provider, request)
        run_finalization = store.finalize_run(
            (request.condition_id,), scientific_result_sha256(store.list_turns())
        )
        expected = _analysis_manifest(
            run_manifest=run_manifest,
            run_finalization=run_finalization,
            result=result,
            spec=spec,
            design=design,
        )
        foreign = _foreign_selection_record()
        proposed = expected.model_copy(
            update={
                "static_selection_record": foreign,
                "static_selection_result_sha256": foreign.selection_result_sha256,
            }
        )

        with pytest.raises(StoreInvariantError, match="pre-execution Phase 3 contract"):
            store.persist_analysis(proposed, result)
        assert store.get_analysis() is None


def test_foreign_selection_tampering_is_rejected_on_read_and_integrity(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    run_manifest = make_manifest(provider.provider_identity)
    result, spec, design = _evaluation_result(provider, run_manifest.dataset_hash)
    run_manifest = _bind_phase3_contract(run_manifest, spec, design)
    request = make_request(provider.provider_identity)
    database = tmp_path / "run.sqlite3"

    with SQLiteRunStore(database, run_manifest) as store:
        complete_request(store, provider, request)
        run_finalization = store.finalize_run(
            (request.condition_id,), scientific_result_sha256(store.list_turns())
        )
        expected = _analysis_manifest(
            run_manifest=run_manifest,
            run_finalization=run_finalization,
            result=result,
            spec=spec,
            design=design,
        )
        store.persist_analysis(expected, result)

    foreign = _foreign_selection_record()
    tampered = expected.model_copy(
        update={
            "static_selection_record": foreign,
            "static_selection_result_sha256": foreign.selection_result_sha256,
        }
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE analysis_manifest SET manifest_json = ?, manifest_sha256 = ?",
            (canonical_json(tampered), canonical_sha256(tampered)),
        )

    with pytest.raises(StoreCorruptionError, match="pre-execution Phase 3 contract"):
        with SQLiteRunStore(database) as reopened:
            reopened.verify_integrity()

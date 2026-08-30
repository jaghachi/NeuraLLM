"""Additional fail-closed coverage for Phase 3 analysis persistence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from neurallm.domain.models import RunManifest
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    MatchedUnitKey,
    Phase3EvaluationResult,
    SequenceExpectation,
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    TurnEvaluationRecord,
    evaluate_phase3,
    select_best_static,
)
from neurallm.evaluation.contract import phase3_analysis_contract_sha256
from neurallm.experiments.analysis import analyze_closed_run
from neurallm.experiments.runner import GitProvenance
from neurallm.experiments.workflow import prepare_experiment
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import scientific_result_sha256
from neurallm.storage import (
    AnalysisFinalization,
    AnalysisManifest,
    RunFinalization,
    SQLiteRunStore,
    StoreCorruptionError,
    StoreInvariantError,
)
from tests.storage.helpers import complete_request, make_manifest, make_request


def _selection_record() -> StaticSelectionRecord:
    first = StaticProfile(
        profile_id="first",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=64,
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


def _evaluation_result(
    provider: FakeProvider, dataset_sha256: str
) -> tuple[
    Phase3EvaluationResult,
    EvaluationSpec,
    ExpectedEvaluationDesign,
]:
    spec = EvaluationSpec(
        focal_policy_id="focal-policy",
        required_serious_comparator_ids=("test-policy",),
        bootstrap_resamples=4,
        bootstrap_seed=17,
        permutation_resamples=4,
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
    records = tuple(
        TurnEvaluationRecord(
            dataset_sha256=dataset_sha256,
            prompt_sequence_id="sequence-a",
            turn_index=0,
            policy_id=policy_id,
            model_seed=7,
            controller_seed=11,
            provider_identity_id=provider.provider_identity.identity_id,
            has_previous_response=False,
            previous_history_commitment_sha256=None,
            task_score=score,
            instruction_adherence=0.9,
            response_length_tokens=64,
            repetition_ratio=0.1,
            action_magnitude=0.2 if policy_id == "focal-policy" else 0.0,
            action_within_bounds=True,
            action_saturated=False,
        )
        for policy_id, score in (("focal-policy", 0.8), ("test-policy", 0.4))
    )
    return evaluate_phase3(records, design=design, spec=spec), spec, design


def _analysis_manifest(
    run_manifest: RunManifest,
    run_finalization: RunFinalization,
    result: Phase3EvaluationResult,
    spec: EvaluationSpec,
    design: ExpectedEvaluationDesign,
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


@contextmanager
def _closed_store(
    database: Path,
    *,
    phase3: bool = True,
) -> Iterator[
    tuple[
        SQLiteRunStore,
        AnalysisManifest,
        Phase3EvaluationResult,
    ]
]:
    provider = FakeProvider()
    base_manifest = make_manifest(provider.provider_identity)
    result, spec, design = _evaluation_result(provider, base_manifest.dataset_hash)
    selection = _selection_record()
    decision_rule = "phase3-baseline-evaluator-v1" if phase3 else "test-v1"
    contract_sha256 = (
        phase3_analysis_contract_sha256(
            experiment_plan_sha256=canonical_sha256("phase3-plan"),
            evaluation_spec=spec,
            evaluation_spec_sha256=canonical_sha256(spec),
            static_selection_record=selection,
            static_selection_result_sha256=selection.selection_result_sha256,
            evaluation_design=design,
            dataset_sha256=base_manifest.dataset_hash,
            dataset_purpose=DatasetPurpose.SYNTHETIC,
            dataset_seal_sha256=None,
        )
        if phase3
        else None
    )
    run_manifest = base_manifest.model_copy(
        update={
            "decision_rule_version": decision_rule,
            "phase3_analysis_contract_sha256": contract_sha256,
        }
    )
    store = SQLiteRunStore(database, run_manifest)
    try:
        request = make_request(provider.provider_identity)
        complete_request(store, provider, request)
        run_finalization = store.finalize_run(
            (request.condition_id,), scientific_result_sha256(store.list_turns())
        )
        yield (
            store,
            _analysis_manifest(run_manifest, run_finalization, result, spec, design),
            result,
        )
    finally:
        store.close()


def test_persist_analysis_rejects_wrong_types_and_duplicate_members(tmp_path: Path) -> None:
    with _closed_store(tmp_path / "run.sqlite3") as (store, manifest, result):
        with pytest.raises(TypeError, match="AnalysisManifest"):
            store.persist_analysis(object(), result)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Phase3EvaluationResult"):
            store.persist_analysis(manifest, object())  # type: ignore[arg-type]

        comparison = result.comparisons[0]
        duplicate_comparisons = result.model_copy(update={"comparisons": (comparison, comparison)})
        with pytest.raises(StoreInvariantError, match="duplicate comparison"):
            store.persist_analysis(manifest, duplicate_comparisons)

        guardrail = result.global_guardrails[0]
        duplicate_guardrails = result.model_copy(
            update={"global_guardrails": (guardrail, guardrail)}
        )
        with pytest.raises(StoreInvariantError, match="nonempty and unique"):
            store.persist_analysis(manifest, duplicate_guardrails)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("phase2", "schema-v2 Phase 3"),
        ("manifest", "run manifest"),
        ("finalization", "run finalization"),
        ("scientific", "scientific result"),
        ("dataset", "run dataset"),
        ("input", "evaluator input"),
    ],
)
def test_analysis_binding_rejects_every_mismatched_identity(
    case: str,
    message: str,
    tmp_path: Path,
) -> None:
    with _closed_store(tmp_path / f"{case}.sqlite3", phase3=case != "phase2") as (
        store,
        manifest,
        result,
    ):
        updates: dict[str, object] = {}
        if case == "manifest":
            updates["run_manifest_sha256"] = "a" * 64
        elif case == "finalization":
            updates["run_finalization_sha256"] = "b" * 64
        elif case == "scientific":
            updates["scientific_result_sha256"] = "c" * 64
        elif case == "dataset":
            updates["dataset_sha256"] = "d" * 64
        elif case == "input":
            updates["evaluation_input_sha256"] = "e" * 64
        mismatched = manifest.model_copy(update=updates)

        with pytest.raises(StoreInvariantError, match=message):
            store.persist_analysis(mismatched, result)


def test_stored_analysis_binding_uses_corruption_error(tmp_path: Path) -> None:
    with _closed_store(tmp_path / "run.sqlite3") as (store, manifest, result):
        mismatched = manifest.model_copy(update={"run_manifest_sha256": "a" * 64})
        with pytest.raises(StoreCorruptionError, match="run manifest"):
            store._validate_analysis_binding(mismatched, result, stored=True)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("manifest", "pre-execution Phase 3 contract"),
        ("decision", "another result"),
        ("finalization", "different closure evidence"),
    ],
)
def test_analysis_rebinding_is_rejected(
    case: str,
    message: str,
    tmp_path: Path,
) -> None:
    with _closed_store(tmp_path / f"{case}.sqlite3") as (store, manifest, result):
        first = store.persist_analysis(manifest, result)
        rebound_manifest = manifest
        if case == "manifest":
            rebound_manifest = manifest.model_copy(update={"experiment_plan_sha256": "a" * 64})
        elif case == "decision":
            other_json = canonical_json({"other": True})
            store._connection.execute(
                "UPDATE analysis_decision SET decision_json = ?, decision_sha256 = ?",
                (other_json, canonical_sha256({"other": True})),
            )
        elif case == "finalization":
            other = first.model_copy(update={"decision_sha256": "f" * 64})
            store._connection.execute(
                """
                UPDATE analysis_finalization
                SET finalization_json = ?, finalization_sha256 = ?
                """,
                (canonical_json(other), canonical_sha256(other)),
            )

        with pytest.raises(StoreInvariantError, match=message):
            store.persist_analysis(rebound_manifest, result)


def test_analysis_member_helper_requires_transaction_and_supported_table(
    tmp_path: Path,
) -> None:
    with SQLiteRunStore(tmp_path / "run.sqlite3") as store:
        with pytest.raises(StoreInvariantError, match="transactionally"):
            store._persist_analysis_member(
                table="guardrail_results",
                id_column="guardrail_id",
                member_id="a" * 64,
                result_json=canonical_json({}),
            )
        with store._transaction():
            with pytest.raises(StoreInvariantError, match="unsupported"):
                store._persist_analysis_member(
                    table="unknown",
                    id_column="unknown_id",
                    member_id="a" * 64,
                    result_json=canonical_json({}),
                )


def test_get_analysis_rejects_orphan_evidence(tmp_path: Path) -> None:
    with SQLiteRunStore(tmp_path / "run.sqlite3") as store:
        store._connection.execute("PRAGMA foreign_keys = OFF")
        payload = {"orphan": True}
        store._connection.execute(
            """
            INSERT INTO analysis_decision(singleton_id, decision_json, decision_sha256)
            VALUES (1, ?, ?)
            """,
            (canonical_json(payload), canonical_sha256(payload)),
        )
        with pytest.raises(StoreCorruptionError, match="without a manifest"):
            store.get_analysis()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-finalization", "not atomically finalized"),
        ("identifier", "identifier does not match"),
        ("membership", "do not match the finalized decision"),
    ],
)
def test_get_analysis_rejects_partial_or_rebound_members(
    case: str,
    message: str,
    tmp_path: Path,
) -> None:
    with _closed_store(tmp_path / f"{case}.sqlite3") as (store, manifest, result):
        store.persist_analysis(manifest, result)
        if case == "missing-finalization":
            store._connection.execute("DELETE FROM analysis_finalization")
        elif case == "identifier":
            store._connection.execute(
                "UPDATE comparison_results SET comparison_id = ?",
                ("f" * 64,),
            )
        elif case == "membership":
            changed = result.comparisons[0].model_copy(
                update={"comparator_policy_id": "other-policy"}
            )
            changed_json = canonical_json(changed)
            changed_hash = canonical_sha256(changed)
            store._connection.execute(
                """
                UPDATE comparison_results
                SET comparison_id = ?, result_json = ?, result_sha256 = ?
                """,
                (changed_hash, changed_json, changed_hash),
            )

        with pytest.raises(StoreCorruptionError, match=message):
            store.get_analysis()


def test_analysis_requires_a_manifest_bound_finalized_run(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    prepared = prepare_experiment(
        repository_root / "configs/experiments/phase3-synthetic-evaluator.yaml",
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=False),
    )
    selection = prepared.loaded_config.config.static_selection_record
    assert selection is not None

    with pytest.raises(StoreInvariantError, match="manifest-bound finalized run"):
        analyze_closed_run(prepared.plan, selection, tmp_path / "empty.sqlite3")


def test_analysis_finalization_model_accepts_exact_evidence_counts() -> None:
    finalization = AnalysisFinalization(
        analysis_manifest_sha256="1" * 64,
        evaluation_result_sha256="2" * 64,
        decision_sha256="3" * 64,
        comparison_result_sha256s=("4" * 64,),
        guardrail_result_sha256s=("5" * 64,),
        comparison_count=1,
        guardrail_count=1,
    )

    assert finalization.comparison_count == 1
    assert finalization.guardrail_count == 1

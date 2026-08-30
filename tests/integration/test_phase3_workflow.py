"""Provider-free Phase 3 preparation and generic-policy execution integration."""

import csv
import json
from pathlib import Path
from typing import NoReturn

import pytest
import yaml

from neurallm.control import (
    BestStaticPolicy,
    HeuristicAdaptivePolicy,
    HeuristicAdaptiveState,
    PolicyState,
    RandomMatchedPolicy,
    RandomMatchedState,
)
from neurallm.control.static import FixedPolicy
from neurallm.domain.models import ActionBounds, DecodingBounds
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    MatchedUnitKey,
    StaticCandidateResult,
    StaticProfile,
    select_best_static,
)
from neurallm.experiments.analysis import evaluation_records_from_store
from neurallm.experiments.dataset import DatasetSeal, PromptDataset
from neurallm.experiments.runner import GitProvenance
from neurallm.experiments.workflow import execute_prepared, prepare_experiment
from neurallm.metrics import METRIC_VERSIONS
from neurallm.providers.fake import (
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.storage import SQLiteRunStore

_PROVENANCE = GitProvenance(source_commit="0" * 40, working_tree_clean=True)


def _dataset(
    purpose: DatasetPurpose,
    *,
    dataset_id: str,
    version: str,
) -> PromptDataset:
    return PromptDataset.model_validate(
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "version": version,
            "purpose": purpose.value,
            "sequences": [
                {
                    "sequence_id": "sequence-a",
                    "cases": [
                        {
                            "case_id": "case-0",
                            "prompt_family": "constrained",
                            "prompt": "Return a non-empty response.",
                            "validator": {"kind": "non_empty"},
                        },
                        {
                            "case_id": "case-1",
                            "prompt_family": "constrained",
                            "prompt": "Return another non-empty response.",
                            "validator": {"kind": "non_empty"},
                        },
                    ],
                }
            ],
        }
    )


def _write_phase3_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    development = _dataset(
        DatasetPurpose.DEVELOPMENT,
        dataset_id="phase3-development",
        version="phase3-development-v1",
    )
    evaluation = _dataset(
        DatasetPurpose.EVALUATION,
        dataset_id="phase3-evaluation",
        version="phase3-evaluation-v1",
    )
    winner = StaticProfile(
        profile_id="selected-static-v1",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=64,
    )
    alternative = StaticProfile(
        profile_id="alternative-static-v1",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.1,
        max_tokens=64,
    )
    selection = select_best_static(
        (
            StaticCandidateResult(profile=winner, unit_scores=(0.9,)),
            StaticCandidateResult(profile=alternative, unit_scores=(0.5,)),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=development.dataset_hash,
        development_unit_keys=(MatchedUnitKey(prompt_sequence_id="sequence-a", model_seed=1),),
    )
    evaluation_seal = DatasetSeal(
        dataset_id=evaluation.dataset_id,
        dataset_version=evaluation.version,
        dataset_sha256=evaluation.dataset_hash,
    )
    config_payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "phase3-workflow",
        "dataset": {
            "path": "evaluation.yaml",
            "version": evaluation.version,
            "purpose": DatasetPurpose.EVALUATION.value,
            "expected_dataset_sha256": evaluation.dataset_hash,
            "seal": evaluation_seal.model_dump(mode="json"),
        },
        "provider": {
            "kind": "fake",
            "expected_identity": fake_provider_identity().model_dump(mode="json"),
            "expected_effective_configuration_json": (fake_provider_effective_configuration_json()),
        },
        "policy_specs": [
            {"kind": "random_matched"},
            {"kind": "heuristic_adaptive"},
            {"kind": "best_static"},
        ],
        "evaluation": EvaluationSpec(
            focal_policy_id="heuristic_adaptive",
            required_serious_comparator_ids=("best_static",),
            negative_control_policy_ids=("random_matched",),
            bootstrap_resamples=32,
            bootstrap_seed=101,
            permutation_resamples=32,
            permutation_seed=202,
        ).model_dump(mode="json"),
        "development_selection_input": {
            "dataset": {
                "path": "development.yaml",
                "version": development.version,
                "purpose": DatasetPurpose.DEVELOPMENT.value,
                "expected_dataset_sha256": development.dataset_hash,
            }
        },
        "static_selection_record": selection.model_dump(mode="json"),
        "model_seeds": [1],
        "controller_seeds": [2],
        "base_decoding_profile_id": winner.profile_id,
        "base_decoding_profile": {
            "temperature": winner.temperature,
            "top_p": winner.top_p,
            "top_k": winner.top_k,
            "presence_penalty": winner.presence_penalty,
            "max_tokens": winner.max_tokens,
        },
        "action_bounds": ActionBounds().model_dump(mode="json"),
        "decoding_bounds": DecodingBounds().model_dump(mode="json"),
        "metric_versions": METRIC_VERSIONS,
        "decision_rule_version": "phase3-baseline-evaluator-v1",
        "database_schema_version": 2,
        "artifact_root": "run",
    }
    development_path = tmp_path / "development.yaml"
    evaluation_path = tmp_path / "evaluation.yaml"
    config_path = tmp_path / "experiment.yaml"
    development_path.write_text(
        yaml.safe_dump(development.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    evaluation_path.write_text(
        yaml.safe_dump(evaluation.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, evaluation_path, development_path


def test_phase3_prepare_is_provider_free_and_builds_typed_runtimes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _, _ = _write_phase3_inputs(tmp_path)

    def forbidden_provider_construction(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("provider-free preparation constructed a provider")

    monkeypatch.setattr(
        "neurallm.experiments.workflow.FakeProvider",
        forbidden_provider_construction,
    )

    prepared = prepare_experiment(config_path, provenance=_PROVENANCE)

    assert prepared.loaded_dataset.dataset.purpose is DatasetPurpose.EVALUATION
    assert prepared.development_selection_dataset is not None
    assert prepared.development_selection_dataset.dataset.purpose is DatasetPurpose.DEVELOPMENT
    assert prepared.plan.dataset_seal == prepared.loaded_config.config.dataset.seal
    assert isinstance(prepared.policy_runtimes["best_static"].policy, BestStaticPolicy)
    assert isinstance(
        prepared.policy_runtimes["heuristic_adaptive"].policy,
        HeuristicAdaptivePolicy,
    )
    assert isinstance(
        prepared.policy_runtimes["random_matched"].policy,
        RandomMatchedPolicy,
    )
    assert prepared.policy_runtimes["best_static"].state_type is PolicyState
    assert prepared.policy_runtimes["heuristic_adaptive"].state_type is HeuristicAdaptiveState
    assert prepared.policy_runtimes["random_matched"].state_type is RandomMatchedState
    assert len(prepared.artifact_identity_sha256) == 64


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("evaluation-purpose", "purpose"),
        ("evaluation-content", "SHA-256"),
        ("evaluation-seal", "canonical seal"),
        ("development-content", "SHA-256"),
    ],
)
def test_phase3_prepare_rejects_dataset_identity_drift(
    drift: str,
    message: str,
    tmp_path: Path,
) -> None:
    config_path, evaluation_path, development_path = _write_phase3_inputs(tmp_path)
    if drift == "evaluation-seal":
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["dataset"]["seal"]["dataset_id"] = "wrong-dataset"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        drift_path = development_path if drift == "development-content" else evaluation_path
        payload = yaml.safe_load(drift_path.read_text(encoding="utf-8"))
        if drift == "evaluation-purpose":
            payload["purpose"] = DatasetPurpose.SYNTHETIC.value
        else:
            payload["sequences"][0]["cases"][0]["prompt"] += " Drifted."
        drift_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_experiment(config_path, provenance=_PROVENANCE)


def test_phase3_fake_execution_uses_generic_runtime_state_and_replays(
    tmp_path: Path,
) -> None:
    config_path, _, _ = _write_phase3_inputs(tmp_path)
    prepared = prepare_experiment(config_path, provenance=_PROVENANCE)

    first = execute_prepared(prepared)
    replay = execute_prepared(prepared)

    assert first.execution.planned_turns == 6
    assert first.execution.committed_turns == 6
    assert first.execution.provider_calls == 6
    assert replay.execution.provider_calls == 0
    assert replay.execution.manifest_sha256 == first.execution.manifest_sha256
    with SQLiteRunStore(prepared.loaded_config.artifact_root / "run.sqlite3") as store:
        records = evaluation_records_from_store(prepared.plan, store)
    assert all(not record.has_previous_response for record in records if record.turn_index == 0)
    assert all(
        record.has_previous_response == (record.policy_id == "heuristic_adaptive")
        for record in records
        if record.turn_index > 0
    )
    decision = json.loads(
        (prepared.loaded_config.artifact_root / "decision.json").read_text(encoding="utf-8")
    )
    assert decision["implementation_phase"] == 3
    assert decision["scientific_decision"] is None
    assert decision["claim_scope"] == "phase-3-statistical-behavior-only"
    assert decision["phase3_baseline_evaluator_verdict"] in {
        "superior",
        "inferior",
        "equivalent",
        "inconclusive",
        "invalid",
    }
    with (prepared.loaded_config.artifact_root / "comparisons.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        comparisons = list(csv.DictReader(handle))
    assert {row["comparator_policy_id"] for row in comparisons} == {
        "best_static",
        "random_matched",
    }
    report = (prepared.loaded_config.artifact_root / "report.md").read_text(encoding="utf-8")
    assert "Baseline evaluator validation" in report
    assert "scientific_decision` remains `null" in report


def test_phase2_smoke_prepare_still_uses_fixed_runtime() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "smoke.yaml"

    prepared = prepare_experiment(config_path, provenance=_PROVENANCE)

    assert prepared.development_selection_dataset is None
    assert tuple(prepared.policy_runtimes) == ("kernel_fixed",)
    assert isinstance(prepared.policy_runtimes["kernel_fixed"].policy, FixedPolicy)
    assert prepared.policy_runtimes["kernel_fixed"].state_type is PolicyState

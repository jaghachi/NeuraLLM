"""Frozen Phase 3 offline dataset, configuration, and plan identities."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurallm.evaluation import DatasetPurpose
from neurallm.experiments.config import load_experiment_config
from neurallm.experiments.dataset import DatasetSeal, load_dataset
from neurallm.experiments.plan import build_plan
from neurallm.experiments.runner import GitProvenance
from neurallm.experiments.workflow import prepare_experiment
from neurallm.experiments.yaml_loader import load_yaml_mapping

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEVELOPMENT_DATASET = _REPOSITORY_ROOT / "datasets/development/phase3-baseline-development-v1.yaml"
_EVALUATION_DATASET = _REPOSITORY_ROOT / "datasets/evaluation/phase3-baseline-evaluation-v1.yaml"
_EVALUATION_SEAL = _REPOSITORY_ROOT / "datasets/evaluation/phase3-baseline-evaluation-v1.seal.yaml"
_SYNTHETIC_DATASET = _REPOSITORY_ROOT / "datasets/synthetic/phase3-evaluator-validation-v1.yaml"

_DEVELOPMENT_SHA256 = "a6c41a046cb84bc9a806866a7393196784eb769118f74cbe4d44d0f3e247df97"
_EVALUATION_SHA256 = "de4c415d71cc3ed0177b189880fa9da040464f41ab14b192bab01cb4eed09199"
_SYNTHETIC_SHA256 = "192d7f5f092eb628cbbc25316aefcbbabd89e4e18dd5180be9e72e5ad426ffbf"
_EVALUATION_SEAL_SHA256 = "89e794e8c80094c15ba9be801306f9ca8090fbd45ab9462b92c172a4a3b65847"
_STATIC_SELECTION_SHA256 = "19be248a50cf6504011168d1e79e3e3cd24d1027017a6cbec443b9019a0bf301"

_REQUIRED_EVALUATION_FAMILIES = {
    "constrained_instruction",
    "long_form_anti_degeneracy",
    "mixed_transition",
    "creative_constrained",
}


@pytest.mark.parametrize(
    ("path", "purpose", "expected_sha256", "sequence_count", "turn_count"),
    [
        (
            _DEVELOPMENT_DATASET,
            DatasetPurpose.DEVELOPMENT,
            _DEVELOPMENT_SHA256,
            6,
            24,
        ),
        (
            _EVALUATION_DATASET,
            DatasetPurpose.EVALUATION,
            _EVALUATION_SHA256,
            8,
            32,
        ),
        (
            _SYNTHETIC_DATASET,
            DatasetPurpose.SYNTHETIC,
            _SYNTHETIC_SHA256,
            4,
            16,
        ),
    ],
)
def test_phase3_datasets_have_frozen_canonical_identity(
    path: Path,
    purpose: DatasetPurpose,
    expected_sha256: str,
    sequence_count: int,
    turn_count: int,
) -> None:
    loaded = load_dataset(path)

    assert loaded.source_path == path.resolve()
    assert loaded.dataset.purpose is purpose
    assert loaded.dataset.dataset_hash == expected_sha256
    assert len(loaded.dataset.sequences) == sequence_count
    assert sum(len(sequence.cases) for sequence in loaded.dataset.sequences) == turn_count
    assert {len(sequence.cases) for sequence in loaded.dataset.sequences} == {4}


def test_development_and_evaluation_cover_the_preregistered_prompt_families() -> None:
    development = load_dataset(_DEVELOPMENT_DATASET).dataset
    evaluation = load_dataset(_EVALUATION_DATASET).dataset

    for dataset in (development, evaluation):
        families = {case.prompt_family for sequence in dataset.sequences for case in sequence.cases}
        assert families == _REQUIRED_EVALUATION_FAMILIES
        long_form_cases = [
            case
            for sequence in dataset.sequences
            for case in sequence.cases
            if case.prompt_family == "long_form_anti_degeneracy"
        ]
        assert long_form_cases
        assert {case.prompt_features["target_length"] for case in long_form_cases} == {160.0}

    development_case_ids = {
        case.case_id for sequence in development.sequences for case in sequence.cases
    }
    evaluation_case_ids = {
        case.case_id for sequence in evaluation.sequences for case in sequence.cases
    }
    assert development_case_ids.isdisjoint(evaluation_case_ids)


def test_evaluation_seal_and_development_selection_evidence_are_frozen() -> None:
    loaded = load_experiment_config(
        _REPOSITORY_ROOT / "configs/experiments/phase3-baseline-evaluation.yaml"
    )
    external_seal = DatasetSeal.model_validate(load_yaml_mapping(_EVALUATION_SEAL))
    selection = loaded.config.static_selection_record

    assert loaded.config.dataset.seal == external_seal
    assert external_seal.dataset_sha256 == _EVALUATION_SHA256
    assert external_seal.seal_sha256 == _EVALUATION_SEAL_SHA256
    assert loaded.development_selection_dataset_path == _DEVELOPMENT_DATASET.resolve()
    assert selection is not None
    assert selection.development_dataset_sha256 == _DEVELOPMENT_SHA256
    assert selection.selection_result_sha256 == _STATIC_SELECTION_SHA256
    assert selection.winning_profile.profile_id == "static-balanced-v1"
    assert tuple(result.profile.profile_id for result in selection.candidate_results) == (
        "static-balanced-v1",
        "static-conservative-v1",
        "static-exploratory-v1",
    )
    assert {len(result.unit_scores) for result in selection.candidate_results} == {12}


@pytest.mark.parametrize(
    (
        "config_name",
        "purpose",
        "dataset_sha256",
        "config_sha256",
        "plan_sha256",
        "turn_count",
        "matched_unit_count",
    ),
    [
        (
            "phase3-baseline-evaluation.yaml",
            DatasetPurpose.EVALUATION,
            _EVALUATION_SHA256,
            "2d670775d19f36cf877cee8dbbf5fd9d41c196dfc3fc10330846bd91516a1587",
            "f236fb5bdc59d1253c1f6a772212c7f462fc6d8094db68fe27515fe4ef6a506c",
            384,
            16,
        ),
        (
            "phase3-synthetic-evaluator.yaml",
            DatasetPurpose.SYNTHETIC,
            _SYNTHETIC_SHA256,
            "e0a6e8e021533340cbb068229cdf3505224a428d4e3d1086df4aaa3b6ec1c2aa",
            "7cd389d71a4d5d936b1797ce23efac8c0cc3ea4b33ac971ee70ddd2e1e68b1bd",
            192,
            8,
        ),
    ],
)
def test_phase3_configs_expand_to_exact_provider_free_plans(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    purpose: DatasetPurpose,
    dataset_sha256: str,
    config_sha256: str,
    plan_sha256: str,
    turn_count: int,
    matched_unit_count: int,
) -> None:
    def _provider_construction_is_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("fixture loading and plan construction must remain provider-free")

    monkeypatch.setattr(
        "neurallm.providers.fake.FakeProvider.__init__",
        _provider_construction_is_forbidden,
    )
    monkeypatch.setattr(
        "neurallm.providers.llama_cpp.LlamaCppProvider.__init__",
        _provider_construction_is_forbidden,
    )

    config_path = _REPOSITORY_ROOT / "configs/experiments" / config_name
    loaded_config = load_experiment_config(config_path)
    loaded_dataset = load_dataset(loaded_config.dataset_path)
    plan = build_plan(loaded_config, loaded_dataset)
    prepared = prepare_experiment(
        config_path,
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=False),
    )

    assert loaded_config.config.provider.kind == "fake"
    assert loaded_config.provider_config_path is None
    assert loaded_config.config.database_schema_version == 2
    assert loaded_config.config.decision_rule_version == "phase3-baseline-evaluator-v1"
    assert loaded_config.config.dataset.purpose is purpose
    assert loaded_config.config.experiment_config_hash == config_sha256
    assert loaded_dataset.dataset.dataset_hash == dataset_sha256
    assert plan.dataset_purpose is purpose
    assert plan.dataset_hash == dataset_sha256
    assert plan.scientific_identity_sha256 == plan_sha256
    assert len(plan.turns) == turn_count
    assert len(plan.matched_units) == matched_unit_count
    assert {turn.decoding_parameters.max_tokens for turn in plan.turns} == {192}
    assert {turn.condition.policy_id for turn in plan.turns} == {
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    }
    assert {unit.expected_condition_count for unit in plan.matched_units} == {24}
    assert prepared.plan == plan
    assert prepared.development_selection_dataset is not None
    assert prepared.development_selection_dataset.dataset.dataset_hash == _DEVELOPMENT_SHA256
    assert set(prepared.policy_runtimes) == {
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    }


def test_synthetic_fixture_declares_all_known_evaluator_outcomes() -> None:
    synthetic = load_dataset(_SYNTHETIC_DATASET).dataset

    assert {sequence.sequence_id for sequence in synthetic.sequences} == {
        "synthetic-known-superior",
        "synthetic-known-inferior",
        "synthetic-identical-equivalent",
        "synthetic-length-confound",
    }
    assert {
        case.prompt_features["synthetic_effect_code"]
        for sequence in synthetic.sequences
        for case in sequence.cases
    } == {-1.0, 0.0, 1.0, 2.0}

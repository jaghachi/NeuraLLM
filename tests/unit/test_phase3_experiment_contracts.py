"""Phase 3 dataset identity, policy configuration, and matched-unit contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from neurallm.domain.models import ActionBounds, DecodingBounds
from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    MatchedUnitKey,
    StaticCandidateResult,
    StaticProfile,
    select_best_static,
)
from neurallm.experiments.config import (
    DevelopmentSelectionInput,
    ExperimentConfig,
    LoadedExperimentConfig,
)
from neurallm.experiments.dataset import (
    DatasetSeal,
    LoadedDataset,
    PromptDataset,
    require_development_selection_input,
)
from neurallm.experiments.matching import materialize_matched_coverage
from neurallm.experiments.plan import build_plan
from neurallm.metrics import METRIC_VERSIONS
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
)

_DEVELOPMENT_SHA256 = "d" * 64


def _dataset_payload(
    *,
    purpose: DatasetPurpose = DatasetPurpose.EVALUATION,
    reverse: bool = False,
) -> dict[str, object]:
    sequences = [
        {
            "sequence_id": "sequence-b",
            "cases": [
                {
                    "case_id": "b-0",
                    "prompt_family": "constrained",
                    "prompt": "Return a non-empty response.",
                    "validator": {"kind": "non_empty"},
                }
            ],
        },
        {
            "sequence_id": "sequence-a",
            "cases": [
                {
                    "case_id": "a-0",
                    "prompt_family": "constrained",
                    "prompt": "Include alpha.",
                    "validator": {
                        "kind": "contains_all",
                        "required_terms": ["alpha"],
                    },
                },
                {
                    "case_id": "a-1",
                    "prompt_family": "structured",
                    "prompt": "Return a JSON object with answer.",
                    "validator": {
                        "kind": "json_object",
                        "required_json_keys": ["answer"],
                    },
                },
            ],
        },
    ]
    if reverse:
        sequences.reverse()
    return {
        "schema_version": 1,
        "dataset_id": "phase3-evaluation",
        "version": "phase3-evaluation-v1",
        "purpose": purpose.value,
        "sequences": sequences,
    }


def _selection_record():
    winner = StaticProfile(
        profile_id="static-winner",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=64,
    )
    alternative = StaticProfile(
        profile_id="static-alternative",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.1,
        max_tokens=64,
    )
    return select_best_static(
        (
            StaticCandidateResult(profile=winner, unit_scores=(0.9, 0.8)),
            StaticCandidateResult(profile=alternative, unit_scores=(0.5, 0.6)),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=_DEVELOPMENT_SHA256,
        development_unit_keys=(
            MatchedUnitKey(prompt_sequence_id="development-a", model_seed=1),
            MatchedUnitKey(prompt_sequence_id="development-b", model_seed=1),
        ),
    )


def _phase3_config_payload(dataset: PromptDataset) -> dict[str, object]:
    selection = _selection_record()
    return {
        "schema_version": 1,
        "experiment_id": "phase3-evaluator",
        "dataset": {
            "path": "evaluation.yaml",
            "version": dataset.version,
            "purpose": DatasetPurpose.EVALUATION.value,
            "expected_dataset_sha256": dataset.dataset_hash,
            "seal": DatasetSeal(
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                dataset_sha256=dataset.dataset_hash,
            ),
        },
        "provider": {
            "kind": "fake",
            "expected_identity": FakeProvider().provider_identity,
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
            bootstrap_seed=101,
            permutation_seed=202,
        ),
        "development_selection_input": {
            "dataset": {
                "path": "development.yaml",
                "version": "phase3-development-v1",
                "purpose": DatasetPurpose.DEVELOPMENT.value,
                "expected_dataset_sha256": _DEVELOPMENT_SHA256,
            }
        },
        "static_selection_record": selection,
        "model_seeds": [2, 1],
        "controller_seeds": [4, 3],
        "base_decoding_profile_id": selection.winning_profile.profile_id,
        "base_decoding_profile": {
            "temperature": selection.winning_profile.temperature,
            "top_p": selection.winning_profile.top_p,
            "top_k": selection.winning_profile.top_k,
            "presence_penalty": selection.winning_profile.presence_penalty,
            "max_tokens": selection.winning_profile.max_tokens,
        },
        "action_bounds": ActionBounds(),
        "decoding_bounds": DecodingBounds(),
        "metric_versions": METRIC_VERSIONS,
        "decision_rule_version": "phase3-baseline-evaluator-v1",
        "database_schema_version": 2,
        "artifact_root": "run",
    }


def _loaded_phase3(*, reverse: bool = False) -> tuple[LoadedExperimentConfig, LoadedDataset]:
    canonical_dataset = PromptDataset.model_validate(_dataset_payload())
    dataset = PromptDataset.model_validate(_dataset_payload(reverse=reverse))
    config = ExperimentConfig.model_validate(_phase3_config_payload(canonical_dataset))
    return (
        LoadedExperimentConfig(
            config=config,
            source_path=Path("config.yaml"),
            dataset_path=Path("evaluation.yaml"),
            provider_config_path=None,
            artifact_root=Path("run"),
            development_selection_dataset_path=Path("development.yaml"),
        ),
        LoadedDataset(dataset=dataset, source_path=Path("evaluation.yaml")),
    )


def test_phase2_legacy_policy_ids_and_untyped_dataset_remain_valid() -> None:
    from tests.unit.test_experiment_plan import loaded_inputs

    loaded_config, loaded_dataset = loaded_inputs()
    plan = build_plan(loaded_config, loaded_dataset)

    assert loaded_config.config.policy_specs is None
    assert loaded_config.config.evaluation is None
    assert loaded_config.config.configured_policy_ids == ("a-policy", "z-policy")
    assert plan.dataset_purpose is None
    assert plan.evaluation is None
    assert plan.matched_units == ()


def test_development_selection_input_rejects_nondevelopment_purpose() -> None:
    dataset = PromptDataset.model_validate(_dataset_payload())
    payload = _phase3_config_payload(dataset)["dataset"]

    with pytest.raises(ValidationError, match="development purpose"):
        DevelopmentSelectionInput.model_validate({"dataset": payload})
    with pytest.raises(ValueError, match="development-purpose"):
        require_development_selection_input(dataset)

    development = PromptDataset.model_validate(_dataset_payload(purpose=DatasetPurpose.DEVELOPMENT))
    assert require_development_selection_input(development) is development


def test_evaluation_reference_requires_valid_matching_seal() -> None:
    dataset = PromptDataset.model_validate(_dataset_payload())
    missing = _phase3_config_payload(dataset)
    missing["dataset"] = dict(missing["dataset"])  # type: ignore[arg-type]
    missing["dataset"].pop("seal")  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="requires a canonical seal"):
        ExperimentConfig.model_validate(missing)

    malformed = _phase3_config_payload(dataset)
    malformed["dataset"] = dict(malformed["dataset"])  # type: ignore[arg-type]
    malformed["dataset"]["seal"] = {  # type: ignore[index]
        "dataset_id": dataset.dataset_id,
        "dataset_version": "wrong-version",
        "dataset_sha256": dataset.dataset_hash,
    }
    with pytest.raises(ValidationError, match="disagrees with its seal"):
        ExperimentConfig.model_validate(malformed)


@pytest.mark.parametrize("drift", ["purpose", "sha256"])
def test_plan_rejects_dataset_purpose_or_hash_mismatch(drift: str) -> None:
    loaded_config, loaded_dataset = _loaded_phase3()
    if drift == "purpose":
        dataset = loaded_dataset.dataset.model_copy(update={"purpose": DatasetPurpose.DEVELOPMENT})
    else:
        payload = _phase3_config_payload(loaded_dataset.dataset)
        payload["dataset"] = dict(payload["dataset"])  # type: ignore[arg-type]
        payload["dataset"]["expected_dataset_sha256"] = "f" * 64  # type: ignore[index]
        payload["dataset"]["seal"] = DatasetSeal(  # type: ignore[index]
            dataset_id=loaded_dataset.dataset.dataset_id,
            dataset_version=loaded_dataset.dataset.version,
            dataset_sha256="f" * 64,
        )
        config = ExperimentConfig.model_validate(payload)
        loaded_config = LoadedExperimentConfig(
            config=config,
            source_path=loaded_config.source_path,
            dataset_path=loaded_config.dataset_path,
            provider_config_path=None,
            artifact_root=loaded_config.artifact_root,
            development_selection_dataset_path=(loaded_config.development_selection_dataset_path),
        )
        dataset = loaded_dataset.dataset

    with pytest.raises(ValueError, match="purpose|SHA-256"):
        build_plan(
            loaded_config,
            LoadedDataset(dataset=dataset, source_path=loaded_dataset.source_path),
        )


def test_policy_specs_are_unique_canonical_and_cannot_mix_with_legacy_ids() -> None:
    dataset = PromptDataset.model_validate(_dataset_payload())
    payload = _phase3_config_payload(dataset)
    config = ExperimentConfig.model_validate(payload)

    assert config.policy_ids is None
    assert config.configured_policy_ids == (
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    )

    duplicate = deepcopy(payload)
    duplicate["policy_specs"] = [{"kind": "best_static"}, {"kind": "best_static"}]
    with pytest.raises(ValidationError, match="duplicate policy identifiers"):
        ExperimentConfig.model_validate(duplicate)

    mixed = deepcopy(payload)
    mixed["policy_ids"] = ["legacy"]
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ExperimentConfig.model_validate(mixed)


def test_phase3_requires_matching_frozen_static_selection_evidence() -> None:
    dataset = PromptDataset.model_validate(_dataset_payload())
    payload = _phase3_config_payload(dataset)

    missing = deepcopy(payload)
    missing.pop("static_selection_record")
    with pytest.raises(ValidationError, match="frozen development selection evidence"):
        ExperimentConfig.model_validate(missing)

    mismatched = deepcopy(payload)
    mismatched["development_selection_input"]["dataset"][  # type: ignore[index]
        "expected_dataset_sha256"
    ] = "e" * 64
    with pytest.raises(ValidationError, match="does not match the declared development input"):
        ExperimentConfig.model_validate(mismatched)

    wrong_winner = deepcopy(payload)
    wrong_winner["base_decoding_profile_id"] = "not-the-selected-profile"
    with pytest.raises(ValidationError, match="shared frozen base profile"):
        ExperimentConfig.model_validate(wrong_winner)


def test_phase3_plan_and_matching_are_deterministic_and_complete() -> None:
    first = build_plan(*_loaded_phase3(reverse=False))
    second = build_plan(*_loaded_phase3(reverse=True))

    assert first.scientific_identity_sha256 == second.scientific_identity_sha256
    assert first.matched_units == second.matched_units
    assert len(first.turns) == 36
    assert len(first.matched_units) == 4
    assert sum(unit.expected_condition_count for unit in first.matched_units) == len(first.turns)
    assert {
        condition_id for unit in first.matched_units for condition_id in unit.expected_condition_ids
    } == {turn.condition.condition_id for turn in first.turns}

    counts_by_sequence = {
        unit.unit_key.prompt_sequence_id: unit.expected_condition_count
        for unit in first.matched_units
    }
    assert counts_by_sequence == {"sequence-a": 12, "sequence-b": 6}


def test_missing_cartesian_cell_fails_and_turns_do_not_inflate_analysis_units() -> None:
    plan = build_plan(*_loaded_phase3())
    conditions = tuple(turn.condition for turn in plan.turns)

    with pytest.raises(ValueError, match="exact Cartesian"):
        materialize_matched_coverage(
            conditions[:-1],
            experiment_id=plan.experiment_id,
            dataset_version=plan.dataset_version,
            sequence_turn_indexes={"sequence-a": (0, 1), "sequence-b": (0,)},
            policy_ids=("best_static", "heuristic_adaptive", "random_matched"),
            model_seeds=(1, 2),
            controller_seeds=(3, 4),
        )

    assert len(plan.matched_units) == 2 * 2
    assert {unit.unit_key.model_seed for unit in plan.matched_units} == {1, 2}
    assert all(unit.controller_seeds == (3, 4) for unit in plan.matched_units)

"""Tests for deterministic experiment-plan expansion."""

from pathlib import Path

import pytest

from neurallm.domain.models import ActionBounds, DecodingBounds
from neurallm.experiments.config import ExperimentConfig, LoadedExperimentConfig
from neurallm.experiments.dataset import LoadedDataset, PromptDataset
from neurallm.experiments.plan import build_plan
from neurallm.metrics import METRIC_VERSIONS
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
)


def config_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "phase2-smoke",
        "dataset": {"path": "dataset.yaml", "version": "smoke-v1"},
        "provider": {
            "kind": "fake",
            "expected_identity": FakeProvider().provider_identity,
            "expected_effective_configuration_json": (fake_provider_effective_configuration_json()),
        },
        "policy_ids": ["kernel_fixed"],
        "model_seeds": [11],
        "controller_seeds": [21],
        "base_decoding_profile_id": "base-v1",
        "base_decoding_profile": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "presence_penalty": 0.0,
            "max_tokens": 64,
        },
        "action_bounds": ActionBounds(),
        "decoding_bounds": DecodingBounds(),
        "metric_versions": METRIC_VERSIONS,
        "decision_rule_version": "phase2-no-scientific-decision-v1",
        "database_schema_version": 1,
        "artifact_root": "run",
    }


def dataset_payload(*, reverse: bool = False) -> dict[str, object]:
    sequences: list[dict[str, object]] = [
        {
            "sequence_id": "sequence-b",
            "cases": [
                {
                    "case_id": "b-0",
                    "prompt_family": "constrained",
                    "prompt": "Return a non-empty answer.",
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
                    "prompt": "Include alpha and beta.",
                    "validator": {
                        "kind": "contains_all",
                        "required_terms": ["alpha", "beta"],
                    },
                },
                {
                    "case_id": "a-1",
                    "prompt_family": "json",
                    "prompt": "Return JSON with answer and reason.",
                    "validator": {
                        "kind": "json_object",
                        "required_json_keys": ["answer", "reason"],
                    },
                },
            ],
        },
    ]
    if reverse:
        sequences.reverse()
    return {
        "schema_version": 1,
        "dataset_id": "phase2-smoke",
        "version": "smoke-v1",
        "sequences": sequences,
    }


def loaded_inputs(*, reverse: bool = False) -> tuple[LoadedExperimentConfig, LoadedDataset]:
    payload = config_payload()
    payload["policy_ids"] = ["z-policy", "a-policy"]
    payload["model_seeds"] = [2, 1]
    payload["controller_seeds"] = [4, 3]
    config = ExperimentConfig.model_validate(payload)
    dataset = PromptDataset.model_validate(dataset_payload(reverse=reverse))
    return (
        LoadedExperimentConfig(
            config=config,
            source_path=Path("config.yaml"),
            dataset_path=Path("dataset.yaml"),
            provider_config_path=None,
            artifact_root=Path("run"),
        ),
        LoadedDataset(dataset=dataset, source_path=Path("dataset.yaml")),
    )


def test_plan_is_complete_unique_and_order_independent() -> None:
    first = build_plan(*loaded_inputs(reverse=False))
    second = build_plan(*loaded_inputs(reverse=True))

    assert len(first.turns) == 24
    assert first.scientific_identity_sha256 == second.scientific_identity_sha256
    assert [turn.condition.condition_id for turn in first.turns] == [
        turn.condition.condition_id for turn in second.turns
    ]
    assert len({turn.logical_request_sha256 for turn in first.turns}) == 24
    assert first.turns[0].condition.prompt_sequence_id == "sequence-a"
    assert first.turns[0].condition.policy_id == "a-policy"
    assert first.turns[0].condition.model_seed == 1
    assert first.turns[0].condition.controller_seed == 3


def test_plan_binds_provider_identity_and_fixed_generation_budget() -> None:
    plan = build_plan(*loaded_inputs())

    assert all(
        turn.condition.provider_identity_id == plan.provider_identity.identity_id
        for turn in plan.turns
    )
    assert {turn.decoding_parameters.max_tokens for turn in plan.turns} == {64}
    assert {turn.decoding_parameters.seed for turn in plan.turns} == {1, 2}


def test_plan_rejects_metric_version_drift() -> None:
    loaded_config, loaded_dataset = loaded_inputs()
    payload = config_payload()
    payload["metric_versions"] = {**METRIC_VERSIONS, "task_score": "drifted"}
    drifted = ExperimentConfig.model_validate(payload)
    loaded_config = LoadedExperimentConfig(
        config=drifted,
        source_path=loaded_config.source_path,
        dataset_path=loaded_config.dataset_path,
        provider_config_path=None,
        artifact_root=loaded_config.artifact_root,
    )

    with pytest.raises(ValueError, match="metric versions"):
        build_plan(loaded_config, loaded_dataset)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("database_schema_version", 999, "database schema version"),
        ("decision_rule_version", "future-rule-v999", "decision rule version"),
    ],
)
def test_plan_rejects_implementation_version_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    loaded_config, loaded_dataset = loaded_inputs()
    payload = config_payload()
    payload[field] = value
    drifted = ExperimentConfig.model_validate(payload)
    loaded_config = LoadedExperimentConfig(
        config=drifted,
        source_path=loaded_config.source_path,
        dataset_path=loaded_config.dataset_path,
        provider_config_path=None,
        artifact_root=loaded_config.artifact_root,
    )

    with pytest.raises(ValueError, match=message):
        build_plan(loaded_config, loaded_dataset)


def test_config_bounds_are_bound_into_plan() -> None:
    loaded_config, loaded_dataset = loaded_inputs()
    assert loaded_config.config.action_bounds == ActionBounds()
    assert loaded_config.config.decoding_bounds == DecodingBounds()

    plan = build_plan(loaded_config, loaded_dataset)

    assert plan.action_bounds == ActionBounds()
    assert plan.decoding_bounds == DecodingBounds()
    assert plan.metric_versions == METRIC_VERSIONS
    assert plan.provider_identity == FakeProvider().provider_identity

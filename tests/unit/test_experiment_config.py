"""Tests for strict experiment configuration and explicit path resolution."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from neurallm.domain.models import ActionBounds, DecodingBounds
from neurallm.experiments.config import ExperimentConfig, load_experiment_config
from neurallm.metrics import METRIC_VERSIONS
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
)


def config_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "phase2-smoke",
        "dataset": {"path": "../../datasets/smoke.yaml", "version": "smoke-v1"},
        "provider": {
            "kind": "fake",
            "expected_identity": FakeProvider().provider_identity.model_dump(mode="json"),
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
        "action_bounds": ActionBounds().model_dump(mode="json"),
        "decoding_bounds": DecodingBounds().model_dump(mode="json"),
        "metric_versions": METRIC_VERSIONS,
        "decision_rule_version": "phase2-no-scientific-decision-v1",
        "database_schema_version": 1,
        "artifact_root": "../../runs/phase2-smoke",
    }


def write_config(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(payload or config_payload(), sort_keys=False),
        encoding="utf-8",
    )


def test_loader_resolves_references_relative_to_explicit_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "experiments" / "smoke.yaml"
    write_config(config_path)

    loaded = load_experiment_config(config_path)

    assert loaded.source_path == config_path.resolve()
    assert loaded.dataset_path == (tmp_path / "datasets" / "smoke.yaml").resolve()
    assert loaded.artifact_root == (tmp_path / "runs" / "phase2-smoke").resolve()
    assert loaded.provider_config_path is None


def test_scientific_config_hash_excludes_incidental_paths() -> None:
    first_payload = config_payload()
    second_payload = config_payload()
    second_payload["dataset"] = {"path": "another/location.yaml", "version": "smoke-v1"}
    second_payload["artifact_root"] = "another/run-root"

    first = ExperimentConfig.model_validate(first_payload)
    second = ExperimentConfig.model_validate(second_payload)

    assert first.experiment_config_hash == second.experiment_config_hash


def test_scientific_config_hash_includes_decoding_bounds() -> None:
    first_payload = config_payload()
    second_payload = config_payload()
    second_payload["decoding_bounds"] = DecodingBounds(temperature=(0.01, 1.5)).model_dump(
        mode="json"
    )

    first = ExperimentConfig.model_validate(first_payload)
    second = ExperimentConfig.model_validate(second_payload)

    assert first.experiment_config_hash != second.experiment_config_hash


def test_policy_and_seed_sets_are_canonicalized() -> None:
    payload = config_payload()
    payload["policy_ids"] = ["z", "a"]
    payload["model_seeds"] = [9, 1]
    payload["controller_seeds"] = [8, 2]

    config = ExperimentConfig.model_validate(payload)

    assert config.policy_ids == ("a", "z")
    assert config.model_seeds == (1, 9)
    assert config.controller_seeds == (2, 8)


def test_duplicate_yaml_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ConstructorError, match="duplicate key"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "mutation",
    [
        {"policy_ids": []},
        {"model_seeds": [1, 1]},
        {"provider": {"kind": "llama_cpp", "expected_identity": {}}},
        {"metric_versions": {}},
    ],
)
def test_invalid_configuration_fails_closed(mutation: dict[str, object]) -> None:
    payload = config_payload()
    payload.update(mutation)

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)

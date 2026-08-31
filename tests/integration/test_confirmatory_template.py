"""Contract test for the provider-free confirmatory preregistration template."""

from __future__ import annotations

import socket
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import pytest
import yaml
from pydantic import ValidationError

from neurallm.domain.models import ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.experiments.config import ExperimentConfig, load_experiment_config
from neurallm.experiments.dataset import load_dataset
from neurallm.experiments.plan import build_plan
from neurallm.experiments.preregistration import publish_preregistration
from neurallm.experiments.protocol import EFFICACY_POLICY_IDS, MODEL_BACKED_POLICY_IDS, RunTier
from neurallm.experiments.yaml_loader import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "model-backed-confirmatory.example.yaml"
EVALUATION_DATASET = ROOT / "datasets" / "evaluation" / "model-backed-confirmatory-v1.yaml"
DEVELOPMENT_DATASET = ROOT / "datasets" / "development" / "phase3-baseline-development-v1.yaml"


def _llama_provider_payload() -> dict[str, object]:
    effective = {"endpoint": "http://127.0.0.1:8080", "request_mode": "completion"}
    return {
        "kind": "llama_cpp",
        "config_path": "../providers/llama_cpp.confirmatory.local.yaml",
        "expected_identity": ProviderIdentity(
            provider_type="llama_cpp",
            implementation_version="llama-cpp-completion-http-v1",
            model_alias="confirmatory-template-test-model",
            build_id="confirmatory-template-test-build",
            provider_config_hash=canonical_sha256(effective),
            model_path="C:/models/confirmatory-template-test.gguf",
            model_sha256="b" * 64,
            chat_template_sha256="c" * 64,
        ).model_dump(mode="json"),
        "expected_effective_configuration_json": canonical_json(effective),
    }


def test_confirmatory_configuration_requires_model_artifact_digest() -> None:
    payload = deepcopy(load_yaml_mapping(TEMPLATE))
    provider = _llama_provider_payload()
    identity = provider["expected_identity"]
    assert isinstance(identity, dict)
    identity["model_sha256"] = None
    payload["provider"] = provider

    with pytest.raises(ValidationError, match="model-artifact SHA-256"):
        ExperimentConfig.model_validate(payload)


def test_confirmatory_template_seals_exact_provider_free_2400_turn_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_payload = load_yaml_mapping(TEMPLATE)
    assert "preregistration" not in source_payload
    assert source_payload["provider"]["kind"] == "llama_cpp"
    assert "PASTE_PREFLIGHT" in source_payload["provider"]["expected_identity"]["model_alias"]
    assert source_payload["provider"]["config_path"].endswith("llama_cpp.confirmatory.local.yaml")
    with pytest.raises(ValidationError, match="provider_config_hash"):
        ExperimentConfig.model_validate(source_payload)

    payload = deepcopy(source_payload)
    unchanged_non_provider = {key: value for key, value in payload.items() if key != "provider"}
    payload["provider"] = _llama_provider_payload()
    assert {
        key: value for key, value in payload.items() if key != "provider"
    } == unchanged_non_provider
    parsed_config = ExperimentConfig.model_validate(payload)

    mirror_root = tmp_path / "mirror"
    config_path = mirror_root / "configs" / "experiments" / "model-backed-confirmatory.yaml"
    evaluation_copy = mirror_root / "datasets" / "evaluation" / EVALUATION_DATASET.name
    development_copy = mirror_root / "datasets" / "development" / DEVELOPMENT_DATASET.name
    config_path.parent.mkdir(parents=True)
    evaluation_copy.parent.mkdir(parents=True)
    development_copy.parent.mkdir(parents=True)
    copyfile(EVALUATION_DATASET, evaluation_copy)
    copyfile(DEVELOPMENT_DATASET, development_copy)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(config_path)
    assert loaded_config.config == parsed_config
    loaded_dataset = load_dataset(
        loaded_config.dataset_path,
        expected_version=loaded_config.config.dataset.version,
    )
    candidate = build_plan(
        loaded_config,
        loaded_dataset,
        require_frozen_preregistration=False,
    )

    protocol = candidate.protocol
    analysis = candidate.confirmatory_analysis
    assert protocol is not None
    assert analysis is not None
    assert protocol.run_tier is RunTier.CONFIRMATORY
    assert protocol.policy_ids == MODEL_BACKED_POLICY_IDS
    assert protocol.efficacy_policy_ids == EFFICACY_POLICY_IDS
    assert protocol.attribution.policy_id not in protocol.efficacy_policy_ids
    assert protocol.schedule.sequence_count == 24
    assert protocol.schedule.turns_per_sequence == 4
    assert protocol.schedule.model_seed_count == 5
    assert protocol.schedule.controller_seed_count == 1
    assert protocol.schedule.policy_count == 5
    assert protocol.schedule.logical_generation_count == 2_400
    assert len(candidate.turns) == 2_400
    assert len(candidate.matched_units) == 120
    assert sum(unit.expected_condition_count for unit in candidate.matched_units) == 2_400

    sequence_ids = {sequence.sequence_id for sequence in loaded_dataset.dataset.sequences}
    assert len(analysis.recovery_events) == 24
    assert {event.prompt_sequence_id for event in analysis.recovery_events} == sequence_ids
    assert all(event.stressor_turn_index == 2 for event in analysis.recovery_events)
    assert all(event.recovery_turn_indexes == (3,) for event in analysis.recovery_events)
    assert all(event.minimum_task_score_target == 0.8 for event in analysis.recovery_events)
    assert all(event.maximum_repetition_ratio_target == 0.2 for event in analysis.recovery_events)
    assert candidate.static_selection_result_sha256 == (
        "19be248a50cf6504011168d1e79e3e3cd24d1027017a6cbec443b9019a0bf301"
    )
    assert loaded_config.config.base_decoding_profile_id == "static-balanced-v1"
    assert loaded_config.config.evaluation is not None
    assert loaded_config.config.evaluation.bootstrap_resamples == 10_000
    assert loaded_config.config.evaluation.confidence_level == 0.95
    assert loaded_config.config.evaluation.practical_effect_threshold == 0.02
    assert loaded_config.config.evaluation.maximum_adherence_regression == 0.01
    assert loaded_config.config.evaluation.maximum_action_saturation_rate == 0.05
    assert loaded_config.config.evaluation.required_matched_coverage == 1.0

    def forbidden_provider_or_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("confirmatory preregistration entered a provider or network path")

    monkeypatch.setattr(
        "neurallm.experiments.workflow.construct_provider",
        forbidden_provider_or_network,
    )
    monkeypatch.setattr(socket, "create_connection", forbidden_provider_or_network)
    publication = publish_preregistration(
        config_path,
        mirror_root / "published" / "confirmatory.seal.json",
    )
    assert publication.scientific_identity_sha256 == candidate.scientific_identity_sha256

    sealed_config = ExperimentConfig.model_validate(
        {
            **loaded_config.config.model_dump(mode="python"),
            "preregistration": publication.seal,
        }
    )
    sealed_plan = build_plan(
        replace(loaded_config, config=sealed_config),
        loaded_dataset,
    )
    assert sealed_plan.preregistration == publication.seal
    assert sealed_plan.scientific_identity_sha256 == candidate.scientific_identity_sha256

    drifted_config = ExperimentConfig.model_validate(
        {
            **sealed_config.model_dump(mode="python"),
            "action_bounds": sealed_config.action_bounds.model_copy(
                update={"temperature_delta": (-0.05, 0.05)}
            ),
        }
    )
    with pytest.raises(ValidationError, match="preregistration seal"):
        build_plan(
            replace(loaded_config, config=drifted_config),
            loaded_dataset,
        )

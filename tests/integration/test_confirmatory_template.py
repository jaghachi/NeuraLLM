"""Contract test for the provider-free confirmatory preregistration template."""

from __future__ import annotations

import json
import socket
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neurallm.cli import main
from neurallm.domain.models import ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_json_bytes
from neurallm.evaluation.pilot_selection import DevelopmentPilotStaticSelectionEvidence
from neurallm.evaluation.selection import StaticProfile
from neurallm.experiments import GitProvenance
from neurallm.experiments.config import ExperimentConfig, load_experiment_config
from neurallm.experiments.dataset import load_dataset
from neurallm.experiments.plan import build_plan
from neurallm.experiments.preregistration import publish_preregistration
from neurallm.experiments.protocol import EFFICACY_POLICY_IDS, MODEL_BACKED_POLICY_IDS, RunTier
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.providers import (
    LlamaCppEffectiveConfiguration,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)
from tests.integration.pilot_selection_helpers import build_test_static_selection_evidence

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "model-backed-confirmatory.example.yaml"
EVALUATION_DATASET = ROOT / "datasets" / "evaluation" / "model-backed-confirmatory-v1.yaml"
DEVELOPMENT_DATASET = ROOT / "datasets" / "development" / "phase3-baseline-development-v1.yaml"


def _llama_provider_config() -> LlamaCppProviderConfig:
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    return LlamaCppProviderConfig(
        base_url="http://127.0.0.1:8080",
        model_alias="confirmatory-template-test-model",
        model_path=str((ROOT / "tests" / "fixtures" / "confirmatory-test.gguf").resolve()),
        model_sha256="b" * 64,
        build_id="confirmatory-template-test-build",
        chat_template_sha256=sha256(chat_template.encode("utf-8")).hexdigest(),
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        write_timeout_seconds=3.0,
        pool_timeout_seconds=4.0,
    )


def _llama_provider_payload(
    provider_config: LlamaCppProviderConfig | None = None,
) -> dict[str, object]:
    resolved_config = provider_config or _llama_provider_config()
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    effective = LlamaCppEffectiveConfiguration(
        client_config=resolved_config,
        model_alias=resolved_config.model_alias,
        model_path=resolved_config.model_path,
        model_sha256=resolved_config.model_sha256,
        build_id=resolved_config.build_id,
        chat_template=chat_template,
        chat_template_sha256=resolved_config.chat_template_sha256,
        default_generation_settings_json=canonical_json(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "presence_penalty": 0.0,
                "n_predict": 192,
                "seed": 6101,
            }
        ),
        total_slots=1,
    )
    return {
        "kind": "llama_cpp",
        "config_path": "../providers/llama_cpp.local.yaml",
        "expected_identity": llama_cpp_provider_identity(effective).model_dump(mode="json"),
        "expected_effective_configuration_json": canonical_json(effective),
    }


def _static_selection_evidence(
    provider: dict[str, object],
) -> DevelopmentPilotStaticSelectionEvidence:
    development = load_dataset(DEVELOPMENT_DATASET).dataset
    identity_payload = provider["expected_identity"]
    effective_configuration_json = provider["expected_effective_configuration_json"]
    assert isinstance(identity_payload, dict)
    assert isinstance(effective_configuration_json, str)
    return build_test_static_selection_evidence(
        development_dataset=development,
        provider_identity=ProviderIdentity.model_validate(identity_payload),
        provider_effective_configuration_json=effective_configuration_json,
        winning_profile=StaticProfile(
            profile_id="static-balanced-v1",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=192,
        ),
    )


def test_confirmatory_configuration_requires_model_artifact_digest() -> None:
    payload = deepcopy(load_yaml_mapping(TEMPLATE))
    provider = _llama_provider_payload()
    selection_evidence = _static_selection_evidence(provider)
    identity = provider["expected_identity"]
    assert isinstance(identity, dict)
    identity["model_sha256"] = None
    payload["provider"] = provider
    payload["static_selection_evidence"] = selection_evidence.model_dump(mode="python")

    with pytest.raises(ValidationError, match="model-artifact SHA-256"):
        ExperimentConfig.model_validate(payload)


def test_confirmatory_template_seals_exact_provider_free_2400_turn_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    tmp_path: Path,
) -> None:
    source_payload = load_yaml_mapping(TEMPLATE)
    assert "preregistration" not in source_payload
    assert source_payload["provider"]["kind"] == "llama_cpp"
    assert "PASTE_PREFLIGHT" in source_payload["provider"]["expected_identity"]["model_alias"]
    assert source_payload["provider"]["config_path"] == "../providers/llama_cpp.local.yaml"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(source_payload)

    provider_config = _llama_provider_config()
    payload = deepcopy(source_payload)
    replaced_fields = {"provider", "static_selection_evidence"}
    unchanged_template_fields = {
        key: value for key, value in payload.items() if key not in replaced_fields
    }
    provider_payload = _llama_provider_payload(provider_config)
    selection_evidence = _static_selection_evidence(provider_payload)
    payload["provider"] = provider_payload
    selection_reference = {
        "path": "../../evidence/development/model-backed-static-selection.json",
        "expected_sha256": selection_evidence.evidence_sha256,
    }
    payload["static_selection_evidence"] = selection_reference
    assert {
        key: value for key, value in payload.items() if key not in replaced_fields
    } == unchanged_template_fields

    mirror_root = tmp_path / "mirror"
    config_path = mirror_root / "configs" / "experiments" / "model-backed-confirmatory.yaml"
    evaluation_copy = mirror_root / "datasets" / "evaluation" / EVALUATION_DATASET.name
    development_copy = mirror_root / "datasets" / "development" / DEVELOPMENT_DATASET.name
    provider_path = mirror_root / "configs" / "providers" / "llama_cpp.local.yaml"
    selection_path = mirror_root / "evidence" / "development" / "model-backed-static-selection.json"
    config_path.parent.mkdir(parents=True)
    evaluation_copy.parent.mkdir(parents=True)
    development_copy.parent.mkdir(parents=True)
    provider_path.parent.mkdir(parents=True)
    selection_path.parent.mkdir(parents=True)
    copyfile(EVALUATION_DATASET, evaluation_copy)
    copyfile(DEVELOPMENT_DATASET, development_copy)
    provider_path.write_text(
        yaml.safe_dump(provider_config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    selection_path.write_bytes(canonical_json_bytes(selection_evidence))
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(config_path)
    assert loaded_config.static_selection_evidence_path == selection_path.resolve()
    assert loaded_config.config.static_selection_evidence == selection_evidence
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
    assert candidate.static_selection_evidence_sha256 == selection_evidence.evidence_sha256
    assert (
        candidate.static_selection_result_sha256
        == selection_evidence.selection_record.selection_result_sha256
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
    monkeypatch.setattr(
        "neurallm.experiments.workflow.LlamaCppProvider.__init__",
        forbidden_provider_or_network,
    )
    monkeypatch.setattr(socket, "create_connection", forbidden_provider_or_network)
    monkeypatch.setattr(
        "neurallm.experiments.workflow.read_git_provenance",
        lambda _path: GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    sealed_config_path = config_path.with_name("model-backed-confirmatory.local.yaml")
    publication = publish_preregistration(
        config_path,
        mirror_root / "published" / "confirmatory.seal.json",
        sealed_config_path,
    )
    assert publication.scientific_identity_sha256 == candidate.scientific_identity_sha256
    assert publication.sealed_config_path == sealed_config_path.resolve()
    assert publication.sealed_config_created is True

    sealed_payload = load_yaml_mapping(sealed_config_path)
    assert sealed_payload["provider"]["config_path"] == "../providers/llama_cpp.local.yaml"
    assert sealed_payload["static_selection_evidence"] == selection_reference
    assert "static_selection_record" not in sealed_payload

    sealed_loaded_config = load_experiment_config(sealed_config_path)
    assert sealed_loaded_config.static_selection_evidence_path == selection_path.resolve()
    assert sealed_loaded_config.config.static_selection_evidence == selection_evidence
    sealed_plan = build_plan(sealed_loaded_config, loaded_dataset)
    assert sealed_plan.preregistration == publication.seal
    assert sealed_plan.scientific_identity_sha256 == candidate.scientific_identity_sha256

    assert main(["run", "--config", str(sealed_config_path), "--dry-run"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "dry-run"
    assert dry_run["provider_constructed"] is False
    assert dry_run["network_requested"] is False
    assert dry_run["planned_turns"] == 2_400

    drifted_config = ExperimentConfig.model_validate(
        {
            **sealed_loaded_config.config.model_dump(mode="python"),
            "model_seeds": tuple(seed + 1 for seed in sealed_loaded_config.config.model_seeds),
        }
    )
    with pytest.raises(ValidationError, match="preregistration seal"):
        build_plan(
            replace(sealed_loaded_config, config=drifted_config),
            loaded_dataset,
        )

"""Integration checks for provider-free preparation and explicit execution."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import pytest
import yaml
from pydantic import ValidationError

from neurallm.domain.models import ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.experiments import GitProvenance
from neurallm.experiments.workflow import execute_prepared, prepare_experiment
from neurallm.providers import (
    LLAMA_CPP_IMPLEMENTATION_VERSION,
    LlamaCppEffectiveConfiguration,
    LlamaCppProviderConfig,
)
from neurallm.reporting import CLOSED_RUN_ARTIFACTS


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "experiments" / "smoke.yaml"


def _write_llama_experiment(tmp_path: Path, *, write_provider: bool) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (root / "datasets" / "development" / "phase2-smoke.yaml").resolve()
    )
    payload["artifact_root"] = str(tmp_path / "run")
    provider_path = tmp_path / "llama.yaml"
    template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    provider_config = LlamaCppProviderConfig(
        base_url="http://127.0.0.1:8080",
        model_alias="test-model",
        model_path="C:/models/test.gguf",
        build_id="test-build",
        chat_template_sha256=sha256(template.encode("utf-8")).hexdigest(),
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        write_timeout_seconds=3.0,
        pool_timeout_seconds=4.0,
    )
    defaults = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "presence_penalty": 0.0,
        "n_predict": 64,
        "seed": 11,
    }
    effective = LlamaCppEffectiveConfiguration(
        client_config=provider_config,
        model_alias=provider_config.model_alias,
        model_path=provider_config.model_path,
        build_id=provider_config.build_id,
        chat_template=template,
        chat_template_sha256=provider_config.chat_template_sha256,
        default_generation_settings_json=canonical_json(defaults),
        total_slots=1,
    )
    identity = ProviderIdentity(
        provider_type="llama_cpp",
        implementation_version=LLAMA_CPP_IMPLEMENTATION_VERSION,
        model_alias=effective.model_alias,
        build_id=effective.build_id,
        provider_config_hash=canonical_sha256(effective),
        model_path=effective.model_path,
        chat_template_sha256=effective.chat_template_sha256,
    )
    payload["provider"] = {
        "kind": "llama_cpp",
        "config_path": str(provider_path),
        "expected_identity": identity.model_dump(mode="json"),
        "expected_effective_configuration_json": canonical_json(effective),
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    if write_provider:
        provider_path.write_text(
            yaml.safe_dump(provider_config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    return config_path, provider_path


def test_prepare_is_provider_free_and_exposes_full_schedule_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_provider_construction(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("dry preparation constructed a provider")

    monkeypatch.setattr(
        "neurallm.experiments.workflow.FakeProvider",
        forbidden_provider_construction,
    )

    prepared = prepare_experiment(
        _config_path(),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )

    assert len(prepared.plan.turns) == 3
    assert len(prepared.plan.scientific_identity_sha256) == 64
    assert len(prepared.artifact_identity_sha256) == 64


def test_provider_free_prepare_rejects_fake_identity_drift(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (root / "datasets" / "development" / "phase2-smoke.yaml").resolve()
    )
    payload["artifact_root"] = str(tmp_path / "run")
    payload["provider"]["expected_identity"]["build_id"] = "drifted"
    config_path = tmp_path / "fake-identity-drift.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="identity is not the built-in contract"):
        prepare_experiment(
            config_path,
            provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("model_seeds", 2**63),
        ("model_seeds", -(2**63) - 1),
        ("controller_seeds", 2**63),
        ("controller_seeds", -(2**63) - 1),
    ),
)
def test_provider_free_prepare_rejects_seeds_outside_sqlite_int64(
    field_name: str,
    value: int,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (root / "datasets" / "development" / "phase2-smoke.yaml").resolve()
    )
    payload["artifact_root"] = str(tmp_path / "run")
    payload[field_name] = [value]
    config_path = tmp_path / f"{field_name}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="less than or equal|greater than or equal"):
        prepare_experiment(
            config_path,
            provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
        )


def test_explicit_execute_closes_and_replays_the_exact_compact_run(tmp_path: Path) -> None:
    prepared = prepare_experiment(
        _config_path(),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    loaded = replace(prepared.loaded_config, artifact_root=tmp_path / "closed-run")
    prepared = replace(prepared, loaded_config=loaded)

    first = execute_prepared(prepared)
    repeated = execute_prepared(prepared)
    independent = execute_prepared(
        replace(
            prepared,
            loaded_config=replace(
                prepared.loaded_config,
                artifact_root=tmp_path / "independent-closed-run",
            ),
        )
    )

    assert first.execution.provider_calls == 3
    assert repeated.execution.provider_calls == 0
    assert repeated.artifacts.manifest_sha256 == first.artifacts.manifest_sha256
    assert (
        independent.artifacts.scientific_result_sha256 == first.artifacts.scientific_result_sha256
    )
    assert independent.execution.provider_calls == 3
    assert {path.name for path in loaded.artifact_root.iterdir()} == CLOSED_RUN_ARTIFACTS


def test_execute_rejects_uncontrolled_run_directory_files(tmp_path: Path) -> None:
    prepared = prepare_experiment(
        _config_path(),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    artifact_root = tmp_path / "run"
    artifact_root.mkdir()
    (artifact_root / "per-turn-response.json").write_text("{}", encoding="utf-8")
    prepared = replace(
        prepared,
        loaded_config=replace(prepared.loaded_config, artifact_root=artifact_root),
    )

    with pytest.raises(ValueError, match="unexpected artifacts"):
        execute_prepared(prepared)


def test_provider_free_prepare_strictly_validates_llama_config_without_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, provider_path = _write_llama_experiment(tmp_path, write_provider=False)
    provenance = GitProvenance(source_commit="0" * 40, working_tree_clean=True)

    with pytest.raises(FileNotFoundError):
        prepare_experiment(config_path, provenance=provenance)

    provider_path.write_text("base_url: 17\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        prepare_experiment(config_path, provenance=provenance)

    config_path, provider_path = _write_llama_experiment(tmp_path, write_provider=True)

    def forbidden_provider_construction(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("provider-free preparation constructed llama.cpp")

    monkeypatch.setattr(
        "neurallm.experiments.workflow.LlamaCppProvider",
        forbidden_provider_construction,
    )
    prepared = prepare_experiment(config_path, provenance=provenance)
    assert prepared.plan.provider_identity.provider_type == "llama_cpp"

    experiment_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_payload["provider"]["expected_identity"]["model_sha256"] = "f" * 64
    config_path.write_text(
        yaml.safe_dump(experiment_payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity disagrees"):
        prepare_experiment(config_path, provenance=provenance)

    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    experiment_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_effective = json.loads(
        experiment_payload["provider"]["expected_effective_configuration_json"]
    )
    raw_effective["client_config"]["base_url"] += "/"
    experiment_payload["provider"]["expected_effective_configuration_json"] = canonical_json(
        raw_effective
    )
    experiment_payload["provider"]["expected_identity"]["provider_config_hash"] = canonical_sha256(
        raw_effective
    )
    config_path.write_text(
        yaml.safe_dump(experiment_payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not normalized preflight evidence"):
        prepare_experiment(config_path, provenance=provenance)

    config_path, provider_path = _write_llama_experiment(tmp_path, write_provider=True)
    provider_payload = yaml.safe_load(provider_path.read_text(encoding="utf-8"))
    provider_payload["read_timeout_seconds"] = 99.0
    provider_path.write_text(
        yaml.safe_dump(provider_payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from expected"):
        prepare_experiment(config_path, provenance=provenance)

    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    experiment_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_payload["model_seeds"] = [-1]
    config_path.write_text(
        yaml.safe_dump(experiment_payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model seeds must be in range"):
        prepare_experiment(config_path, provenance=provenance)

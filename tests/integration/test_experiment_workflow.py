"""Integration checks for provider-free preparation and explicit execution."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import yaml
from pydantic import ValidationError

import neurallm.experiments.workflow as workflow
from neurallm.domain.models import ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.experiments import GitProvenance
from neurallm.experiments.protocol import RunTier
from neurallm.experiments.scientific_analysis import ConfirmatoryAnalysisContext
from neurallm.experiments.workflow import (
    LiveProviderAuthorizationError,
    execute_prepared,
    prepare_experiment,
)
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


def test_provider_free_prepare_rejects_fake_effective_configuration_drift(
    tmp_path: Path,
) -> None:
    prepared = prepare_experiment(
        _config_path(),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    drifted_selection = prepared.loaded_config.config.provider.model_copy(
        update={
            "expected_effective_configuration_json": canonical_json(
                {"generation_method": "different-method"}
            )
        }
    )
    drifted_config = prepared.loaded_config.config.model_copy(
        update={"provider": drifted_selection}
    )
    drifted_loaded = replace(
        prepared.loaded_config,
        config=drifted_config,
        artifact_root=tmp_path / "unused",
    )

    with pytest.raises(ValueError, match="effective configuration is not the built-in contract"):
        workflow._validate_declared_provider_configuration(drifted_loaded)


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


def test_workflow_denies_live_execution_before_directory_or_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    prepared = prepare_experiment(
        config_path,
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )

    def forbidden_provider_construction(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("unauthorized workflow constructed the live provider")

    monkeypatch.setattr(
        "neurallm.experiments.workflow.construct_provider",
        forbidden_provider_construction,
    )

    assert not prepared.loaded_config.artifact_root.exists()
    with pytest.raises(LiveProviderAuthorizationError, match="allow_live_provider=True"):
        execute_prepared(prepared)
    assert not prepared.loaded_config.artifact_root.exists()


def test_workflow_live_authorization_reaches_provider_construction_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    prepared = prepare_experiment(
        config_path,
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )

    class AuthorizedConstructionReached(RuntimeError):
        pass

    def observe_provider_construction(*_args: object, **_kwargs: object) -> NoReturn:
        raise AuthorizedConstructionReached

    monkeypatch.setattr(
        "neurallm.experiments.workflow.construct_provider",
        observe_provider_construction,
    )

    with pytest.raises(AuthorizedConstructionReached):
        execute_prepared(prepared, allow_live_provider=True)
    assert prepared.loaded_config.artifact_root.is_dir()


def test_llama_provider_construction_enforces_preflight_identity_and_closes_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    prepared = prepare_experiment(
        config_path,
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    selection = prepared.loaded_config.config.provider

    for drift, message in (
        (None, None),
        ("identity", "identity does not match"),
        ("effective", "effective configuration does not match"),
    ):
        constructed: list[object] = []

        def make_stub_provider(
            drift_value: str | None,
            constructed_providers: list[object],
        ) -> type:
            class StubProvider:
                def __init__(self, config: LlamaCppProviderConfig) -> None:
                    self.config = config
                    self.closed = False
                    self.provider_identity = (
                        selection.expected_identity.model_copy(update={"build_id": "drifted"})
                        if drift_value == "identity"
                        else selection.expected_identity
                    )
                    self.effective_configuration_json = (
                        canonical_json({"drifted": True})
                        if drift_value == "effective"
                        else selection.expected_effective_configuration_json
                    )
                    constructed_providers.append(self)

                def close(self) -> None:
                    self.closed = True

            return StubProvider

        stub_provider_type = make_stub_provider(drift, constructed)
        monkeypatch.setattr(workflow, "LlamaCppProvider", stub_provider_type)
        if message is None:
            provider = workflow.construct_provider(prepared.loaded_config)
            assert provider is constructed[-1]
            assert constructed[-1].closed is False  # type: ignore[attr-defined]
        else:
            with pytest.raises(ValueError, match=message):
                workflow.construct_provider(prepared.loaded_config)
            assert constructed[-1].closed is True  # type: ignore[attr-defined]


def test_llama_construction_rejects_missing_resolved_config_path(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    prepared = prepare_experiment(
        config_path,
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )

    with pytest.raises(ValueError, match="execution requires an explicit provider config path"):
        workflow.construct_provider(replace(prepared.loaded_config, provider_config_path=None))


def test_live_provider_is_closed_when_execution_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _ = _write_llama_experiment(tmp_path, write_provider=True)
    prepared = prepare_experiment(
        config_path,
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    selection = prepared.loaded_config.config.provider

    class StubLlamaCppProvider:
        provider_identity = selection.expected_identity
        effective_configuration_json = selection.expected_effective_configuration_json

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    provider = StubLlamaCppProvider()
    monkeypatch.setattr(workflow, "LlamaCppProvider", StubLlamaCppProvider)
    monkeypatch.setattr(workflow, "construct_provider", lambda _config: provider)

    def fail_manifest(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("manifest construction failed")

    monkeypatch.setattr(workflow, "build_run_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="manifest construction failed"):
        execute_prepared(prepared, allow_live_provider=True)
    assert provider.closed is True


def test_confirmatory_workflow_persists_and_forwards_exact_claim_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "confirmatory-run"
    spec = object()
    preregistration_sha256 = "1" * 64
    dataset_seal_sha256 = "2" * 64
    scientific_identity_sha256 = "3" * 64
    dataset_sha256 = "4" * 64
    context = ConfirmatoryAnalysisContext(
        analysis_contract_sha256="5" * 64,
        evaluation_input_sha256="6" * 64,
        causal_mechanism_validated=True,
        claim_eligible=True,
        run_manifest_sha256="7" * 64,
        run_finalization_sha256="8" * 64,
    )
    result = object()
    execution = object()
    artifacts = object()
    prepared = SimpleNamespace(
        loaded_config=SimpleNamespace(
            config=SimpleNamespace(provider=SimpleNamespace(kind="llama_cpp")),
            artifact_root=output_directory,
        ),
        loaded_dataset=object(),
        plan=SimpleNamespace(
            protocol=SimpleNamespace(run_tier=RunTier.CONFIRMATORY),
            confirmatory_analysis=spec,
            preregistration=SimpleNamespace(seal_sha256=preregistration_sha256),
            dataset_seal=SimpleNamespace(seal_sha256=dataset_seal_sha256),
            dataset_purpose=DatasetPurpose.EVALUATION,
            scientific_identity_sha256=scientific_identity_sha256,
            dataset_hash=dataset_sha256,
        ),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
        policy_runtimes={},
    )

    class StubLlamaCppProvider:
        provider_identity = ProviderIdentity(
            provider_type="llama_cpp",
            implementation_version="llama-cpp-completion-http-v1",
            model_alias="workflow-test-model",
            build_id="workflow-test-build",
            provider_config_hash="9" * 64,
            model_path="C:/models/workflow-test.gguf",
            chat_template_sha256="a" * 64,
        )

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    provider = StubLlamaCppProvider()
    run_manifest = object()
    run_finalization = SimpleNamespace(scientific_result_sha256="b" * 64)
    analysis_manifest = object()
    persisted: dict[str, object] = {}

    class StubStore:
        def __init__(self, database_path: Path) -> None:
            assert database_path == output_directory / "run.sqlite3"

        def __enter__(self) -> StubStore:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_manifest(self) -> object:
            return run_manifest

        def get_finalization(self) -> object:
            return run_finalization

        def persist_scientific_analysis(
            self,
            submitted_manifest: object,
            submitted_result: object,
            *,
            context: object,
        ) -> None:
            persisted.update(
                manifest=submitted_manifest,
                result=submitted_result,
                context=context,
            )

        def get_scientific_analysis(self) -> object:
            return SimpleNamespace(result=result)

        def verify_integrity(self) -> None:
            persisted["verified"] = True

        def compact(self) -> None:
            persisted["compacted"] = True

    manifest_fields: dict[str, object] = {}

    def capture_analysis_manifest(**fields: object) -> object:
        manifest_fields.update(fields)
        return analysis_manifest

    monkeypatch.setattr(workflow, "LlamaCppProvider", StubLlamaCppProvider)
    monkeypatch.setattr(workflow, "construct_provider", lambda _loaded: provider)
    monkeypatch.setattr(workflow, "build_run_manifest", lambda *_args: run_manifest)
    monkeypatch.setattr(workflow, "execute_plan", lambda *_args: execution)
    monkeypatch.setattr(
        workflow,
        "analyze_closed_confirmatory_run",
        lambda _plan, _database: (result, context),
    )
    monkeypatch.setattr(workflow, "ScientificAnalysisManifest", capture_analysis_manifest)
    monkeypatch.setattr(workflow, "SQLiteRunStore", StubStore)
    monkeypatch.setattr(workflow, "export_closed_run", lambda _output: artifacts)
    monkeypatch.setattr(
        workflow, "canonical_sha256", lambda value: "c" * 64 if value is spec else "d" * 64
    )

    summary = execute_prepared(prepared, allow_live_provider=True)

    assert summary.execution is execution
    assert summary.artifacts is artifacts
    assert provider.closed is True
    assert persisted == {
        "manifest": analysis_manifest,
        "result": result,
        "context": context,
        "verified": True,
        "compacted": True,
    }
    assert manifest_fields == {
        "claim_eligible": True,
        "causal_mechanism_validated": True,
        "run_manifest_sha256": context.run_manifest_sha256,
        "run_finalization_sha256": context.run_finalization_sha256,
        "scientific_result_sha256": run_finalization.scientific_result_sha256,
        "scientific_identity_sha256": scientific_identity_sha256,
        "preregistration_sha256": preregistration_sha256,
        "confirmatory_analysis_contract_sha256": context.analysis_contract_sha256,
        "confirmatory_analysis_spec": spec,
        "confirmatory_analysis_spec_sha256": "c" * 64,
        "dataset_sha256": dataset_sha256,
        "dataset_purpose": DatasetPurpose.EVALUATION,
        "dataset_seal_sha256": dataset_seal_sha256,
        "evaluation_input_sha256": context.evaluation_input_sha256,
    }

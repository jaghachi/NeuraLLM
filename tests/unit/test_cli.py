"""Tests for the explicit, zero-network-by-default Phase 3 CLI."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from neurallm.cli import main
from neurallm.domain.serialization import canonical_json
from neurallm.experiments import GitProvenance


def test_status_is_machine_readable_and_truthful(capsys: Any) -> None:
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "confirmatory_decision_engine_available": True,
        "confirmatory_run_completed": False,
        "implementation_phase": 5,
        "live_provider_validated": False,
        "live_smoke_completed": False,
        "live_smoke_template": "configs/experiments/model-backed-live-smoke.example.yaml",
        "model_backed_protocol_available": True,
        "offline_engineering_smoke_config": (
            "configs/experiments/model-backed-engineering-smoke.yaml"
        ),
        "package": "neurallm",
        "phase_2_kernel_available": True,
        "phase_3_baseline_evaluator_available": True,
        "phase_4_causal_attribution_available": True,
        "readiness": "READY_FOR_LIVE_SMOKE",
        "scientific_decision": None,
        "version": "2.0.0b1",
    }


def test_dry_run_emits_complete_schedule_without_execute_path(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    config = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "smoke.yaml"

    def forbidden_execute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry run entered execute path")

    monkeypatch.setattr("neurallm.cli.execute_prepared", forbidden_execute)

    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "run"
    assert payload["mode"] == "dry-run"
    assert payload["provider_constructed"] is False
    assert payload["network_requested"] is False
    assert payload["planned_turns"] == 3
    assert len(payload["schedule"]) == 3
    assert len(payload["scientific_identity_sha256"]) == 64
    assert len(payload["artifact_identity_sha256"]) == 64
    assert payload["artifact_names"] == [
        "comparisons.csv",
        "decision.json",
        "manifest.json",
        "report.md",
        "results.csv",
        "run.sqlite3",
    ]


def test_validation_failure_is_machine_readable(capsys: Any, tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    assert main(["validate", "--config", str(missing)]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert payload["error"] == "FileNotFoundError"
    assert captured.out == ""


def test_preflight_uses_only_explicit_provider_config_and_emits_canonical_json(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    provider_payload = {
        "base_url": "http://127.0.0.1:8080",
        "model_alias": "explicit-model",
        "model_path": "C:/models/explicit.gguf",
        "model_sha256": "d" * 64,
        "build_id": "explicit-build",
        "chat_template_sha256": "a" * 64,
        "connect_timeout_seconds": 1.0,
        "read_timeout_seconds": 2.0,
        "write_timeout_seconds": 3.0,
        "pool_timeout_seconds": 4.0,
    }
    provider_path = tmp_path / "llama_cpp.local.yaml"
    provider_path.write_text(
        yaml.safe_dump(provider_payload, sort_keys=False),
        encoding="utf-8",
    )
    observed_configs: list[Any] = []
    result_payload = {
        "schema_version": 1,
        "provider_kind": "llama_cpp",
        "expected_identity": {
            "provider_type": "llama_cpp",
            "implementation_version": "llama-cpp-completion-http-v1",
            "model_alias": "explicit-model",
            "build_id": "explicit-build",
            "provider_config_hash": "b" * 64,
            "model_path": "C:/models/explicit.gguf",
            "model_sha256": "d" * 64,
            "chat_template_sha256": "a" * 64,
        },
        "provider_identity_id": "c" * 64,
        "expected_effective_configuration_json": '{"identity":"explicit"}',
        "completion_requested": False,
    }

    def fake_preflight(config: Any) -> object:
        observed_configs.append(config)
        return SimpleNamespace(model_dump=lambda **_kwargs: result_payload)

    monkeypatch.setattr("neurallm.cli.preflight_llama_cpp", fake_preflight)

    assert main(["preflight", "--provider-config", str(provider_path)]) == 0
    output = capsys.readouterr().out
    expected = {
        "command": "preflight",
        "provider_config_path": str(provider_path.resolve()),
        **result_payload,
    }
    assert output == f"{canonical_json(expected)}\n"
    assert len(observed_configs) == 1
    observed = observed_configs[0]
    assert observed.model_alias == "explicit-model"
    assert observed.read_timeout_seconds == 2.0


def test_llama_cpp_execute_requires_second_explicit_authorization_before_construction(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    prepared = SimpleNamespace(
        loaded_config=SimpleNamespace(
            config=SimpleNamespace(provider=SimpleNamespace(kind="llama_cpp"))
        )
    )
    monkeypatch.setattr("neurallm.cli.prepare_experiment", lambda _path: prepared)
    monkeypatch.setattr(
        "neurallm.cli._prepared_payload",
        lambda _prepared: {"provider_kind": "llama_cpp"},
    )

    def forbidden_execute(_prepared: object) -> None:
        raise AssertionError("unauthorized llama.cpp execution constructed a provider")

    monkeypatch.setattr("neurallm.cli.execute_prepared", forbidden_execute)

    assert main(["run", "--config", "machine-local.yaml", "--execute"]) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error == {
        "error": "LiveProviderAuthorizationError",
        "message": (
            "llama_cpp execution requires --allow-live-provider in addition to --execute; "
            "no provider was constructed"
        ),
    }
    assert captured.out == ""


def test_llama_cpp_dry_run_remains_provider_free_without_live_authorization(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    prepared = SimpleNamespace(
        loaded_config=SimpleNamespace(
            config=SimpleNamespace(provider=SimpleNamespace(kind="llama_cpp"))
        )
    )
    monkeypatch.setattr("neurallm.cli.prepare_experiment", lambda _path: prepared)
    monkeypatch.setattr(
        "neurallm.cli._prepared_payload",
        lambda _prepared: {"provider_kind": "llama_cpp", "planned_turns": 1},
    )
    monkeypatch.setattr("neurallm.cli._schedule", lambda _prepared: [])

    def forbidden_execute(_prepared: object) -> None:
        raise AssertionError("llama.cpp dry run constructed a provider")

    monkeypatch.setattr("neurallm.cli.execute_prepared", forbidden_execute)

    assert main(["run", "--config", "machine-local.yaml", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["provider_constructed"] is False
    assert output["network_requested"] is False


def test_llama_cpp_execute_accepts_both_explicit_gates(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    prepared = SimpleNamespace(
        loaded_config=SimpleNamespace(
            config=SimpleNamespace(provider=SimpleNamespace(kind="llama_cpp"))
        )
    )
    monkeypatch.setattr("neurallm.cli.prepare_experiment", lambda _path: prepared)
    monkeypatch.setattr(
        "neurallm.cli._prepared_payload",
        lambda _prepared: {"provider_kind": "llama_cpp", "planned_turns": 1},
    )
    calls: list[tuple[object, bool]] = []
    result = SimpleNamespace(
        execution=SimpleNamespace(
            provider_calls=1,
            previously_committed_turns=0,
            dispatched_this_invocation=1,
            successful_responses_this_invocation=1,
            uncertain_dispatches_this_invocation=0,
            committed_turns=1,
            manifest_sha256="d" * 64,
        ),
        artifacts=SimpleNamespace(
            scientific_result_sha256="e" * 64,
            artifact_names=("run.sqlite3",),
            implementation_phase=5,
            phase3_baseline_evaluator_verdict=None,
            scientific_decision="INVALID_RUN",
        ),
    )

    def execute(prepared_value: object, *, allow_live_provider: bool) -> object:
        calls.append((prepared_value, allow_live_provider))
        return result

    monkeypatch.setattr("neurallm.cli.execute_prepared", execute)

    assert (
        main(
            [
                "run",
                "--config",
                "machine-local.yaml",
                "--execute",
                "--allow-live-provider",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "execute"
    assert output["provider_calls"] == 1
    assert output["scientific_decision"] == "INVALID_RUN"
    assert calls == [(prepared, True)]


def test_validate_plan_execute_analyze_and_report_commands(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    source_config = root / "configs" / "experiments" / "smoke.yaml"
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (root / "datasets" / "development" / "phase2-smoke.yaml").resolve()
    )
    run_directory = tmp_path / "closed-run"
    payload["artifact_root"] = str(run_directory)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "neurallm.experiments.workflow.read_git_provenance",
        lambda _path: GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )

    assert main(["validate", "--config", str(config_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["planned_turns"] == 3

    assert main(["plan", "--config", str(config_path)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert len(planned["schedule"]) == 3

    assert main(["run", "--config", str(config_path), "--execute"]) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["provider_calls"] == 3
    assert executed["committed_turns"] == 3

    assert main(["analyze", "--run-dir", str(run_directory)]) == 0
    analyzed = json.loads(capsys.readouterr().out)
    assert analyzed["scientific_decision"] is None
    assert analyzed["committed_turns"] == 3

    assert main(["report", "--run-dir", str(run_directory)]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["manifest_sha256"] == analyzed["manifest_sha256"]


def test_main_module_import_has_no_cli_side_effect(capsys: Any) -> None:
    module = importlib.import_module("neurallm.__main__")

    assert module.main is main
    assert capsys.readouterr().out == ""

"""Tests for the explicit, zero-network-by-default Phase 3 CLI."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import yaml

from neurallm.cli import main
from neurallm.experiments import GitProvenance


def test_status_is_machine_readable_and_truthful(capsys: Any) -> None:
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "implementation_phase": 3,
        "live_provider_validated": False,
        "package": "neurallm",
        "phase_2_kernel_available": True,
        "phase_3_baseline_evaluator_available": True,
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

"""Focused tests for explicit, store-verified CLI status evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from neurallm.cli import main
from neurallm.domain.serialization import canonical_json
from neurallm.reporting import ArtifactExportSummary
from neurallm.reporting.status import (
    _canonical_json_object,
    _DecisionStatusProjection,
    _resolve_explicit_path,
    _verify_summary,
    load_verified_status,
)

_EXPECTED_COMMITTED_TURNS = {
    "engineering_smoke": 20,
    "development_pilot": 240,
    "confirmatory": 2_400,
}


def _status_run(
    root: Path,
    name: str,
    *,
    run_tier: str,
    hash_character: str,
    scientific_decision: str | None = None,
    provider_type: str = "llama_cpp",
    committed_turns: int | None = None,
) -> tuple[Path, dict[str, object], ArtifactExportSummary]:
    run_directory = (root / name).resolve()
    run_directory.mkdir()
    result_hash_character = format((int(hash_character, 16) + 1) % 16, "x")
    resolved_committed_turns = (
        _EXPECTED_COMMITTED_TURNS[run_tier] if committed_turns is None else committed_turns
    )
    payload: dict[str, object] = {
        "schema_version": 2 if run_tier == "confirmatory" else 1,
        "implementation_phase": 5,
        "run_tier": run_tier,
        "scientific_decision": scientific_decision,
        "manifest_sha256": hash_character * 64,
        "scientific_result_sha256": result_hash_character * 64,
        "provider_type": provider_type,
        "committed_turns": resolved_committed_turns,
        "database_integrity_verified": True,
    }
    (run_directory / "decision.json").write_text(
        f"{canonical_json(payload)}\n",
        encoding="utf-8",
    )
    summary = ArtifactExportSummary(
        output_directory=run_directory,
        manifest_sha256=str(payload["manifest_sha256"]),
        scientific_result_sha256=str(payload["scientific_result_sha256"]),
        committed_turns=resolved_committed_turns,
        artifact_names=("decision.json", "run.sqlite3"),
        implementation_phase=5,
        phase3_baseline_evaluator_verdict=None,
        scientific_decision=scientific_decision,
    )
    return run_directory, payload, summary


def _install_verified_exports(
    monkeypatch: Any,
    fixtures: dict[Path, tuple[dict[str, object], ArtifactExportSummary]],
) -> list[Path]:
    calls: list[Path] = []

    def fake_export(output_directory: Path) -> ArtifactExportSummary:
        resolved = output_directory.resolve(strict=True)
        calls.append(resolved)
        payload, summary = fixtures[resolved]
        (resolved / "decision.json").write_text(
            f"{canonical_json(payload)}\n",
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr("neurallm.reporting.status.export_closed_run", fake_export)
    return calls


def _projection_payload() -> dict[str, object]:
    return {
        "implementation_phase": 5,
        "provider_type": "llama_cpp",
        "committed_turns": 20,
        "manifest_sha256": "a" * 64,
        "scientific_result_sha256": "b" * 64,
        "scientific_decision": None,
        "database_integrity_verified": True,
        "run_tier": "engineering_smoke",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"run_tier": None}, "Phase 5 status evidence must declare a run tier"),
        (
            {"implementation_phase": 4},
            "only Phase 5 status evidence may declare a run tier",
        ),
        (
            {
                "run_tier": "confirmatory",
                "committed_turns": 2_400,
                "provider_type": "fake",
                "scientific_decision": "INCONCLUSIVE",
            },
            "confirmatory status evidence must use llama_cpp",
        ),
        (
            {"run_tier": "confirmatory", "committed_turns": 2_400},
            "confirmatory status evidence must contain a decision",
        ),
        (
            {"scientific_decision": "INCONCLUSIVE"},
            "non-confirmatory status evidence cannot contain a decision",
        ),
    ),
)
def test_status_projection_rejects_incoherent_phase_tier_and_decision_fields(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _DecisionStatusProjection.model_validate({**_projection_payload(), **updates})


def test_status_path_and_canonical_json_guards(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="run directory must be a pathlib.Path"):
        _resolve_explicit_path("not-a-path", expected_kind="run directory")  # type: ignore[arg-type]

    file_path = tmp_path / "decision.json"
    file_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="run directory must be a directory"):
        _resolve_explicit_path(file_path, expected_kind="run directory")

    directory_path = tmp_path / "directory"
    directory_path.mkdir()
    with pytest.raises(ValueError, match="status artifact must be a file"):
        _resolve_explicit_path(directory_path, expected_kind="status artifact")

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object with string keys"):
        _canonical_json_object(list_path)

    noncanonical_path = tmp_path / "noncanonical.json"
    noncanonical_path.write_text('{"b": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="must use canonical JSON"):
        _canonical_json_object(noncanonical_path)


def test_status_summary_cross_checks_directory_and_fields(tmp_path: Path) -> None:
    projection = _DecisionStatusProjection.model_validate(_projection_payload())
    run_directory = (tmp_path / "run").resolve()
    run_directory.mkdir()
    summary = ArtifactExportSummary(
        output_directory=(tmp_path / "another-run").resolve(),
        manifest_sha256="a" * 64,
        scientific_result_sha256="b" * 64,
        committed_turns=20,
        artifact_names=("decision.json", "run.sqlite3"),
        implementation_phase=5,
        phase3_baseline_evaluator_verdict=None,
        scientific_decision=None,
    )
    with pytest.raises(ValueError, match="resolved to a different run directory"):
        _verify_summary(run_directory, projection, summary)

    summary = replace(summary, output_directory=run_directory, manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="disagrees with the verified run export"):
        _verify_summary(run_directory, projection, summary)


def test_status_rejects_ambiguous_sources_and_invalid_grid_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run directories or status artifacts, not both"):
        load_verified_status((tmp_path,), (tmp_path,))
    with pytest.raises(TypeError, match="candidate grid must be a pathlib.Path or None"):
        load_verified_status(candidate_grid_path="grid.json")  # type: ignore[arg-type]

    wrong_name = tmp_path / "status.json"
    wrong_name.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adjacent decision.json"):
        load_verified_status(status_artifacts=(wrong_name,))


def test_status_progresses_only_from_explicit_verified_llama_cpp_tiers(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    smoke, smoke_payload, smoke_summary = _status_run(
        tmp_path,
        "01-smoke",
        run_tier="engineering_smoke",
        hash_character="a",
    )
    pilot, pilot_payload, pilot_summary = _status_run(
        tmp_path,
        "02-pilot",
        run_tier="development_pilot",
        hash_character="b",
    )
    confirmatory, confirmatory_payload, confirmatory_summary = _status_run(
        tmp_path,
        "03-confirmatory",
        run_tier="confirmatory",
        hash_character="c",
        scientific_decision="INCONCLUSIVE",
    )
    calls = _install_verified_exports(
        monkeypatch,
        {
            smoke: (smoke_payload, smoke_summary),
            pilot: (pilot_payload, pilot_summary),
            confirmatory: (confirmatory_payload, confirmatory_summary),
        },
    )

    def forbidden_static_selection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("one pilot must not claim static-selection readiness")

    monkeypatch.setattr(
        "neurallm.reporting.status.build_static_selection_evidence",
        forbidden_static_selection,
    )

    assert main(["status", "--run-dir", str(smoke)]) == 0
    smoke_status = json.loads(capsys.readouterr().out)
    assert smoke_status["readiness"] == "READY_FOR_DEVELOPMENT_PILOT"
    assert smoke_status["live_provider_validated"] is True
    assert smoke_status["live_smoke_completed"] is True
    assert smoke_status["development_pilot_completed"] is False

    assert main(["status", "--run-dir", str(pilot), "--run-dir", str(pilot)]) == 0
    pilot_status = json.loads(capsys.readouterr().out)
    assert pilot_status["readiness"] == "READY_FOR_ADDITIONAL_DEVELOPMENT_PILOT"
    assert pilot_status["live_smoke_completed"] is False
    assert pilot_status["development_pilot_completed"] is True
    assert pilot_status["static_selection_ready"] is False

    assert (
        main(
            [
                "status",
                "--run-dir",
                str(smoke),
                "--run-dir",
                str(smoke),
                "--run-dir",
                str(pilot),
                "--run-dir",
                str(confirmatory),
            ]
        )
        == 0
    )
    completed = json.loads(capsys.readouterr().out)
    assert completed["readiness"] == "CONFIRMATORY_RUN_COMPLETED"
    assert completed["scientific_decision"] == "INCONCLUSIVE"
    assert completed["live_smoke_completed"] is True
    assert completed["development_pilot_completed"] is True
    assert completed["static_selection_ready"] is False
    assert completed["confirmatory_run_completed"] is True
    assert [item["committed_turns"] for item in completed["status_evidence"]] == [
        20,
        240,
        2_400,
    ]
    assert [item["run_directory"] for item in completed["status_evidence"]] == [
        str(smoke),
        str(pilot),
        str(confirmatory),
    ]
    assert calls.count(smoke) == 2
    assert calls.count(pilot) == 2
    assert calls.count(confirmatory) == 1


def test_two_pilots_remain_intermediate_with_or_without_an_explicit_grid(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    first, first_payload, first_summary = _status_run(
        tmp_path,
        "pilot-a",
        run_tier="development_pilot",
        hash_character="a",
    )
    second, second_payload, second_summary = _status_run(
        tmp_path,
        "pilot-b",
        run_tier="development_pilot",
        hash_character="b",
    )
    export_calls = _install_verified_exports(
        monkeypatch,
        {
            first: (first_payload, first_summary),
            second: (second_payload, second_summary),
        },
    )

    arguments = [
        "status",
        "--run-dir",
        str(first),
        "--run-dir",
        str(second),
    ]
    assert main(arguments) == 0
    without_grid = json.loads(capsys.readouterr().out)
    assert without_grid["readiness"] == "READY_FOR_ADDITIONAL_DEVELOPMENT_PILOT"
    assert without_grid["static_selection_ready"] is False

    candidate_grid_path = (tmp_path / "candidate-grid.json").resolve()
    candidate_grid_path.write_text("test grid\n", encoding="utf-8")
    candidate_grid = SimpleNamespace(candidate_profiles=(object(), object(), object()))
    grid_load_calls: list[Path] = []

    def fake_load_candidate_grid(path: Path) -> object:
        grid_load_calls.append(path)
        return candidate_grid

    def forbidden_static_selection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("two pilots do not cover the declared three-profile grid")

    monkeypatch.setattr(
        "neurallm.reporting.status.load_development_pilot_candidate_grid",
        fake_load_candidate_grid,
    )
    monkeypatch.setattr(
        "neurallm.reporting.status.build_static_selection_evidence",
        forbidden_static_selection,
    )

    assert main([*arguments, "--candidate-grid", str(candidate_grid_path)]) == 0
    with_grid = json.loads(capsys.readouterr().out)
    assert with_grid["readiness"] == "READY_FOR_ADDITIONAL_DEVELOPMENT_PILOT"
    assert with_grid["static_selection_ready"] is False
    assert grid_load_calls == [candidate_grid_path]
    assert export_calls.count(first) == 2
    assert export_calls.count(second) == 2


def test_status_rederives_static_selection_from_deduped_artifact_parents(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    first, first_payload, first_summary = _status_run(
        tmp_path,
        "pilot-a",
        run_tier="development_pilot",
        hash_character="a",
    )
    second, second_payload, second_summary = _status_run(
        tmp_path,
        "pilot-b",
        run_tier="development_pilot",
        hash_character="b",
    )
    third, third_payload, third_summary = _status_run(
        tmp_path,
        "pilot-c",
        run_tier="development_pilot",
        hash_character="c",
    )
    export_calls = _install_verified_exports(
        monkeypatch,
        {
            first: (first_payload, first_summary),
            second: (second_payload, second_summary),
            third: (third_payload, third_summary),
        },
    )
    without_grid_arguments = [
        "status",
        "--status-artifact",
        str(first / "decision.json"),
        "--status-artifact",
        str(second / "decision.json"),
        "--status-artifact",
        str(third / "decision.json"),
    ]
    assert main(without_grid_arguments) == 2
    missing_grid = capsys.readouterr()
    assert json.loads(missing_grid.err) == {
        "error": "ValueError",
        "message": "3 or more development-pilot runs require an explicit candidate grid",
    }
    assert missing_grid.out == ""

    candidate_grid_path = (tmp_path / "candidate-grid.json").resolve()
    candidate_grid_path.write_text("test grid\n", encoding="utf-8")
    candidate_grid = SimpleNamespace(candidate_profiles=(object(), object(), object()))
    grid_load_calls: list[Path] = []
    selection_calls: list[tuple[tuple[Path, ...], object]] = []

    def fake_load_candidate_grid(path: Path) -> object:
        grid_load_calls.append(path)
        return candidate_grid

    def fake_build_static_selection(
        run_directories: tuple[Path, ...],
        grid: object,
    ) -> object:
        selection_calls.append((run_directories, grid))
        return object()

    monkeypatch.setattr(
        "neurallm.reporting.status.load_development_pilot_candidate_grid",
        fake_load_candidate_grid,
    )
    monkeypatch.setattr(
        "neurallm.reporting.status.build_static_selection_evidence",
        fake_build_static_selection,
    )

    assert (
        main(
            [
                "status",
                "--status-artifact",
                str(first / "decision.json"),
                "--status-artifact",
                str(first / "decision.json"),
                "--status-artifact",
                str(second / "decision.json"),
                "--status-artifact",
                str(third / "decision.json"),
                "--candidate-grid",
                str(candidate_grid_path),
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["readiness"] == "READY_FOR_STATIC_SELECTION"
    assert status["development_pilot_completed"] is True
    assert status["static_selection_ready"] is True
    assert [item["run_directory"] for item in status["status_evidence"]] == [
        str(first),
        str(second),
        str(third),
    ]
    assert grid_load_calls == [candidate_grid_path]
    assert selection_calls == [((first, second, third), candidate_grid)]
    assert export_calls.count(first) == 2
    assert export_calls.count(second) == 2
    assert export_calls.count(third) == 2


def test_status_rejects_cross_run_static_selection_incompatibility(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    first, first_payload, first_summary = _status_run(
        tmp_path,
        "pilot-a",
        run_tier="development_pilot",
        hash_character="a",
    )
    second, second_payload, second_summary = _status_run(
        tmp_path,
        "pilot-b",
        run_tier="development_pilot",
        hash_character="b",
    )
    third, third_payload, third_summary = _status_run(
        tmp_path,
        "pilot-c",
        run_tier="development_pilot",
        hash_character="c",
    )
    fourth, fourth_payload, fourth_summary = _status_run(
        tmp_path,
        "pilot-d",
        run_tier="development_pilot",
        hash_character="d",
    )
    selection_calls: list[tuple[Path, ...]] = []
    _install_verified_exports(
        monkeypatch,
        {
            first: (first_payload, first_summary),
            second: (second_payload, second_summary),
            third: (third_payload, third_summary),
            fourth: (fourth_payload, fourth_summary),
        },
    )
    candidate_grid_path = (tmp_path / "candidate-grid.json").resolve()
    candidate_grid_path.write_text("test grid\n", encoding="utf-8")
    candidate_grid = SimpleNamespace(candidate_profiles=(object(), object(), object()))
    monkeypatch.setattr(
        "neurallm.reporting.status.load_development_pilot_candidate_grid",
        lambda _path: candidate_grid,
    )

    def reject_incompatible(
        run_directories: tuple[Path, ...],
        _grid: object,
    ) -> None:
        selection_calls.append(run_directories)
        raise ValueError("pilot candidates differ from the declared candidate grid")

    monkeypatch.setattr(
        "neurallm.reporting.status.build_static_selection_evidence",
        reject_incompatible,
    )

    assert (
        main(
            [
                "status",
                "--run-dir",
                str(first),
                "--run-dir",
                str(second),
                "--run-dir",
                str(third),
                "--run-dir",
                str(fourth),
                "--candidate-grid",
                str(candidate_grid_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "error": "ValueError",
        "message": "pilot candidates differ from the declared candidate grid",
    }
    assert selection_calls == [(first, second, third, fourth)]
    assert captured.out == ""


@pytest.mark.parametrize(
    ("run_tier", "committed_turns", "expected_turns", "hash_character"),
    (
        ("engineering_smoke", 19, 20, "a"),
        ("development_pilot", 239, 240, "b"),
        ("confirmatory", 2_399, 2_400, "c"),
    ),
)
def test_status_rejects_phase5_tier_committed_turn_mismatches(
    run_tier: str,
    committed_turns: int,
    expected_turns: int,
    hash_character: str,
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    scientific_decision = "INCONCLUSIVE" if run_tier == "confirmatory" else None
    run_directory, payload, summary = _status_run(
        tmp_path,
        run_tier,
        run_tier=run_tier,
        hash_character=hash_character,
        scientific_decision=scientific_decision,
        committed_turns=committed_turns,
    )
    _install_verified_exports(monkeypatch, {run_directory: (payload, summary)})

    assert main(["status", "--run-dir", str(run_directory)]) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error["error"] == "ValidationError"
    assert (
        f"{run_tier} status evidence must contain exactly {expected_turns} committed turns"
        in error["message"]
    )
    assert captured.out == ""


def test_status_artifact_must_match_regenerated_canonical_store(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    run_directory, authoritative, summary = _status_run(
        tmp_path,
        "smoke",
        run_tier="engineering_smoke",
        hash_character="d",
    )
    _install_verified_exports(monkeypatch, {run_directory: (authoritative, summary)})
    status_artifact = run_directory / "decision.json"

    assert main(["status", "--status-artifact", str(status_artifact)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status_evidence"][0]["source_kind"] == "status_artifact"

    stale = {**authoritative, "manifest_sha256": "e" * 64}
    status_artifact.write_text(f"{canonical_json(stale)}\n", encoding="utf-8")
    assert main(["status", "--status-artifact", str(status_artifact)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "error": "ValueError",
        "message": "status artifact does not match its verified canonical run store",
    }
    assert captured.out == ""


def test_status_rejects_multiple_distinct_confirmatory_runs(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    first, first_payload, first_summary = _status_run(
        tmp_path,
        "confirmatory-a",
        run_tier="confirmatory",
        hash_character="a",
        scientific_decision="INCONCLUSIVE",
    )
    second, second_payload, second_summary = _status_run(
        tmp_path,
        "confirmatory-b",
        run_tier="confirmatory",
        hash_character="b",
        scientific_decision="INCONCLUSIVE",
    )
    _install_verified_exports(
        monkeypatch,
        {
            first: (first_payload, first_summary),
            second: (second_payload, second_summary),
        },
    )

    assert (
        main(
            [
                "status",
                "--run-dir",
                str(first),
                "--run-dir",
                str(second),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "error": "ValueError",
        "message": "status evidence contains multiple distinct confirmatory runs",
    }
    assert captured.out == ""


def test_cli_does_not_swallow_unexpected_programming_errors(
    monkeypatch: Any,
) -> None:
    def fail_unexpectedly(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unexpected programming fault")

    monkeypatch.setattr(
        "neurallm.reporting.status_cli.load_verified_status",
        fail_unexpectedly,
    )

    with pytest.raises(AssertionError, match="unexpected programming fault"):
        main(["status"])

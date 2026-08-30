"""Explicit, machine-readable NeuraLLM Phase 2 command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from neurallm import __version__
from neurallm.domain.serialization import canonical_json
from neurallm.experiments.workflow import (
    PreparedExperiment,
    execute_prepared,
    prepare_experiment,
)
from neurallm.reporting import CLOSED_RUN_ARTIFACTS, export_closed_run


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="explicit experiment YAML path",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurallm",
        description="Deterministic research software for neural decoding control.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="print implementation and scientific status")

    validate = subparsers.add_parser("validate", help="validate config, dataset, and contracts")
    _add_config_argument(validate)
    plan = subparsers.add_parser("plan", help="print the complete deterministic schedule")
    _add_config_argument(plan)
    run = subparsers.add_parser("run", help="dry-run or explicitly execute an experiment")
    _add_config_argument(run)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_const",
        const="dry-run",
        dest="run_mode",
        help="construct schedule and identities without constructing a provider",
    )
    mode.add_argument(
        "--execute",
        action="store_const",
        const="execute",
        dest="run_mode",
        help="construct the selected provider and execute exactly once per pending turn",
    )

    analyze = subparsers.add_parser("analyze", help="verify and derive closed-run artifacts")
    analyze.add_argument("--run-dir", type=Path, required=True)
    report = subparsers.add_parser("report", help="reproduce the closed-run report")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


def _schedule(prepared: PreparedExperiment) -> list[dict[str, object]]:
    return [
        {
            "condition_id": turn.condition.condition_id,
            "logical_request_sha256": turn.logical_request_sha256,
            "prompt_sequence_id": turn.condition.prompt_sequence_id,
            "prompt_case_id": turn.prompt_case_id,
            "turn_index": turn.condition.turn_index,
            "policy_id": turn.condition.policy_id,
            "model_seed": turn.condition.model_seed,
            "controller_seed": turn.condition.controller_seed,
        }
        for turn in prepared.plan.turns
    ]


def _prepared_payload(prepared: PreparedExperiment) -> dict[str, object]:
    return {
        "experiment_id": prepared.plan.experiment_id,
        "config_path": str(prepared.loaded_config.source_path),
        "dataset_path": str(prepared.loaded_dataset.source_path),
        "artifact_root": str(prepared.loaded_config.artifact_root),
        "provider_kind": prepared.loaded_config.config.provider.kind,
        "provider_identity_id": prepared.plan.provider_identity.identity_id,
        "source_commit": prepared.provenance.source_commit,
        "working_tree_clean": prepared.provenance.working_tree_clean,
        "experiment_config_hash": prepared.plan.experiment_config_hash,
        "dataset_hash": prepared.plan.dataset_hash,
        "scientific_identity_sha256": prepared.plan.scientific_identity_sha256,
        "artifact_identity_sha256": prepared.artifact_identity_sha256,
        "planned_turns": len(prepared.plan.turns),
    }


def _print(payload: object) -> None:
    print(canonical_json(payload))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one explicit command; dry modes never construct a provider."""

    args = _build_parser().parse_args(argv)
    try:
        if args.command == "status":
            _print(
                {
                    "package": "neurallm",
                    "version": __version__,
                    "implementation_phase": 2,
                    "phase_2_kernel_available": True,
                    "scientific_decision": None,
                    "live_provider_validated": False,
                }
            )
            return 0

        if args.command in {"validate", "plan", "run"}:
            prepared = prepare_experiment(args.config)
            payload = _prepared_payload(prepared)
            if args.command == "validate":
                payload["command"] = "validate"
                payload["valid"] = True
                _print(payload)
                return 0
            if args.command == "plan":
                payload["command"] = "plan"
                payload["schedule"] = _schedule(prepared)
                _print(payload)
                return 0
            if args.run_mode == "dry-run":
                payload["command"] = "run"
                payload["mode"] = "dry-run"
                payload["provider_constructed"] = False
                payload["network_requested"] = False
                payload["artifact_names"] = sorted(CLOSED_RUN_ARTIFACTS)
                payload["schedule"] = _schedule(prepared)
                _print(payload)
                return 0
            result = execute_prepared(prepared)
            payload.update(
                {
                    "command": "run",
                    "mode": "execute",
                    "provider_calls": result.execution.provider_calls,
                    "committed_turns": result.execution.committed_turns,
                    "manifest_sha256": result.execution.manifest_sha256,
                    "scientific_result_sha256": (result.artifacts.scientific_result_sha256),
                    "artifact_names": list(result.artifacts.artifact_names),
                }
            )
            _print(payload)
            return 0

        if args.command in {"analyze", "report"}:
            summary = export_closed_run(args.run_dir)
            _print(
                {
                    "command": args.command,
                    "run_directory": str(summary.output_directory),
                    "manifest_sha256": summary.manifest_sha256,
                    "scientific_result_sha256": summary.scientific_result_sha256,
                    "committed_turns": summary.committed_turns,
                    "artifact_names": list(summary.artifact_names),
                    "scientific_decision": None,
                }
            )
            return 0
    except Exception as exc:
        print(
            canonical_json(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    raise AssertionError(f"unhandled command: {args.command}")

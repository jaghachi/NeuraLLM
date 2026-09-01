"""Explicit, machine-readable NeuraLLM command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from yaml import YAMLError

from neurallm import __version__
from neurallm.domain.serialization import canonical_json
from neurallm.experiments.preregistration import publish_preregistration
from neurallm.experiments.runner import GenerationDispatchError
from neurallm.experiments.static_selection import freeze_static_selection
from neurallm.experiments.workflow import (
    LiveProviderAuthorizationError,
    PreparedExperiment,
    execute_prepared,
    prepare_experiment,
)
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.providers import LlamaCppProviderConfig, LlamaCppProviderError, preflight_llama_cpp
from neurallm.reporting import CLOSED_RUN_ARTIFACTS, export_closed_run
from neurallm.reporting.status_cli import add_status_subcommand, build_status_payload
from neurallm.storage import StorageError

EXPECTED_CLI_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    YAMLError,
    LiveProviderAuthorizationError,
    GenerationDispatchError,
    LlamaCppProviderError,
    StorageError,
)


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
    add_status_subcommand(subparsers)

    preflight = subparsers.add_parser(
        "preflight",
        help="inspect one explicit llama.cpp identity without requesting a completion",
    )
    preflight.add_argument(
        "--provider-config",
        type=Path,
        required=True,
        help="explicit machine-local llama.cpp provider YAML path",
    )

    preregister = subparsers.add_parser(
        "preregister",
        help="publish a frozen confirmatory identity without constructing a provider",
    )
    _add_config_argument(preregister)
    preregister.add_argument(
        "--output",
        type=Path,
        required=True,
        help="explicit canonical JSON preregistration seal output path",
    )

    freeze_selection = subparsers.add_parser(
        "freeze-static-selection",
        help="derive and publish best_static evidence from finalized live pilot runs",
    )
    freeze_selection.add_argument(
        "--candidate-run-dir",
        action="append",
        type=Path,
        required=True,
        dest="candidate_run_directories",
        help="finalized llama.cpp development-pilot run directory; repeat for each profile",
    )
    freeze_selection.add_argument(
        "--candidate-grid",
        type=Path,
        required=True,
        help="exact canonical JSON candidate grid committed before pilot execution",
    )
    freeze_selection.add_argument(
        "--output",
        type=Path,
        required=True,
        help="canonical JSON static-selection evidence output path",
    )
    preregister.add_argument(
        "--sealed-config-output",
        type=Path,
        help=(
            "optional executable YAML with the seal embedded; must be beside --config "
            "so relative references remain unchanged"
        ),
    )

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
    run.add_argument(
        "--allow-live-provider",
        action="store_true",
        help=(
            "additionally authorize llama.cpp network inspection and completion dispatch; "
            "not required for the fake provider"
        ),
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
                build_status_payload(
                    tuple(args.status_run_directories or ()),
                    tuple(args.status_artifacts or ()),
                    args.candidate_grid,
                    package_version=__version__,
                )
            )
            return 0

        if args.command == "preflight":
            provider_config_path = args.provider_config.expanduser().resolve(strict=True)
            provider_config = LlamaCppProviderConfig.model_validate(
                load_yaml_mapping(provider_config_path)
            )
            preflight = preflight_llama_cpp(provider_config)
            _print(
                {
                    "command": "preflight",
                    "provider_config_path": str(provider_config_path),
                    **preflight.model_dump(mode="json"),
                }
            )
            return 0

        if args.command == "preregister":
            publication = publish_preregistration(
                args.config,
                args.output,
                args.sealed_config_output,
            )
            payload: dict[str, object] = {
                "command": "preregister",
                "config_path": str(publication.config_path),
                "dataset_path": str(publication.dataset_path),
                "output_path": str(publication.output_path),
                "experiment_id": publication.seal.experiment_id,
                "run_tier": publication.seal.run_tier,
                "scientific_identity_sha256": (publication.scientific_identity_sha256),
                "preregistration_sha256": publication.preregistration_sha256,
                "created": publication.created,
                "provider_constructed": False,
                "network_requested": False,
            }
            if publication.development_selection_dataset_path is not None:
                payload["development_selection_dataset_path"] = str(
                    publication.development_selection_dataset_path
                )
            if publication.sealed_config_path is not None:
                payload["sealed_config_path"] = str(publication.sealed_config_path)
                payload["sealed_config_created"] = publication.sealed_config_created
            _print(payload)
            return 0

        if args.command == "freeze-static-selection":
            selection_publication = freeze_static_selection(
                tuple(args.candidate_run_directories),
                args.candidate_grid,
                args.output,
            )
            _print(
                {
                    "command": "freeze-static-selection",
                    "output_path": str(selection_publication.output_path),
                    "created": selection_publication.created,
                    "candidate_count": len(selection_publication.evidence.candidates),
                    "source_run_directories": [
                        str(path) for path in selection_publication.source_run_directories
                    ],
                    "candidate_grid_path": str(selection_publication.candidate_grid_path),
                    "candidate_grid_sha256": (selection_publication.evidence.candidate_grid_sha256),
                    "winning_profile": (
                        selection_publication.evidence.selection_record.winning_profile
                    ),
                    "selection_result_sha256": (
                        selection_publication.evidence.selection_record.selection_result_sha256
                    ),
                    "static_selection_evidence_sha256": (
                        selection_publication.evidence.evidence_sha256
                    ),
                    "provider_constructed": False,
                    "network_requested": False,
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
            if (
                prepared.loaded_config.config.provider.kind == "llama_cpp"
                and not args.allow_live_provider
            ):
                raise LiveProviderAuthorizationError(
                    "llama_cpp execution requires --allow-live-provider in addition to "
                    "--execute; no provider was constructed"
                )
            result = execute_prepared(
                prepared,
                allow_live_provider=args.allow_live_provider,
            )
            payload.update(
                {
                    "command": "run",
                    "mode": "execute",
                    "provider_calls": result.execution.provider_calls,
                    "previously_committed_turns": (result.execution.previously_committed_turns),
                    "dispatched_this_invocation": (result.execution.dispatched_this_invocation),
                    "successful_responses_this_invocation": (
                        result.execution.successful_responses_this_invocation
                    ),
                    "uncertain_dispatches_this_invocation": (
                        result.execution.uncertain_dispatches_this_invocation
                    ),
                    "committed_turns": result.execution.committed_turns,
                    "manifest_sha256": result.execution.manifest_sha256,
                    "scientific_result_sha256": (result.artifacts.scientific_result_sha256),
                    "artifact_names": list(result.artifacts.artifact_names),
                    "implementation_phase": result.artifacts.implementation_phase,
                    "phase3_baseline_evaluator_verdict": (
                        result.artifacts.phase3_baseline_evaluator_verdict
                    ),
                    "scientific_decision": result.artifacts.scientific_decision,
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
                    "implementation_phase": summary.implementation_phase,
                    "phase3_baseline_evaluator_verdict": (
                        summary.phase3_baseline_evaluator_verdict
                    ),
                    "scientific_decision": summary.scientific_decision,
                }
            )
            return 0
    except EXPECTED_CLI_EXCEPTIONS as exc:
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

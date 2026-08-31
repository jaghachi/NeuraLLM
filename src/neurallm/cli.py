"""Explicit, machine-readable NeuraLLM command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from neurallm import __version__
from neurallm.domain.serialization import canonical_json
from neurallm.experiments.preregistration import publish_preregistration
from neurallm.experiments.workflow import (
    LiveProviderAuthorizationError,
    PreparedExperiment,
    execute_prepared,
    prepare_experiment,
)
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.providers import LlamaCppProviderConfig, preflight_llama_cpp
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
                {
                    "package": "neurallm",
                    "version": __version__,
                    "implementation_phase": 5,
                    "phase_2_kernel_available": True,
                    "phase_3_baseline_evaluator_available": True,
                    "phase_4_causal_attribution_available": True,
                    "model_backed_protocol_available": True,
                    "confirmatory_decision_engine_available": True,
                    "offline_engineering_smoke_config": (
                        "configs/experiments/model-backed-engineering-smoke.yaml"
                    ),
                    "live_smoke_template": (
                        "configs/experiments/model-backed-live-smoke.example.yaml"
                    ),
                    "readiness": "READY_FOR_LIVE_SMOKE",
                    "scientific_decision": None,
                    "live_provider_validated": False,
                    "live_smoke_completed": False,
                    "confirmatory_run_completed": False,
                }
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
            publication = publish_preregistration(args.config, args.output)
            _print(
                {
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

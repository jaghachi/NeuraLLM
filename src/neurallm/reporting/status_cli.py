"""Command-line parser and payload helpers for verified project status."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from neurallm.reporting.status import load_verified_status


def add_status_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the explicit, provider-free status command."""

    status = subparsers.add_parser("status", help="print implementation and scientific status")
    status_sources = status.add_mutually_exclusive_group()
    status_sources.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        dest="status_run_directories",
        help="explicit closed-run directory to verify; repeat for multiple tiers",
    )
    status_sources.add_argument(
        "--status-artifact",
        action="append",
        type=Path,
        dest="status_artifacts",
        help="explicit adjacent decision.json to verify; repeat for multiple tiers",
    )
    status.add_argument(
        "--candidate-grid",
        type=Path,
        help=(
            "explicit development-pilot candidate-grid canonical JSON; required once "
            "the declared pilot grid may be complete"
        ),
    )


def build_status_payload(
    run_directories: Sequence[Path],
    status_artifacts: Sequence[Path],
    candidate_grid_path: Path | None,
    *,
    package_version: str,
) -> dict[str, object]:
    """Verify explicit evidence and build the machine-readable status payload."""

    status = load_verified_status(
        run_directories,
        status_artifacts,
        candidate_grid_path,
    )
    return {
        "package": "neurallm",
        "version": package_version,
        "implementation_phase": 5,
        "phase_2_kernel_available": True,
        "phase_3_baseline_evaluator_available": True,
        "phase_4_causal_attribution_available": True,
        "model_backed_protocol_available": True,
        "confirmatory_decision_engine_available": True,
        "offline_engineering_smoke_config": (
            "configs/experiments/model-backed-engineering-smoke.yaml"
        ),
        "live_smoke_template": "configs/experiments/model-backed-live-smoke.example.yaml",
        **status.model_dump(mode="json"),
    }

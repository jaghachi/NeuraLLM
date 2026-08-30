"""Minimal, zero-network Phase 1 command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from neurallm import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurallm",
        description="Deterministic research software for neural decoding control.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="print the implementation and scientific status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a CLI command without constructing a provider."""

    args = _build_parser().parse_args(argv)
    if args.command == "status":
        print(
            json.dumps(
                {
                    "package": "neurallm",
                    "version": __version__,
                    "implementation_phase": 1,
                    "scientific_decision": None,
                    "live_provider_validated": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")

"""Tests for the zero-network Phase 1 CLI."""

from __future__ import annotations

import importlib
import json
from typing import Any

from neurallm.cli import main


def test_status_is_machine_readable_and_truthful(capsys: Any) -> None:
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "implementation_phase": 1,
        "live_provider_validated": False,
        "package": "neurallm",
        "scientific_decision": None,
        "version": "2.0.0a1",
    }


def test_main_module_import_has_no_cli_side_effect(capsys: Any) -> None:
    module = importlib.import_module("neurallm.__main__")

    assert module.main is main
    assert capsys.readouterr().out == ""

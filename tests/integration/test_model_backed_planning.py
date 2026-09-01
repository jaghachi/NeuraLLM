"""Provider-free validation and planning for offline model-backed tiers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import pytest

from neurallm.cli import main
from neurallm.evaluation import DatasetPurpose
from neurallm.experiments import GitProvenance, build_plan, load_dataset
from neurallm.experiments.config import load_experiment_config
from neurallm.experiments.protocol import (
    ATTRIBUTION_POLICY_ID,
    EFFICACY_POLICY_IDS,
    MODEL_BACKED_POLICY_IDS,
    RunTier,
)
from neurallm.experiments.yaml_loader import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "experiments"


@pytest.mark.parametrize(
    (
        "config_name",
        "tier",
        "sequence_count",
        "turns_per_sequence",
        "model_seed_count",
        "model_seeds",
        "logical_generation_count",
        "decision_rule_version",
        "dataset_name",
        "dataset_sha256",
    ),
    (
        (
            "model-backed-engineering-smoke.yaml",
            RunTier.ENGINEERING_SMOKE,
            2,
            2,
            1,
            (4101,),
            20,
            "engineering-smoke-no-scientific-decision-v1",
            "model-backed-engineering-smoke-v1.yaml",
            "14c382a04acbe9394474f05cf84d8389833058afc2dc6feda21a023d46e45ef3",
        ),
        (
            "model-backed-development-pilot.yaml",
            RunTier.DEVELOPMENT_PILOT,
            6,
            4,
            2,
            (4101, 4102),
            240,
            "development-pilot-no-scientific-decision-v1",
            "phase3-baseline-development-v1.yaml",
            "a6c41a046cb84bc9a806866a7393196784eb769118f74cbe4d44d0f3e247df97",
        ),
    ),
)
def test_offline_tiers_validate_plan_and_dry_run_without_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    config_name: str,
    tier: RunTier,
    sequence_count: int,
    turns_per_sequence: int,
    model_seed_count: int,
    model_seeds: tuple[int, ...],
    logical_generation_count: int,
    decision_rule_version: str,
    dataset_name: str,
    dataset_sha256: str,
) -> None:
    def forbidden_provider_or_network(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("offline validation and planning must remain provider-free")

    monkeypatch.setattr(
        "neurallm.providers.fake.FakeProvider.__init__",
        forbidden_provider_or_network,
    )
    monkeypatch.setattr(
        "neurallm.providers.llama_cpp.LlamaCppProvider.__init__",
        forbidden_provider_or_network,
    )
    monkeypatch.setattr("httpx.Client.__init__", forbidden_provider_or_network)
    monkeypatch.setattr("neurallm.cli.execute_prepared", forbidden_provider_or_network)
    monkeypatch.setattr(
        "neurallm.experiments.workflow.read_git_provenance",
        lambda _path: GitProvenance(source_commit="0" * 40, working_tree_clean=False),
    )

    config_path = CONFIG_ROOT / config_name
    loaded_config = load_experiment_config(config_path)
    loaded_dataset = load_dataset(loaded_config.dataset_path)
    plan = build_plan(loaded_config, loaded_dataset)

    assert loaded_config.config.provider.kind == "fake"
    assert loaded_config.provider_config_path is None
    assert loaded_config.dataset_path == ROOT / "datasets" / "development" / dataset_name
    assert loaded_dataset.dataset.purpose is DatasetPurpose.DEVELOPMENT
    assert loaded_dataset.dataset.dataset_hash == dataset_sha256
    assert len(loaded_dataset.dataset.sequences) == sequence_count
    assert {len(sequence.cases) for sequence in loaded_dataset.dataset.sequences} == {
        turns_per_sequence
    }
    assert loaded_config.config.model_seeds == model_seeds
    assert loaded_config.config.controller_seeds == (5101,)
    assert plan.protocol is not None
    assert plan.protocol.run_tier is tier
    assert plan.protocol.policy_ids == MODEL_BACKED_POLICY_IDS
    assert plan.protocol.efficacy_policy_ids == EFFICACY_POLICY_IDS
    assert plan.protocol.attribution.policy_id == ATTRIBUTION_POLICY_ID
    assert ATTRIBUTION_POLICY_ID not in plan.protocol.efficacy_policy_ids
    assert plan.protocol.schedule.sequence_count == sequence_count
    assert plan.protocol.schedule.turns_per_sequence == turns_per_sequence
    assert plan.protocol.schedule.model_seed_count == model_seed_count
    assert plan.protocol.schedule.controller_seed_count == 1
    assert plan.protocol.schedule.policy_count == 5
    assert plan.protocol.schedule.logical_generation_count == logical_generation_count
    assert len(plan.turns) == logical_generation_count
    assert Counter(turn.condition.policy_id for turn in plan.turns) == {
        policy_id: logical_generation_count // 5 for policy_id in MODEL_BACKED_POLICY_IDS
    }

    assert plan.decision_rule_version == decision_rule_version
    assert tier is not RunTier.CONFIRMATORY
    assert "no-scientific-decision" in plan.decision_rule_version
    assert plan.evaluation is None
    assert plan.evaluation_spec_sha256 is None
    assert plan.preregistration is None
    assert plan.dataset_seal is None
    assert plan.matched_units == ()

    assert main(["validate", "--config", str(config_path)]) == 0
    validate_payload = json.loads(capsys.readouterr().out)
    assert validate_payload["valid"] is True
    assert validate_payload["provider_kind"] == "fake"
    assert validate_payload["planned_turns"] == logical_generation_count

    assert main(["plan", "--config", str(config_path)]) == 0
    plan_payload = json.loads(capsys.readouterr().out)
    assert plan_payload["provider_kind"] == "fake"
    assert plan_payload["planned_turns"] == logical_generation_count
    assert len(plan_payload["schedule"]) == logical_generation_count

    assert main(["run", "--config", str(config_path), "--dry-run"]) == 0
    dry_run_payload = json.loads(capsys.readouterr().out)
    assert dry_run_payload["provider_constructed"] is False
    assert dry_run_payload["network_requested"] is False
    assert dry_run_payload["planned_turns"] == logical_generation_count
    assert dry_run_payload["schedule"] == plan_payload["schedule"]


def test_live_smoke_example_is_an_unsealed_llama_template_for_exact_twenty_request_grid() -> None:
    payload = load_yaml_mapping(CONFIG_ROOT / "model-backed-live-smoke.example.yaml")

    assert payload["provider"]["kind"] == "llama_cpp"
    assert payload["provider"]["config_path"] == "../providers/llama_cpp.local.yaml"
    assert "PASTE" in payload["provider"]["expected_identity"]["model_alias"]
    assert "PASTE" in payload["provider"]["expected_effective_configuration_json"]
    assert payload["protocol"]["run_tier"] == "engineering_smoke"
    assert payload["protocol"]["schedule"] == {
        "sequence_count": 2,
        "turns_per_sequence": 2,
        "model_seed_count": 1,
        "controller_seed_count": 1,
        "policy_count": 5,
        "logical_generation_count": 20,
    }
    assert payload["provider"]["kind"] != "fake"
    assert "preregistration" not in payload
    assert "evaluation" not in payload

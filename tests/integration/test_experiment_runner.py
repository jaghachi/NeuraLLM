"""End-to-end and crash/resume tests for the Phase 2 experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurallm.domain.models import ProviderIdentity
from neurallm.experiments import (
    GitProvenance,
    build_fixed_policy_runtimes,
    build_plan,
    build_run_manifest,
    execute_plan,
    load_dataset,
    load_experiment_config,
)
from neurallm.experiments.plan import ExperimentPlan
from neurallm.providers.base import GenerationRequest, GenerationResponse
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import export_closed_run
from neurallm.storage import SQLiteRunStore, TurnState, UncertainDispatchError


class CountingFakeProvider:
    """Record generation calls while retaining the real deterministic fake contract."""

    def __init__(self) -> None:
        self._delegate = FakeProvider()
        self.calls = 0

    @property
    def provider_identity(self) -> ProviderIdentity:
        return self._delegate.provider_identity

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return self._delegate.generate(request)


def _smoke_plan() -> ExperimentPlan:
    root = Path(__file__).resolve().parents[2]
    loaded_config = load_experiment_config(root / "configs" / "experiments" / "smoke.yaml")
    loaded_dataset = load_dataset(
        loaded_config.dataset_path,
        expected_version=loaded_config.config.dataset.version,
    )
    return build_plan(loaded_config, loaded_dataset)


def _manifest(plan: ExperimentPlan, provider: CountingFakeProvider):
    runtimes = build_fixed_policy_runtimes(plan)
    manifest = build_run_manifest(
        plan,
        provider.provider_identity,
        runtimes,
        GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    return manifest, runtimes


def test_fake_plan_executes_end_to_end_and_committed_replay_makes_zero_calls(
    tmp_path: Path,
) -> None:
    plan = _smoke_plan()
    provider = CountingFakeProvider()
    manifest, runtimes = _manifest(plan, provider)
    database_path = tmp_path / "run.sqlite3"

    first = execute_plan(plan, manifest, provider, runtimes, database_path)
    repeated = execute_plan(plan, manifest, provider, runtimes, database_path)

    assert first.planned_turns == 3
    assert first.committed_turns == 3
    assert first.provider_calls == 3
    assert repeated.provider_calls == 0
    assert repeated.manifest_sha256 == first.manifest_sha256
    assert provider.calls == 3

    with SQLiteRunStore(database_path) as store:
        turns = store.list_turns()
        finalization = store.get_finalization()
        assert len(turns) == 3
        assert finalization is not None
        assert finalization.expected_condition_count == 3
        assert finalization == store.get_finalization()
        assert all(turn.state is TurnState.COMMITTED for turn in turns)
        trace = json.loads(turns[0].policy_trace_json or "null")
        assert set(trace["action_application"]) == {
            "final_decoding_parameters",
            "raw_action",
            "saturation",
            "step_clamped_action",
        }
        assert trace["action"] == trace["action_application"]["step_clamped_action"]
        assert trace["action_application"]["final_decoding_parameters"]["max_tokens"] == 64


def test_crash_after_first_committed_turn_rejects_closed_run_export(tmp_path: Path) -> None:
    plan = _smoke_plan()
    provider = CountingFakeProvider()
    manifest, runtimes = _manifest(plan, provider)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    database_path = run_directory / "run.sqlite3"

    def crash_after_first_commit(state: TurnState, _turn: object) -> None:
        if state is TurnState.COMMITTED:
            raise RuntimeError("simulated crash after first commit")

    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_plan(
            plan,
            manifest,
            provider,
            runtimes,
            database_path,
            checkpoint_hook=crash_after_first_commit,
        )

    with pytest.raises(ValueError, match="not finalized"):
        export_closed_run(run_directory)


def test_resume_after_persisted_response_never_regenerates_that_turn(tmp_path: Path) -> None:
    plan = _smoke_plan()
    provider = CountingFakeProvider()
    manifest, runtimes = _manifest(plan, provider)
    database_path = tmp_path / "run.sqlite3"
    crashed = False

    def crash_once(state: TurnState, _turn: object) -> None:
        nonlocal crashed
        if state is TurnState.RESPONSE_PERSISTED and not crashed:
            crashed = True
            raise RuntimeError("simulated crash after durable response")

    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_plan(
            plan,
            manifest,
            provider,
            runtimes,
            database_path,
            checkpoint_hook=crash_once,
        )
    assert provider.calls == 1

    resumed = execute_plan(plan, manifest, provider, runtimes, database_path)

    assert resumed.provider_calls == 2
    assert provider.calls == 3
    with SQLiteRunStore(database_path) as store:
        assert len(store.list_turns(TurnState.COMMITTED)) == 3


def test_resume_from_ambiguous_dispatch_fails_closed_without_provider_call(
    tmp_path: Path,
) -> None:
    plan = _smoke_plan()
    provider = CountingFakeProvider()
    manifest, runtimes = _manifest(plan, provider)
    database_path = tmp_path / "run.sqlite3"

    def crash_before_provider(state: TurnState, _turn: object) -> None:
        if state is TurnState.DISPATCHING:
            raise RuntimeError("simulated process loss before response persistence")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        execute_plan(
            plan,
            manifest,
            provider,
            runtimes,
            database_path,
            checkpoint_hook=crash_before_provider,
        )
    assert provider.calls == 0

    with pytest.raises(UncertainDispatchError, match="cannot be retried"):
        execute_plan(plan, manifest, provider, runtimes, database_path)
    assert provider.calls == 0
    with SQLiteRunStore(database_path) as store:
        assert len(store.list_turns(TurnState.UNCERTAIN_DISPATCH)) == 1


def test_manifest_rejects_provider_identity_drift() -> None:
    plan = _smoke_plan()
    runtimes = build_fixed_policy_runtimes(plan)
    drifted = plan.provider_identity.model_copy(update={"build_id": "drifted"})

    with pytest.raises(ValueError, match="does not .*match"):
        build_run_manifest(
            plan,
            drifted,
            runtimes,
            GitProvenance(source_commit="0" * 40, working_tree_clean=True),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "experiment_config_hash",
        "dataset_hash",
        "provider_config_hash",
        "provider_identity",
        "provider_effective_configuration_json",
        "policy_config_hashes",
        "metric_versions",
        "seed_schedule",
        "action_bounds",
        "decoding_bounds",
        "decision_rule_version",
        "database_schema_version",
    ),
)
def test_execute_rejects_every_manifest_scientific_binding_drift(
    field_name: str,
    tmp_path: Path,
) -> None:
    plan = _smoke_plan()
    provider = CountingFakeProvider()
    manifest, runtimes = _manifest(plan, provider)
    drift_by_field = {
        "experiment_config_hash": "1" * 64,
        "dataset_hash": "2" * 64,
        "provider_config_hash": "3" * 64,
        "provider_identity": manifest.provider_identity.model_copy(update={"build_id": "drifted"}),
        "provider_effective_configuration_json": (
            '{"generation_method":"request_sha256_v2",'
            '"implementation_version":"1.0.0","provider_type":"fake"}'
        ),
        "policy_config_hashes": {"kernel_fixed": "4" * 64},
        "metric_versions": {"drifted_metric": "v1"},
        "seed_schedule": manifest.seed_schedule.model_copy(update={"model_seeds": (999,)}),
        "action_bounds": manifest.action_bounds.model_copy(
            update={"temperature_delta": (-0.09, 0.09)}
        ),
        "decoding_bounds": manifest.decoding_bounds.model_copy(update={"temperature": (0.02, 2.0)}),
        "decision_rule_version": "drifted-rule-v1",
        "database_schema_version": manifest.database_schema_version + 1,
    }
    drifted = manifest.model_copy(update={field_name: drift_by_field[field_name]})
    database_path = tmp_path / f"{field_name}.sqlite3"

    with pytest.raises(ValueError, match="does not .*match"):
        execute_plan(plan, drifted, provider, runtimes, database_path)

    assert provider.calls == 0
    assert not database_path.exists()

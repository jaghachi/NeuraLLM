"""Offline end-to-end proof for the complete five-arm engineering tier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from neurallm.domain.models import ProviderIdentity
from neurallm.experiments.runner import (
    GenerationDispatchError,
    GitProvenance,
    build_run_manifest,
    execute_plan,
)
from neurallm.experiments.workflow import execute_prepared, prepare_experiment
from neurallm.providers.base import GenerationRequest, GenerationResponse
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import CLOSED_RUN_ARTIFACTS
from neurallm.storage import SQLiteRunStore, TurnState, UncertainDispatchError


class CountingFakeProvider:
    """Count logical generation calls while retaining the strict fake contract."""

    def __init__(self) -> None:
        self._delegate = FakeProvider()
        self.calls = 0

    @property
    def provider_identity(self) -> ProviderIdentity:
        return self._delegate.provider_identity

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return self._delegate.generate(request)


class FailingFakeProvider:
    """Raise after dispatch so the runner must persist transport ambiguity."""

    def __init__(self) -> None:
        self._delegate = FakeProvider()
        self.calls = 0

    @property
    def provider_identity(self) -> ProviderIdentity:
        return self._delegate.provider_identity

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        del request
        self.calls += 1
        raise RuntimeError("provider transport failed after dispatch")


def _isolated_smoke_config(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    source = root / "configs" / "experiments" / "model-backed-engineering-smoke.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (root / "datasets" / "development" / "model-backed-engineering-smoke-v1.yaml")
        .resolve()
        .as_posix()
    )
    payload["artifact_root"] = str((tmp_path / "run").resolve().as_posix())
    target = tmp_path / "model-backed-engineering-smoke.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def test_five_arm_fake_smoke_executes_20_generations_and_replays_without_dispatch(
    tmp_path: Path,
) -> None:
    config_path = _isolated_smoke_config(tmp_path)
    provenance = GitProvenance(source_commit="0" * 40, working_tree_clean=True)
    prepared = prepare_experiment(config_path, provenance=provenance)

    first = execute_prepared(prepared)
    replay = execute_prepared(prepared)

    assert first.execution.planned_turns == 20
    assert first.execution.previously_committed_turns == 0
    assert first.execution.dispatched_this_invocation == 20
    assert first.execution.successful_responses_this_invocation == 20
    assert first.execution.uncertain_dispatches_this_invocation == 0
    assert first.execution.provider_calls == 20
    assert replay.execution.previously_committed_turns == 20
    assert replay.execution.dispatched_this_invocation == 0
    assert replay.execution.successful_responses_this_invocation == 0
    assert replay.execution.provider_calls == 0
    assert first.artifacts.implementation_phase == 5
    assert set(first.artifacts.artifact_names) == CLOSED_RUN_ARTIFACTS

    run_directory = prepared.loaded_config.artifact_root
    decision = json.loads((run_directory / "decision.json").read_text(encoding="utf-8"))
    assert decision["run_tier"] == "engineering_smoke"
    assert decision["scientific_decision"] is None
    assert decision["execution_accounting"] == {
        "planned_logical_generations": 20,
        "dispatched_logical_generations": 20,
        "successful_responses": 20,
        "uncertain_dispatches": 0,
        "committed_logical_generations": 20,
    }
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Engineering validity",
        "## Controller activity",
        "## End-to-end efficacy",
        "## Persistent-state attribution",
        "## Guardrail outcomes",
        "## Limitations",
        "## Final decision",
    ):
        assert heading in report

    with SQLiteRunStore(run_directory / "run.sqlite3") as store:
        finalization = store.get_finalization()
        assert finalization is not None
        assert finalization.execution_accounting is not None
        assert finalization.execution_accounting.committed_logical_generations == 20


def test_model_backed_resume_reuses_persisted_response_and_closes_exact_accounting(
    tmp_path: Path,
) -> None:
    prepared = prepare_experiment(
        _isolated_smoke_config(tmp_path),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    provider = CountingFakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database_path = tmp_path / "persisted-response-resume.sqlite3"
    crashed = False

    def crash_once(state: TurnState, _turn: object) -> None:
        nonlocal crashed
        if state is TurnState.RESPONSE_PERSISTED and not crashed:
            crashed = True
            raise RuntimeError("simulated crash after durable model-backed response")

    with pytest.raises(RuntimeError, match="after durable model-backed response"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database_path,
            checkpoint_hook=crash_once,
        )
    assert provider.calls == 1

    resumed = execute_plan(
        prepared.plan,
        manifest,
        provider,
        prepared.policy_runtimes,
        database_path,
    )
    assert resumed.planned_turns == 20
    assert resumed.previously_committed_turns == 0
    assert resumed.dispatched_this_invocation == 19
    assert resumed.successful_responses_this_invocation == 19
    assert resumed.uncertain_dispatches_this_invocation == 0
    assert resumed.committed_turns == 20
    assert resumed.provider_calls == 19
    assert provider.calls == 20

    replay = execute_plan(
        prepared.plan,
        manifest,
        provider,
        prepared.policy_runtimes,
        database_path,
    )
    assert replay.planned_turns == 20
    assert replay.previously_committed_turns == 20
    assert replay.dispatched_this_invocation == 0
    assert replay.successful_responses_this_invocation == 0
    assert replay.uncertain_dispatches_this_invocation == 0
    assert replay.committed_turns == 20
    assert replay.provider_calls == 0
    assert provider.calls == 20

    with SQLiteRunStore(database_path) as store:
        finalization = store.get_finalization()
        assert finalization is not None
        assert finalization.execution_accounting is not None
        assert finalization.execution_accounting.model_dump(mode="json") == {
            "planned_logical_generations": 20,
            "dispatched_logical_generations": 20,
            "successful_responses": 20,
            "uncertain_dispatches": 0,
            "committed_logical_generations": 20,
        }


def test_model_backed_ambiguous_dispatch_fails_closed_without_provider_call(
    tmp_path: Path,
) -> None:
    prepared = prepare_experiment(
        _isolated_smoke_config(tmp_path),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    provider = CountingFakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database_path = tmp_path / "ambiguous-dispatch.sqlite3"

    def crash_before_provider(state: TurnState, _turn: object) -> None:
        if state is TurnState.DISPATCHING:
            raise RuntimeError("simulated model-backed loss at dispatch boundary")

    with pytest.raises(RuntimeError, match="loss at dispatch boundary"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database_path,
            checkpoint_hook=crash_before_provider,
        )
    assert provider.calls == 0

    with pytest.raises(UncertainDispatchError, match="cannot be retried"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database_path,
        )
    assert provider.calls == 0

    with SQLiteRunStore(database_path) as store:
        assert store.get_finalization() is None
        assert len(store.list_turns(TurnState.UNCERTAIN_DISPATCH)) == 1


def test_model_backed_provider_failure_is_durably_uncertain_and_never_retried(
    tmp_path: Path,
) -> None:
    prepared = prepare_experiment(
        _isolated_smoke_config(tmp_path),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    provider = FailingFakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    database_path = tmp_path / "provider-failure.sqlite3"

    with pytest.raises(GenerationDispatchError, match="transport failed after dispatch"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database_path,
        )
    assert provider.calls == 1

    with SQLiteRunStore(database_path) as store:
        uncertain = store.list_turns(TurnState.UNCERTAIN_DISPATCH)
        assert len(uncertain) == 1
        assert uncertain[0].uncertain_reason == (
            "RuntimeError: provider transport failed after dispatch"
        )
        assert store.get_finalization() is None

    with pytest.raises(UncertainDispatchError, match="cannot be retried"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            database_path,
        )
    assert provider.calls == 1


def test_model_backed_execution_requires_an_explicit_database_path_type(
    tmp_path: Path,
) -> None:
    prepared = prepare_experiment(
        _isolated_smoke_config(tmp_path),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    provider = FakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )

    with pytest.raises(TypeError, match="pathlib.Path"):
        execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            str(tmp_path / "run.sqlite3"),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "drifted_value"),
    (
        ("matched_history_policy_sources", {}),
        ("run_tier", "development_pilot"),
        ("scientific_identity_sha256", "f" * 64),
        ("preregistration_sha256", "e" * 64),
        ("confirmatory_analysis_contract_sha256", "d" * 64),
    ),
)
def test_model_backed_execution_rejects_protocol_manifest_field_drift(
    field_name: str,
    drifted_value: object,
    tmp_path: Path,
) -> None:
    prepared = prepare_experiment(
        _isolated_smoke_config(tmp_path),
        provenance=GitProvenance(source_commit="0" * 40, working_tree_clean=True),
    )
    provider = FakeProvider()
    manifest = build_run_manifest(
        prepared.plan,
        provider.provider_identity,
        prepared.policy_runtimes,
        prepared.provenance,
    )
    drifted = manifest.model_copy(update={field_name: drifted_value})
    database_path = tmp_path / f"{field_name}.sqlite3"

    with pytest.raises(ValueError, match="does not exactly match"):
        execute_plan(
            prepared.plan,
            drifted,
            provider,
            prepared.policy_runtimes,
            database_path,
        )

    assert not database_path.exists()

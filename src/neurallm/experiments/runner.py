"""Deterministic Phase 2 execution and crash-safe resume orchestration."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from neurallm.control.action_space import ActionApplication, apply_action
from neurallm.control.policy import (
    ControlPolicy,
    PolicyContext,
    PolicyState,
    PolicyTrace,
)
from neurallm.control.static import FixedPolicy
from neurallm.domain.models import (
    ControllerObservation,
    ProviderIdentity,
    RunManifest,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.experiments.plan import ExperimentPlan, PlannedTurn
from neurallm.metrics.base import MetricContext
from neurallm.metrics.deterministic import compute_response_metrics
from neurallm.providers.base import (
    GenerationProvider,
    GenerationRequest,
)
from neurallm.reporting.artifacts import scientific_result_sha256
from neurallm.storage import (
    ResumeAction,
    SQLiteRunStore,
    StoreInvariantError,
    TurnState,
)


class AppliedPolicyTrace(PolicyTrace):
    """Policy decision plus every auditable action-clamping stage."""

    action_application: ActionApplication


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """Exact source state bound into a run manifest."""

    source_commit: str
    working_tree_clean: bool


@dataclass(frozen=True, slots=True)
class PolicyRuntime:
    """A policy together with its declared persisted state type."""

    policy: ControlPolicy
    state_type: type[PolicyState]
    config_sha256: str


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Counts and identities from one execution or safe resume."""

    planned_turns: int
    committed_turns: int
    provider_calls: int
    manifest_sha256: str


class CheckpointHook(Protocol):
    """Test and supervision hook invoked only after durable checkpoints."""

    def __call__(self, state: TurnState, turn: PlannedTurn) -> None: ...


def read_git_provenance(anchor: Path) -> GitProvenance:
    """Read the exact repository commit and whole-worktree cleanliness."""

    if not isinstance(anchor, Path):
        raise TypeError("anchor must be a pathlib.Path")
    anchor = anchor.expanduser().resolve(strict=True)
    working_directory = anchor if anchor.is_dir() else anchor.parent

    def run_git(arguments: list[str]) -> str:
        completed = subprocess.run(
            ["git", "-C", str(working_directory), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    source_commit = run_git(["rev-parse", "HEAD"])
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("git returned an invalid lowercase commit SHA")
    status = run_git(["status", "--porcelain=v1", "--untracked-files=normal"])
    return GitProvenance(source_commit=source_commit, working_tree_clean=not status)


def build_fixed_policy_runtimes(plan: ExperimentPlan) -> Mapping[str, PolicyRuntime]:
    """Build the only policy runtime authorized in Phase 2."""

    policy_ids = sorted({turn.condition.policy_id for turn in plan.turns})
    unsupported = tuple(policy_id for policy_id in policy_ids if policy_id != "kernel_fixed")
    if unsupported:
        raise ValueError(f"Phase 2 execution does not implement policies: {unsupported!r}")
    return {
        policy_id: PolicyRuntime(
            policy=FixedPolicy(policy_id),
            state_type=PolicyState,
            config_sha256=canonical_sha256(
                {
                    "policy_type": "fixed",
                    "implementation_version": "phase2-fixed-v1",
                    "policy_id": policy_id,
                }
            ),
        )
        for policy_id in policy_ids
    }


def build_run_manifest(
    plan: ExperimentPlan,
    provider_identity: ProviderIdentity,
    policy_runtimes: Mapping[str, PolicyRuntime],
    provenance: GitProvenance,
) -> RunManifest:
    """Bind the complete scientific and source identity before execution."""

    if provider_identity != plan.provider_identity:
        raise ValueError("provider identity does not exactly match the experiment plan")
    planned_policy_ids = {turn.condition.policy_id for turn in plan.turns}
    if set(policy_runtimes) != planned_policy_ids:
        raise ValueError("policy runtimes do not exactly cover the experiment plan")
    return RunManifest(
        source_commit=provenance.source_commit,
        working_tree_clean=provenance.working_tree_clean,
        experiment_config_hash=plan.experiment_config_hash,
        dataset_hash=plan.dataset_hash,
        provider_config_hash=provider_identity.provider_config_hash,
        provider_identity=provider_identity,
        provider_effective_configuration_json=plan.provider_effective_configuration_json,
        policy_config_hashes={
            policy_id: runtime.config_sha256
            for policy_id, runtime in sorted(policy_runtimes.items())
        },
        metric_versions=plan.metric_versions,
        seed_schedule=SeedSchedule(
            model_seeds=tuple(sorted({turn.condition.model_seed for turn in plan.turns})),
            controller_seeds=tuple(sorted({turn.condition.controller_seed for turn in plan.turns})),
        ),
        action_bounds=plan.action_bounds,
        decoding_bounds=plan.decoding_bounds,
        decision_rule_version=plan.decision_rule_version,
        database_schema_version=plan.database_schema_version,
    )


def _trajectory_key(turn: PlannedTurn) -> tuple[object, ...]:
    condition = turn.condition
    return (
        condition.experiment_id,
        condition.dataset_version,
        condition.prompt_sequence_id,
        condition.policy_id,
        condition.model_seed,
        condition.controller_seed,
        condition.provider_identity_id,
        condition.base_decoding_profile_id,
    )


def _policy_decision(
    store: SQLiteRunStore,
    turn: PlannedTurn,
    previous_condition_id: str | None,
    runtime: PolicyRuntime,
    plan: ExperimentPlan,
) -> tuple[GenerationRequest, PolicyState, AppliedPolicyTrace]:
    if previous_condition_id is None:
        if turn.condition.turn_index != 0:
            raise StoreInvariantError("nonzero plan turn has no immediate predecessor")
        previous_metrics = None
        policy_state = runtime.policy.initial_state(
            PolicyContext(
                condition=turn.condition,
                initial_decoding_parameters=turn.decoding_parameters,
                action_bounds=plan.action_bounds,
            )
        )
    else:
        if turn.condition.turn_index == 0:
            raise StoreInvariantError("turn zero cannot bind previous history")
        committed = store.get_committed_history(previous_condition_id)
        previous_metrics = committed.metrics
        policy_state = store.load_policy_state(previous_condition_id, runtime.state_type)

    observation = ControllerObservation(
        turn_index=turn.condition.turn_index,
        prompt_family=turn.prompt_family,
        current_prompt_features=turn.prompt_features,
        previous_response_metrics=previous_metrics,
        has_previous_response=previous_metrics is not None,
    )
    raw_action, next_state, proposed_trace = runtime.policy.act(observation, policy_state)
    if proposed_trace.policy_id != turn.condition.policy_id:
        raise ValueError("policy returned a trace for another policy")
    if proposed_trace.turn_index != turn.condition.turn_index:
        raise ValueError("policy returned a trace for another turn")
    if proposed_trace.action != raw_action:
        raise ValueError("policy trace action differs from the returned raw action")
    application = apply_action(
        turn.decoding_parameters,
        raw_action,
        plan.action_bounds,
        plan.decoding_bounds,
    )
    request = GenerationRequest(
        prompt=turn.prompt,
        decoding_parameters=application.final_decoding_parameters,
        condition=turn.condition,
    )
    trace = AppliedPolicyTrace(
        policy_id=proposed_trace.policy_id,
        turn_index=proposed_trace.turn_index,
        action=application.step_clamped_action,
        action_application=application,
    )
    return request, next_state, trace


def execute_plan(
    plan: ExperimentPlan,
    manifest: RunManifest,
    provider: GenerationProvider,
    policy_runtimes: Mapping[str, PolicyRuntime],
    database_path: Path,
    *,
    checkpoint_hook: CheckpointHook | None = None,
) -> ExecutionSummary:
    """Execute or safely resume one plan without regenerating committed turns."""

    if provider.provider_identity != manifest.provider_identity:
        raise ValueError("provider identity does not match the run manifest")
    expected_manifest = build_run_manifest(
        plan,
        provider.provider_identity,
        policy_runtimes,
        GitProvenance(
            source_commit=manifest.source_commit,
            working_tree_clean=manifest.working_tree_clean,
        ),
    )
    if manifest != expected_manifest:
        raise ValueError(
            "run manifest does not exactly match the plan, provider, and policy runtimes"
        )
    if not isinstance(database_path, Path):
        raise TypeError("database_path must be a pathlib.Path")

    provider_calls = 0
    planned_ids = {turn.condition.condition_id for turn in plan.turns}
    previous_by_trajectory: dict[tuple[object, ...], str] = {}
    with SQLiteRunStore(database_path, manifest) as store:
        store.verify_integrity()
        unknown_ids = {turn.condition_id for turn in store.list_turns()} - planned_ids
        if unknown_ids:
            raise StoreInvariantError(
                f"run store contains conditions outside the current plan: {sorted(unknown_ids)!r}"
            )

        for turn in plan.turns:
            condition_id = turn.condition.condition_id
            trajectory = _trajectory_key(turn)
            previous_condition_id = previous_by_trajectory.get(trajectory)
            expected_previous_index = turn.condition.turn_index - 1
            if (previous_condition_id is None) != (turn.condition.turn_index == 0):
                raise StoreInvariantError(
                    f"plan trajectory is missing turn {expected_previous_index} before "
                    f"{turn.condition.turn_index}"
                )
            runtime = policy_runtimes.get(turn.condition.policy_id)
            if runtime is None:
                raise ValueError(f"no runtime for policy {turn.condition.policy_id!r}")
            request, next_state, applied_trace = _policy_decision(
                store,
                turn,
                previous_condition_id,
                runtime,
                plan,
            )
            history = (
                None
                if previous_condition_id is None
                else store.history_binding_for(previous_condition_id)
            )
            stored = store.prepare_turn(request, history)
            action = store.resume_action(condition_id)

            if action is ResumeAction.DISPATCH_PREPARED:
                store.begin_dispatch(condition_id)
                if checkpoint_hook is not None:
                    checkpoint_hook(TurnState.DISPATCHING, turn)
                try:
                    response = provider.generate(request)
                    provider_calls += 1
                except Exception as exc:
                    store.mark_dispatch_uncertain(
                        condition_id,
                        f"{type(exc).__name__}: {exc}",
                    )
                    raise
                stored = store.persist_response(condition_id, response)
                if checkpoint_hook is not None:
                    checkpoint_hook(TurnState.RESPONSE_PERSISTED, turn)
                action = ResumeAction.COMPUTE_METRICS

            if action is ResumeAction.COMPUTE_METRICS:
                if stored.response is None:
                    stored = store.get_turn(condition_id)
                if stored.response is None:
                    raise StoreInvariantError("response checkpoint is missing its response")
                metrics = compute_response_metrics(
                    MetricContext(
                        prompt_case_id=turn.prompt_case_id,
                        prompt_family=turn.prompt_family,
                        prompt=turn.prompt,
                        response_text=stored.response.text,
                        validator=turn.validator,
                    )
                )
                stored = store.persist_metrics(condition_id, metrics)
                if checkpoint_hook is not None:
                    checkpoint_hook(TurnState.METRICS_COMPUTED, turn)
                action = ResumeAction.COMMIT

            if action is ResumeAction.COMMIT:
                store.commit_turn(condition_id, next_state, applied_trace)
                if checkpoint_hook is not None:
                    checkpoint_hook(TurnState.COMMITTED, turn)
            elif action is ResumeAction.SKIP_COMMITTED:
                if (
                    stored.response is None
                    or stored.metrics is None
                    or stored.policy_state_json is None
                    or stored.policy_trace_json is None
                ):
                    raise StoreInvariantError("committed turn is missing replay evidence")
                expected_metrics = compute_response_metrics(
                    MetricContext(
                        prompt_case_id=turn.prompt_case_id,
                        prompt_family=turn.prompt_family,
                        prompt=turn.prompt,
                        response_text=stored.response.text,
                        validator=turn.validator,
                    )
                )
                if stored.metrics != expected_metrics:
                    raise StoreInvariantError("committed metrics do not reconstruct exactly")
                if stored.policy_state_json != canonical_json(next_state):
                    raise StoreInvariantError("committed policy state does not reconstruct exactly")
                if stored.policy_trace_json != canonical_json(applied_trace):
                    raise StoreInvariantError("committed policy trace does not reconstruct exactly")

            previous_by_trajectory[trajectory] = condition_id

        store.verify_integrity()
        stored_turns = store.list_turns()
        if len(stored_turns) != len(plan.turns) or any(
            turn.state is not TurnState.COMMITTED for turn in stored_turns
        ):
            raise StoreInvariantError("execution did not commit the complete planned schedule")
        finalization = store.finalize_run(
            tuple(planned_ids),
            scientific_result_sha256(stored_turns),
        )
        store.verify_integrity()
        store.compact()

    return ExecutionSummary(
        planned_turns=len(plan.turns),
        committed_turns=len(plan.turns),
        provider_calls=provider_calls,
        manifest_sha256=finalization.manifest_sha256,
    )


__all__ = [
    "AppliedPolicyTrace",
    "CheckpointHook",
    "ExecutionSummary",
    "GitProvenance",
    "PolicyRuntime",
    "build_fixed_policy_runtimes",
    "build_run_manifest",
    "execute_plan",
    "read_git_provenance",
]

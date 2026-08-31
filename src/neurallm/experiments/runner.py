"""Deterministic experiment execution and crash-safe resume orchestration."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self, assert_never, cast

from pydantic import SerializeAsAny, model_validator

from neurallm.control.action_space import ActionApplication, apply_action
from neurallm.control.heuristic import HeuristicAdaptivePolicy, HeuristicAdaptiveState
from neurallm.control.neural import (
    NeuralMatchedHistoryStateResetPolicy,
    NeuralPersistentPolicy,
    NeuralPolicyState,
    NeuralPolicyTrace,
    SimulatedNeuralPolicy,
)
from neurallm.control.policy import (
    ControlPolicy,
    PolicyContext,
    PolicyState,
    PolicyTrace,
)
from neurallm.control.random_policy import RandomMatchedPolicy, RandomMatchedState
from neurallm.control.specs import (
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    PolicySpec,
    RandomMatchedPolicySpec,
)
from neurallm.control.static import BestStaticPolicy, FixedPolicy
from neurallm.domain.models import (
    ControllerObservation,
    NonEmptyString,
    ProviderIdentity,
    RunManifest,
    SeedSchedule,
    Sha256Hex,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.experiments.plan import ExperimentPlan, PlannedTurn
from neurallm.experiments.protocol import MODEL_BACKED_POLICY_IDS, RunTier
from neurallm.metrics.base import MetricContext
from neurallm.metrics.deterministic import compute_response_metrics
from neurallm.providers.base import (
    GenerationProvider,
    GenerationRequest,
)
from neurallm.storage import (
    DurableExecutionAccounting,
    ResumeAction,
    SQLiteRunStore,
    StoreInvariantError,
    TurnInputEvidence,
    TurnState,
    scientific_result_sha256,
)


class AppliedPolicyTrace(PolicyTrace):
    """Policy decision plus every auditable action-clamping stage."""

    action_application: ActionApplication


class DetailedAppliedPolicyTrace(AppliedPolicyTrace):
    """Phase 3 action evidence retaining the policy-specific decision trace."""

    trace_schema_version: Literal["phase3-applied-policy-trace-v1"] = (
        "phase3-applied-policy-trace-v1"
    )
    history_access: Literal["none", "own_previous_response"]
    observation_has_previous_response: bool
    policy_trace: SerializeAsAny[PolicyTrace]

    @model_validator(mode="after")
    def validate_history_access(self) -> Self:
        expected = self.history_access == "own_previous_response" and self.turn_index > 0
        if self.observation_has_previous_response != expected:
            raise ValueError("policy observation history does not match declared access")
        return self


class CausalAppliedPolicyTrace(AppliedPolicyTrace):
    """Phase 4 neural trace bound to its exact committed observation source."""

    trace_schema_version: Literal["phase4-causal-applied-policy-trace-v1"] = (
        "phase4-causal-applied-policy-trace-v1"
    )
    history_access: Literal[
        "own_previous_response",
        "matched_focal_previous_response",
    ]
    observation_has_previous_response: bool
    history_source_policy_id: NonEmptyString | None
    history_source_condition_id: Sha256Hex | None
    history_commitment_sha256: Sha256Hex | None
    observation_metrics_sha256: Sha256Hex | None
    policy_trace: SerializeAsAny[NeuralPolicyTrace]

    @model_validator(mode="after")
    def validate_causal_history(self) -> Self:
        expected = self.turn_index > 0
        if self.observation_has_previous_response != expected:
            raise ValueError("neural observation history does not match the logical turn")
        evidence = (
            self.history_source_policy_id,
            self.history_source_condition_id,
            self.history_commitment_sha256,
            self.observation_metrics_sha256,
        )
        if expected and any(value is None for value in evidence):
            raise ValueError("neural history evidence must be complete exactly after turn zero")
        if not expected and any(value is not None for value in evidence):
            raise ValueError("turn-zero neural history evidence must be null")
        if expected:
            if self.history_access == "own_previous_response":
                if self.history_source_policy_id != self.policy_id:
                    raise ValueError("own-history trace names another source policy")
            elif self.history_source_policy_id == self.policy_id:
                raise ValueError("matched-focal trace must name another source policy")
        return self


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
    history_access: Literal[
        "none",
        "own_previous_response",
        "matched_focal_previous_response",
    ]
    history_source_policy_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Counts and identities from one execution or safe resume."""

    planned_turns: int
    previously_committed_turns: int
    dispatched_this_invocation: int
    successful_responses_this_invocation: int
    uncertain_dispatches_this_invocation: int
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
            history_access="none",
        )
        for policy_id in policy_ids
    }


def _runtime_from_policy_spec(spec: PolicySpec) -> PolicyRuntime:
    """Construct one typed policy runtime from its immutable specification."""

    if isinstance(spec, BestStaticPolicySpec):
        return PolicyRuntime(
            policy=BestStaticPolicy(spec),
            state_type=PolicyState,
            config_sha256=canonical_sha256(spec),
            history_access=spec.history_access,
        )
    if isinstance(spec, RandomMatchedPolicySpec):
        return PolicyRuntime(
            policy=cast(ControlPolicy, RandomMatchedPolicy(spec)),
            state_type=RandomMatchedState,
            config_sha256=canonical_sha256(spec),
            history_access=spec.history_access,
        )
    if isinstance(spec, HeuristicAdaptivePolicySpec):
        return PolicyRuntime(
            policy=cast(ControlPolicy, HeuristicAdaptivePolicy(spec)),
            state_type=HeuristicAdaptiveState,
            config_sha256=canonical_sha256(spec),
            history_access=spec.history_access,
        )
    if isinstance(spec, NeuralPersistentPolicySpec):
        return PolicyRuntime(
            policy=cast(ControlPolicy, NeuralPersistentPolicy(spec)),
            state_type=NeuralPolicyState,
            config_sha256=canonical_sha256(spec),
            history_access=spec.history_access,
        )
    if isinstance(spec, NeuralMatchedHistoryStateResetPolicySpec):
        return PolicyRuntime(
            policy=cast(ControlPolicy, NeuralMatchedHistoryStateResetPolicy(spec)),
            state_type=NeuralPolicyState,
            config_sha256=canonical_sha256(spec),
            history_access=spec.history_access,
            history_source_policy_id=spec.history_source_policy_id,
        )
    assert_never(spec)


def build_policy_runtimes(
    plan: ExperimentPlan,
    policy_specs: tuple[PolicySpec, ...],
) -> Mapping[str, PolicyRuntime]:
    """Build typed policy runtimes that exactly cover a Phase 3 plan."""

    planned_policy_ids = {turn.condition.policy_id for turn in plan.turns}
    configured_policy_ids = tuple(spec.policy_id for spec in policy_specs)
    if len(configured_policy_ids) != len(set(configured_policy_ids)):
        raise ValueError("policy specifications must not contain duplicate policy identifiers")
    if set(configured_policy_ids) != planned_policy_ids:
        raise ValueError("policy specifications do not exactly cover the experiment plan")
    neural_specs = tuple(
        spec
        for spec in policy_specs
        if isinstance(
            spec,
            (NeuralPersistentPolicySpec, NeuralMatchedHistoryStateResetPolicySpec),
        )
    )
    if plan.protocol is None and plan.evaluation is not None and neural_specs:
        raise ValueError("neural policies are not admitted to Phase 3 efficacy evaluation")
    reset_specs = tuple(
        spec for spec in policy_specs if isinstance(spec, NeuralMatchedHistoryStateResetPolicySpec)
    )
    if reset_specs and not any(
        isinstance(spec, NeuralPersistentPolicySpec) for spec in policy_specs
    ):
        raise ValueError("matched-history neural reset requires neural_persistent")
    if reset_specs:
        if plan.protocol is None and set(configured_policy_ids) != {
            "neural_persistent",
            "neural_matched_history_state_reset",
        }:
            raise ValueError("Phase 4 requires exactly the two neural attribution policies")
        if plan.protocol is not None and tuple(sorted(configured_policy_ids)) != (
            MODEL_BACKED_POLICY_IDS
        ):
            raise ValueError("model-backed execution requires the exact five policy runtimes")
    return {
        spec.policy_id: _runtime_from_policy_spec(spec)
        for spec in sorted(policy_specs, key=lambda item: item.policy_id)
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
    phase3_analysis_contract_sha256 = None
    confirmatory_analysis_contract_sha256 = None
    if plan.protocol is None and plan.evaluation is not None:
        from neurallm.experiments.analysis import build_phase3_analysis_contract_sha256

        phase3_analysis_contract_sha256 = build_phase3_analysis_contract_sha256(plan)
    if plan.protocol is not None and plan.protocol.run_tier is RunTier.CONFIRMATORY:
        if provider_identity.provider_type != "llama_cpp" or provider_identity.model_sha256 is None:
            raise ValueError(
                "confirmatory execution requires a digest-bound live llama_cpp provider"
            )
        from neurallm.experiments.scientific_analysis import (
            build_confirmatory_analysis_contract_sha256,
        )

        confirmatory_analysis_contract_sha256 = build_confirmatory_analysis_contract_sha256(plan)
    if (
        plan.protocol is not None
        and plan.protocol.run_tier is RunTier.CONFIRMATORY
        and not provenance.working_tree_clean
    ):
        raise ValueError("confirmatory execution requires a clean source worktree")
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
        matched_history_policy_sources={
            policy_id: runtime.history_source_policy_id
            for policy_id, runtime in sorted(policy_runtimes.items())
            if runtime.history_source_policy_id is not None
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
        evaluation_spec_json=(None if plan.evaluation is None else canonical_json(plan.evaluation)),
        evaluation_spec_sha256=plan.evaluation_spec_sha256,
        phase3_analysis_contract_sha256=phase3_analysis_contract_sha256,
        run_tier=(None if plan.protocol is None else plan.protocol.run_tier.value),
        scientific_identity_sha256=(
            None if plan.protocol is None else plan.scientific_identity_sha256
        ),
        preregistration_sha256=(
            None if plan.preregistration is None else plan.preregistration.seal_sha256
        ),
        confirmatory_analysis_contract_sha256=(confirmatory_analysis_contract_sha256),
    )


def _trajectory_key(
    turn: PlannedTurn,
    *,
    policy_id: str | None = None,
) -> tuple[object, ...]:
    condition = turn.condition
    return (
        condition.experiment_id,
        condition.dataset_version,
        condition.prompt_sequence_id,
        condition.policy_id if policy_id is None else policy_id,
        condition.model_seed,
        condition.controller_seed,
        condition.provider_identity_id,
        condition.base_decoding_profile_id,
    )


def _validate_causal_schedule(
    plan: ExperimentPlan,
    policy_runtimes: Mapping[str, PolicyRuntime],
) -> None:
    """Reject missing or forward causal edges before any provider dispatch."""

    for policy_id, matched_runtime in policy_runtimes.items():
        source_policy_id = matched_runtime.history_source_policy_id
        if source_policy_id is None:
            continue
        source_coordinates = {
            (_trajectory_key(turn), turn.condition.turn_index)
            for turn in plan.turns
            if turn.condition.policy_id == source_policy_id
        }
        matched_coordinates = {
            (
                _trajectory_key(turn, policy_id=source_policy_id),
                turn.condition.turn_index,
            )
            for turn in plan.turns
            if turn.condition.policy_id == policy_id
        }
        if source_coordinates != matched_coordinates:
            raise ValueError("matched-history attribution arms must have exact paired coverage")

    positions: dict[tuple[tuple[object, ...], int], int] = {}
    turns_by_coordinate: dict[tuple[tuple[object, ...], int], PlannedTurn] = {}
    for position, turn in enumerate(plan.turns):
        coordinate = (_trajectory_key(turn), turn.condition.turn_index)
        if coordinate in positions:
            raise ValueError("experiment plan repeats a policy trajectory turn")
        positions[coordinate] = position
        turns_by_coordinate[coordinate] = turn
    for position, turn in enumerate(plan.turns):
        runtime = policy_runtimes.get(turn.condition.policy_id)
        if runtime is None:
            raise ValueError(f"no runtime for policy {turn.condition.policy_id!r}")
        if runtime.history_source_policy_id is not None:
            paired_coordinate = (
                _trajectory_key(turn, policy_id=runtime.history_source_policy_id),
                turn.condition.turn_index,
            )
            paired_turn = turns_by_coordinate.get(paired_coordinate)
            paired_position = positions.get(paired_coordinate)
            if paired_turn is None or paired_position is None:
                raise ValueError("matched-history plan omits the paired focal current turn")
            if paired_position >= position:
                raise ValueError(
                    "paired focal current turn must be scheduled before its reset pair"
                )
            current_inputs = (
                turn.prompt_case_id,
                turn.prompt_family,
                turn.prompt_features,
                turn.prompt,
                turn.validator,
                turn.decoding_parameters,
            )
            paired_inputs = (
                paired_turn.prompt_case_id,
                paired_turn.prompt_family,
                paired_turn.prompt_features,
                paired_turn.prompt,
                paired_turn.validator,
                paired_turn.decoding_parameters,
            )
            if current_inputs != paired_inputs:
                raise ValueError(
                    "matched-history attribution pairs must share exact current inputs"
                )
        if turn.condition.turn_index == 0:
            continue
        source_policy_id = runtime.history_source_policy_id or turn.condition.policy_id
        predecessor = (
            _trajectory_key(turn, policy_id=source_policy_id),
            turn.condition.turn_index - 1,
        )
        predecessor_position = positions.get(predecessor)
        if predecessor_position is None:
            raise ValueError("experiment plan omits a declared causal predecessor")
        if predecessor_position >= position:
            raise ValueError("causal predecessor must be scheduled before its dependent turn")


def _policy_decision(
    store: SQLiteRunStore,
    turn: PlannedTurn,
    previous_condition_id: str | None,
    runtime: PolicyRuntime,
    plan: ExperimentPlan,
) -> tuple[
    GenerationRequest,
    PolicyState,
    AppliedPolicyTrace | DetailedAppliedPolicyTrace | CausalAppliedPolicyTrace,
]:
    committed = None
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
        previous_metrics = committed.metrics if runtime.history_access != "none" else None
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
    trace: AppliedPolicyTrace | DetailedAppliedPolicyTrace | CausalAppliedPolicyTrace
    if isinstance(runtime.policy, SimulatedNeuralPolicy):
        if runtime.history_access not in {
            "own_previous_response",
            "matched_focal_previous_response",
        }:
            raise ValueError("simulated neural policy requires response-history access")
        neural_history_access = cast(
            Literal[
                "own_previous_response",
                "matched_focal_previous_response",
            ],
            runtime.history_access,
        )
        trace = CausalAppliedPolicyTrace(
            policy_id=proposed_trace.policy_id,
            turn_index=proposed_trace.turn_index,
            action=application.step_clamped_action,
            action_application=application,
            history_access=neural_history_access,
            observation_has_previous_response=previous_metrics is not None,
            history_source_policy_id=(None if committed is None else committed.condition.policy_id),
            history_source_condition_id=(None if committed is None else committed.condition_id),
            history_commitment_sha256=(
                None if committed is None else committed.history_commitment_sha256
            ),
            observation_metrics_sha256=(
                None if committed is None else canonical_sha256(committed.metrics)
            ),
            policy_trace=proposed_trace,
        )
    elif plan.database_schema_version >= 2:
        if runtime.history_access == "matched_focal_previous_response":
            raise ValueError("only simulated neural policies may use matched focal history")
        detailed_history_access = runtime.history_access
        trace = DetailedAppliedPolicyTrace(
            policy_id=proposed_trace.policy_id,
            turn_index=proposed_trace.turn_index,
            action=application.step_clamped_action,
            action_application=application,
            history_access=detailed_history_access,
            observation_has_previous_response=previous_metrics is not None,
            policy_trace=proposed_trace,
        )
    else:
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
    _validate_causal_schedule(plan, policy_runtimes)

    provider_calls = 0
    dispatched_this_invocation = 0
    uncertain_dispatches_this_invocation = 0
    planned_ids = {turn.condition.condition_id for turn in plan.turns}
    committed_by_coordinate: dict[tuple[tuple[object, ...], int], str] = {}
    with SQLiteRunStore(database_path, manifest) as store:
        store.verify_integrity()
        previously_committed_turns = len(store.list_turns(TurnState.COMMITTED))
        unknown_ids = {turn.condition_id for turn in store.list_turns()} - planned_ids
        if unknown_ids:
            raise StoreInvariantError(
                f"run store contains conditions outside the current plan: {sorted(unknown_ids)!r}"
            )

        for turn in plan.turns:
            condition_id = turn.condition.condition_id
            runtime = policy_runtimes.get(turn.condition.policy_id)
            if runtime is None:
                raise ValueError(f"no runtime for policy {turn.condition.policy_id!r}")
            history_source_policy_id = runtime.history_source_policy_id or turn.condition.policy_id
            history_trajectory = _trajectory_key(
                turn,
                policy_id=history_source_policy_id,
            )
            previous_condition_id = (
                None
                if turn.condition.turn_index == 0
                else committed_by_coordinate.get(
                    (history_trajectory, turn.condition.turn_index - 1)
                )
            )
            expected_previous_index = turn.condition.turn_index - 1
            if (previous_condition_id is None) != (turn.condition.turn_index == 0):
                raise StoreInvariantError(
                    f"plan trajectory is missing turn {expected_previous_index} before "
                    f"{turn.condition.turn_index}"
                )
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
            input_evidence = (
                TurnInputEvidence(
                    condition_id=condition_id,
                    prompt_case_id=turn.prompt_case_id,
                    prompt_family=turn.prompt_family,
                    prompt_features=turn.prompt_features,
                    validator=turn.validator,
                )
                if manifest.database_schema_version >= 2
                else None
            )
            stored = store.prepare_turn(request, history, input_evidence)
            action = store.resume_action(condition_id)

            if action is ResumeAction.DISPATCH_PREPARED:
                store.begin_dispatch(condition_id)
                dispatched_this_invocation += 1
                if checkpoint_hook is not None:
                    checkpoint_hook(TurnState.DISPATCHING, turn)
                try:
                    response = provider.generate(request)
                    provider_calls += 1
                except Exception as exc:
                    uncertain_dispatches_this_invocation += 1
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

            coordinate = (_trajectory_key(turn), turn.condition.turn_index)
            if coordinate in committed_by_coordinate:
                raise StoreInvariantError("plan repeats a policy trajectory turn")
            committed_by_coordinate[coordinate] = condition_id

        store.verify_integrity()
        stored_turns = store.list_turns()
        if len(stored_turns) != len(plan.turns) or any(
            turn.state is not TurnState.COMMITTED for turn in stored_turns
        ):
            raise StoreInvariantError("execution did not commit the complete planned schedule")
        durable_accounting = (
            None
            if plan.protocol is None
            else DurableExecutionAccounting(
                planned_logical_generations=len(plan.turns),
                dispatched_logical_generations=len(plan.turns),
                successful_responses=len(plan.turns),
                uncertain_dispatches=0,
                committed_logical_generations=len(plan.turns),
            )
        )
        finalization = store.finalize_run(
            tuple(planned_ids),
            scientific_result_sha256(stored_turns),
            durable_accounting,
        )
        store.verify_integrity()
        store.compact()

    return ExecutionSummary(
        planned_turns=len(plan.turns),
        previously_committed_turns=previously_committed_turns,
        dispatched_this_invocation=dispatched_this_invocation,
        successful_responses_this_invocation=provider_calls,
        uncertain_dispatches_this_invocation=uncertain_dispatches_this_invocation,
        committed_turns=len(plan.turns),
        provider_calls=provider_calls,
        manifest_sha256=finalization.manifest_sha256,
    )


__all__ = [
    "AppliedPolicyTrace",
    "CausalAppliedPolicyTrace",
    "CheckpointHook",
    "DetailedAppliedPolicyTrace",
    "ExecutionSummary",
    "GitProvenance",
    "PolicyRuntime",
    "build_fixed_policy_runtimes",
    "build_policy_runtimes",
    "build_run_manifest",
    "execute_plan",
    "read_git_provenance",
]

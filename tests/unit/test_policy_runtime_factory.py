"""Construction-time policy runtime dispatch contracts."""

from pathlib import Path

import pytest

from neurallm.control import (
    BestStaticPolicy,
    BestStaticPolicySpec,
    HeuristicAdaptivePolicy,
    HeuristicAdaptivePolicySpec,
    HeuristicAdaptiveState,
    PolicyContext,
    PolicyState,
    RandomMatchedPolicy,
    RandomMatchedPolicySpec,
    RandomMatchedState,
)
from neurallm.control.static import FixedPolicy
from neurallm.domain.serialization import canonical_sha256
from neurallm.experiments.config import ExperimentConfig, LoadedExperimentConfig
from neurallm.experiments.dataset import LoadedDataset, PromptDataset
from neurallm.experiments.plan import ExperimentPlan, build_plan
from neurallm.experiments.runner import (
    build_fixed_policy_runtimes,
    build_policy_runtimes,
)
from tests.unit.test_experiment_plan import config_payload, dataset_payload


def _plan_with_policy_ids(policy_ids: list[str]) -> ExperimentPlan:
    payload = config_payload()
    payload["policy_ids"] = policy_ids
    config = ExperimentConfig.model_validate(payload)
    dataset = PromptDataset.model_validate(dataset_payload())
    return build_plan(
        LoadedExperimentConfig(
            config=config,
            source_path=Path("config.yaml"),
            dataset_path=Path("dataset.yaml"),
            provider_config_path=None,
            artifact_root=Path("run"),
        ),
        LoadedDataset(dataset=dataset, source_path=Path("dataset.yaml")),
    )


def test_typed_runtime_factory_dispatches_every_spec_before_execution() -> None:
    plan = _plan_with_policy_ids(["random_matched", "best_static", "heuristic_adaptive"])
    specs = (
        RandomMatchedPolicySpec(),
        BestStaticPolicySpec(),
        HeuristicAdaptivePolicySpec(),
    )

    runtimes = build_policy_runtimes(plan, specs)

    assert tuple(runtimes) == (
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    )
    assert isinstance(runtimes["best_static"].policy, BestStaticPolicy)
    assert isinstance(runtimes["heuristic_adaptive"].policy, HeuristicAdaptivePolicy)
    assert isinstance(runtimes["random_matched"].policy, RandomMatchedPolicy)
    assert runtimes["best_static"].state_type is PolicyState
    assert runtimes["heuristic_adaptive"].state_type is HeuristicAdaptiveState
    assert runtimes["random_matched"].state_type is RandomMatchedState
    assert {policy_id: runtime.history_access for policy_id, runtime in runtimes.items()} == {
        "best_static": "none",
        "heuristic_adaptive": "own_previous_response",
        "random_matched": "none",
    }
    assert {spec.policy_id: runtimes[spec.policy_id].config_sha256 for spec in specs} == {
        spec.policy_id: canonical_sha256(spec) for spec in specs
    }

    for policy_id, runtime in runtimes.items():
        first_turn = next(
            turn
            for turn in plan.turns
            if turn.condition.policy_id == policy_id and turn.condition.turn_index == 0
        )
        state = runtime.policy.initial_state(
            PolicyContext(
                condition=first_turn.condition,
                initial_decoding_parameters=first_turn.decoding_parameters,
                action_bounds=plan.action_bounds,
            )
        )
        assert type(state) is runtime.state_type


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (
            (BestStaticPolicySpec(), RandomMatchedPolicySpec()),
            "exactly cover",
        ),
        (
            (
                BestStaticPolicySpec(),
                BestStaticPolicySpec(),
                RandomMatchedPolicySpec(),
                HeuristicAdaptivePolicySpec(),
            ),
            "duplicate policy identifiers",
        ),
    ],
)
def test_typed_runtime_factory_rejects_incomplete_or_duplicate_specs(
    specs: tuple[
        BestStaticPolicySpec | RandomMatchedPolicySpec | HeuristicAdaptivePolicySpec,
        ...,
    ],
    message: str,
) -> None:
    plan = _plan_with_policy_ids(["best_static", "heuristic_adaptive", "random_matched"])

    with pytest.raises(ValueError, match=message):
        build_policy_runtimes(plan, specs)


def test_phase2_fixed_runtime_identity_remains_unchanged() -> None:
    plan = _plan_with_policy_ids(["kernel_fixed"])

    runtimes = build_fixed_policy_runtimes(plan)

    runtime = runtimes["kernel_fixed"]
    assert isinstance(runtime.policy, FixedPolicy)
    assert runtime.state_type is PolicyState
    assert runtime.history_access == "none"
    assert runtime.config_sha256 == canonical_sha256(
        {
            "policy_type": "fixed",
            "implementation_version": "phase2-fixed-v1",
            "policy_id": "kernel_fixed",
        }
    )

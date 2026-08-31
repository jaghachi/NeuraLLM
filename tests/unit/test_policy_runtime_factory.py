"""Construction-time policy runtime dispatch contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurallm.control import (
    BestStaticPolicy,
    BestStaticPolicySpec,
    HeuristicAdaptivePolicy,
    HeuristicAdaptivePolicySpec,
    HeuristicAdaptiveState,
    NeuralMatchedHistoryStateResetPolicy,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicy,
    NeuralPersistentPolicySpec,
    NeuralPolicyState,
    PolicyContext,
    PolicyState,
    RandomMatchedPolicy,
    RandomMatchedPolicySpec,
    RandomMatchedState,
)
from neurallm.control.static import FixedPolicy
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation import DatasetPurpose, EvaluationSpec
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


def _phase4_dataset() -> PromptDataset:
    payload = dataset_payload()
    payload["purpose"] = DatasetPurpose.DEVELOPMENT.value
    return PromptDataset.model_validate(payload)


def _phase4_config_payload(dataset: PromptDataset) -> dict[str, object]:
    payload = config_payload()
    payload.pop("policy_ids")
    payload["dataset"] = {
        "path": "dataset.yaml",
        "version": dataset.version,
        "purpose": DatasetPurpose.DEVELOPMENT.value,
        "expected_dataset_sha256": dataset.dataset_hash,
    }
    payload["policy_specs"] = [
        {"kind": "neural_persistent"},
        {"kind": "neural_matched_history_state_reset"},
    ]
    payload["decision_rule_version"] = "phase4-neural-mechanism-only-v1"
    payload["database_schema_version"] = 2
    return payload


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


def test_neural_runtime_factory_binds_focal_history_and_shared_state_type() -> None:
    plan = _plan_with_policy_ids(["neural_persistent", "neural_matched_history_state_reset"])
    persistent_spec = NeuralPersistentPolicySpec()
    reset_spec = NeuralMatchedHistoryStateResetPolicySpec()

    runtimes = build_policy_runtimes(plan, (persistent_spec, reset_spec))

    persistent = runtimes[persistent_spec.policy_id]
    reset = runtimes[reset_spec.policy_id]
    assert isinstance(persistent.policy, NeuralPersistentPolicy)
    assert isinstance(reset.policy, NeuralMatchedHistoryStateResetPolicy)
    assert persistent.state_type is reset.state_type is NeuralPolicyState
    assert persistent.history_access == "own_previous_response"
    assert persistent.history_source_policy_id is None
    assert reset.history_access == "matched_focal_previous_response"
    assert reset.history_source_policy_id == persistent_spec.policy_id
    assert persistent.config_sha256 == canonical_sha256(persistent_spec)
    assert reset.config_sha256 == canonical_sha256(reset_spec)


def test_neural_reset_runtime_requires_its_declared_focal_policy() -> None:
    plan = _plan_with_policy_ids(["neural_matched_history_state_reset"])

    with pytest.raises(ValueError, match="requires neural_persistent"):
        build_policy_runtimes(
            plan,
            (NeuralMatchedHistoryStateResetPolicySpec(),),
        )


@pytest.mark.parametrize(
    ("specs", "evaluation"),
    [
        (
            (BestStaticPolicySpec(), NeuralPersistentPolicySpec()),
            EvaluationSpec(
                focal_policy_id="neural_persistent",
                required_serious_comparator_ids=("best_static",),
                bootstrap_seed=101,
                permutation_seed=202,
            ),
        ),
        (
            (
                BestStaticPolicySpec(),
                NeuralPersistentPolicySpec(),
                NeuralMatchedHistoryStateResetPolicySpec(),
            ),
            EvaluationSpec(
                focal_policy_id="neural_matched_history_state_reset",
                required_serious_comparator_ids=("best_static",),
                negative_control_policy_ids=("neural_persistent",),
                bootstrap_seed=101,
                permutation_seed=202,
            ),
        ),
    ],
    ids=("persistent", "matched-history-reset"),
)
def test_runtime_factory_rejects_neural_policies_with_an_evaluation_spec(
    specs: tuple[
        BestStaticPolicySpec
        | NeuralPersistentPolicySpec
        | NeuralMatchedHistoryStateResetPolicySpec,
        ...,
    ],
    evaluation: EvaluationSpec,
) -> None:
    plan = _plan_with_policy_ids([spec.policy_id for spec in specs])
    phase3_shaped_plan = plan.model_copy(update={"evaluation": evaluation})

    with pytest.raises(ValueError, match="not admitted to Phase 3 efficacy evaluation"):
        build_policy_runtimes(phase3_shaped_plan, specs)


def test_configuration_rejects_matched_reset_without_focal_policy() -> None:
    payload = config_payload()
    payload.pop("policy_ids")
    payload["policy_specs"] = [{"kind": "neural_matched_history_state_reset"}]

    with pytest.raises(ValidationError, match="declared focal source policy"):
        ExperimentConfig.model_validate(payload)


def test_phase4_plan_rejects_schema_v1_before_schedule_materialization() -> None:
    dataset = _phase4_dataset()
    payload = _phase4_config_payload(dataset)
    payload["database_schema_version"] = 1
    config = ExperimentConfig.model_validate(payload)

    with pytest.raises(ValueError, match="Phase 4 requires the current database schema"):
        build_plan(
            LoadedExperimentConfig(
                config=config,
                source_path=Path("config.yaml"),
                dataset_path=Path("dataset.yaml"),
                provider_config_path=None,
                artifact_root=Path("run"),
            ),
            LoadedDataset(dataset=dataset, source_path=Path("dataset.yaml")),
        )


def test_phase4_config_and_plan_reject_an_extra_best_static_arm() -> None:
    dataset = _phase4_dataset()
    payload = _phase4_config_payload(dataset)
    policy_specs = list(payload["policy_specs"])  # type: ignore[arg-type]
    policy_specs.append({"kind": "best_static"})
    payload["policy_specs"] = policy_specs

    with pytest.raises(ValidationError, match="exactly the two neural attribution policies"):
        ExperimentConfig.model_validate(payload)

    valid_config = ExperimentConfig.model_validate(_phase4_config_payload(dataset))
    assert valid_config.policy_specs is not None
    bypassed_config = valid_config.model_copy(
        update={"policy_specs": (*valid_config.policy_specs, BestStaticPolicySpec())}
    )
    with pytest.raises(ValidationError, match="exactly the two neural attribution policies"):
        build_plan(
            LoadedExperimentConfig(
                config=bypassed_config,
                source_path=Path("config.yaml"),
                dataset_path=Path("dataset.yaml"),
                provider_config_path=None,
                artifact_root=Path("run"),
            ),
            LoadedDataset(dataset=dataset, source_path=Path("dataset.yaml")),
        )


def test_phase4_rejects_an_unpinned_dataset_while_legacy_phase2_is_unchanged() -> None:
    phase4_dataset = _phase4_dataset()
    unpinned_phase4 = _phase4_config_payload(phase4_dataset)
    unpinned_phase4["dataset"] = {
        "path": "dataset.yaml",
        "version": phase4_dataset.version,
    }

    with pytest.raises(ValidationError, match="pinned development-purpose dataset"):
        ExperimentConfig.model_validate(unpinned_phase4)

    legacy_config = ExperimentConfig.model_validate(config_payload())
    legacy_dataset = PromptDataset.model_validate(dataset_payload())
    legacy_plan = build_plan(
        LoadedExperimentConfig(
            config=legacy_config,
            source_path=Path("config.yaml"),
            dataset_path=Path("dataset.yaml"),
            provider_config_path=None,
            artifact_root=Path("run"),
        ),
        LoadedDataset(dataset=legacy_dataset, source_path=Path("dataset.yaml")),
    )

    assert legacy_config.dataset.purpose is None
    assert legacy_plan.dataset_purpose is None
    assert legacy_plan.decision_rule_version == "phase2-no-scientific-decision-v1"


def test_config_rejects_a_base_decoding_profile_outside_legal_bounds() -> None:
    payload = config_payload()
    payload["decoding_bounds"] = {
        "temperature": [0.01, 2.0],
        "top_p": [0.01, 0.8],
        "top_k": [0, 200],
        "presence_penalty": [-2.0, 2.0],
    }

    with pytest.raises(ValidationError, match="base decoding profile exceeds"):
        ExperimentConfig.model_validate(payload)

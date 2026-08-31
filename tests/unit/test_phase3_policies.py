"""Determinism, bounds, and behavior tests for Phase 3 control policies."""

from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from neurallm.control import (
    BestStaticPolicy,
    ControlPolicy,
    FixedPolicy,
    HeuristicAdaptivePolicy,
    HeuristicAdaptivePolicySpec,
    HeuristicAdaptiveState,
    PolicyContext,
    PolicyState,
    RandomMatchedPolicy,
    RandomMatchedState,
)
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    CountMetricValue,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
    ResponseMetrics,
    UnitIntervalMetricValue,
)

_INPUT_HASH = "0" * 64
_PROVIDER_ID = "1" * 64
_ZERO_ACTION = ControllerAction(
    temperature_delta=0.0,
    top_p_delta=0.0,
    top_k_delta=0,
    presence_penalty_delta=0.0,
)


def _context(
    policy_id: str,
    *,
    controller_seed: int = 23,
    model_seed: int = 17,
    prompt_sequence_id: str = "sequence-a",
    turn_index: int = 0,
    action_bounds: ActionBounds | None = None,
) -> PolicyContext:
    return PolicyContext(
        condition=ExperimentCondition(
            experiment_id="phase3-policy-tests",
            dataset_version="synthetic-v1",
            prompt_sequence_id=prompt_sequence_id,
            turn_index=turn_index,
            policy_id=policy_id,
            model_seed=model_seed,
            controller_seed=controller_seed,
            provider_identity_id=_PROVIDER_ID,
            base_decoding_profile_id="base-v1",
        ),
        initial_decoding_parameters=DecodingParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=128,
            seed=model_seed,
        ),
        action_bounds=action_bounds or ActionBounds(),
    )


def _unit_metric(value: float | None) -> UnitIntervalMetricValue:
    return UnitIntervalMetricValue(
        value=value,
        availability=value is not None,
        metric_version="policy-test-v1",
        input_hash=_INPUT_HASH,
    )


def _count_metric(value: int | None) -> CountMetricValue:
    return CountMetricValue(
        value=value,
        availability=value is not None,
        metric_version="policy-test-v1",
        input_hash=_INPUT_HASH,
    )


def _metrics(
    *,
    repetition: float | None = 0.0,
    adherence: float | None = 1.0,
    response_length: int | None = 64,
) -> ResponseMetrics:
    return ResponseMetrics(
        task_score=_unit_metric(1.0),
        instruction_adherence=_unit_metric(adherence),
        response_length_tokens=_count_metric(response_length),
        repetition_ratio=_unit_metric(repetition),
        repeated_3_gram_ratio=_unit_metric(0.0),
        repeated_4_gram_ratio=_unit_metric(0.0),
        distinct_2=_unit_metric(1.0),
        distinct_3=_unit_metric(1.0),
        late_window_repetition_ratio=_unit_metric(0.0),
        format_validity=_unit_metric(1.0),
        semantic_similarity=_unit_metric(None),
    )


def _observation(
    turn_index: int,
    metrics: ResponseMetrics | None = None,
) -> ControllerObservation:
    return ControllerObservation(
        turn_index=turn_index,
        prompt_family="synthetic",
        current_prompt_features=PromptFeatures({"difficulty": 0.5}),
        previous_response_metrics=metrics,
        has_previous_response=metrics is not None,
    )


def _advanced_random_state(
    policy: RandomMatchedPolicy,
    context: PolicyContext,
) -> RandomMatchedState:
    initial = policy.initial_state(context)
    action, state, trace = policy.act(_observation(0), initial)
    assert action == _ZERO_ACTION
    assert trace.action == _ZERO_ACTION
    return state


def _advanced_heuristic_state(
    policy: HeuristicAdaptivePolicy,
    context: PolicyContext,
) -> HeuristicAdaptiveState:
    initial = policy.initial_state(context)
    action, state, trace = policy.act(_observation(0), initial)
    assert action == _ZERO_ACTION
    assert trace.action == _ZERO_ACTION
    return state


def test_best_static_uses_the_shared_interface_and_preserves_fixed_policy() -> None:
    policy = BestStaticPolicy()
    fixed = FixedPolicy()

    assert isinstance(policy, ControlPolicy)
    assert isinstance(fixed, ControlPolicy)
    state = policy.initial_state(_context(policy.policy_id))
    action, next_state, trace = policy.act(_observation(0), state)

    assert action == _ZERO_ACTION
    assert next_state == PolicyState()
    assert trace.policy_id == "best_static"
    assert set(action.model_dump()) == {
        "temperature_delta",
        "top_p_delta",
        "top_k_delta",
        "presence_penalty_delta",
    }


def test_all_phase3_policies_return_exact_zero_action_at_turn_zero() -> None:
    best_static = BestStaticPolicy()
    random_policy = RandomMatchedPolicy()
    heuristic = HeuristicAdaptivePolicy()

    static_action, _, _ = best_static.act(
        _observation(0),
        best_static.initial_state(_context(best_static.policy_id)),
    )
    random_action, _, _ = random_policy.act(
        _observation(0),
        random_policy.initial_state(_context(random_policy.policy_id)),
    )
    heuristic_action, _, heuristic_trace = heuristic.act(
        _observation(0),
        heuristic.initial_state(_context(heuristic.policy_id)),
    )

    assert static_action == random_action == heuristic_action == _ZERO_ACTION
    assert heuristic_trace.repetition_excess == 0.0
    assert heuristic_trace.adherence_deficit == 0.0
    assert heuristic_trace.length_deviation == 0.0
    assert heuristic_trace.clean_decay_applied is False


def test_random_replay_ignores_global_rng_and_is_bound_to_condition_identity() -> None:
    first_policy = RandomMatchedPolicy()
    context = _context(first_policy.policy_id, controller_seed=123)
    first_state = _advanced_random_state(first_policy, context)

    random.seed(1)
    random.random()
    first_action, _, first_trace = first_policy.act(
        _observation(1, _metrics(repetition=1.0, adherence=0.0)),
        first_state,
    )

    random.seed(999_999)
    for _ in range(20):
        random.random()
    replay_policy = RandomMatchedPolicy()
    replay_state = _advanced_random_state(replay_policy, context)
    replay_action, _, replay_trace = replay_policy.act(
        _observation(1, _metrics(repetition=0.0, adherence=1.0)),
        replay_state,
    )

    assert first_action == replay_action
    assert first_trace.draw_sha256 == replay_trace.draw_sha256
    assert context.action_bounds.contains(first_action)
    assert first_state.initial_condition_id == context.condition.condition_id


def test_random_changed_controller_seed_changes_the_draw() -> None:
    policy = RandomMatchedPolicy()
    first = _advanced_random_state(policy, _context(policy.policy_id, controller_seed=1))
    second = _advanced_random_state(policy, _context(policy.policy_id, controller_seed=2))

    first_action, _, first_trace = policy.act(_observation(1, _metrics()), first)
    second_action, _, second_trace = policy.act(_observation(1, _metrics()), second)

    assert first_action != second_action
    assert first_trace.draw_sha256 != second_trace.draw_sha256


@given(controller_seed=st.integers(min_value=-(2**63), max_value=2**63 - 1))
def test_random_actions_are_deterministic_and_bounded_for_int64_seeds(
    controller_seed: int,
) -> None:
    bounds = ActionBounds(
        temperature_delta=(-0.03, 0.07),
        top_p_delta=(-0.02, 0.01),
        top_k_delta=(-3, 8),
        presence_penalty_delta=(-0.12, 0.04),
    )
    policy = RandomMatchedPolicy()
    context = _context(
        policy.policy_id,
        controller_seed=controller_seed,
        action_bounds=bounds,
    )
    state = _advanced_random_state(policy, context)

    first, _, _ = policy.act(_observation(1, _metrics()), state)
    second, _, _ = policy.act(_observation(1, _metrics(repetition=1.0)), state)

    assert first == second
    assert bounds.contains(first)


def test_heuristic_stress_response_is_materially_non_static_on_all_dimensions() -> None:
    policy = HeuristicAdaptivePolicy()
    state = _advanced_heuristic_state(policy, _context(policy.policy_id))

    action, next_state, trace = policy.act(
        _observation(1, _metrics(repetition=1.0, adherence=1.0, response_length=64)),
        state,
    )

    assert action.temperature_delta == pytest.approx(0.075)
    assert action.top_p_delta == pytest.approx(0.0375)
    assert action.top_k_delta == 8
    assert action.presence_penalty_delta == pytest.approx(0.15)
    assert all(value != 0 for value in action.model_dump().values())
    assert state.action_bounds.contains(action)
    assert next_state.last_action == action
    assert trace.repetition_excess == 1.0
    assert trace.combined_drive == pytest.approx(0.75)
    assert trace.clean_decay_applied is False


@pytest.mark.parametrize(
    ("metrics", "expected_drive"),
    [
        (_metrics(repetition=0.0, adherence=0.0, response_length=64), -0.50),
        (_metrics(repetition=0.0, adherence=1.0, response_length=0), 0.25),
        (_metrics(repetition=0.0, adherence=1.0, response_length=512), -0.25),
    ],
)
def test_heuristic_transparently_reacts_to_adherence_and_length(
    metrics: ResponseMetrics,
    expected_drive: float,
) -> None:
    policy = HeuristicAdaptivePolicy()
    state = _advanced_heuristic_state(policy, _context(policy.policy_id))

    action, _, trace = policy.act(_observation(1, metrics), state)

    assert trace.combined_drive == pytest.approx(expected_drive)
    assert action.temperature_delta == pytest.approx(
        expected_drive * (0.1 if expected_drive >= 0 else 0.1)
    )
    assert state.action_bounds.contains(action)


def test_heuristic_clean_feedback_decays_the_prior_action_toward_zero() -> None:
    policy = HeuristicAdaptivePolicy()
    initial = _advanced_heuristic_state(policy, _context(policy.policy_id))
    stressed_action, stressed_state, _ = policy.act(
        _observation(1, _metrics(repetition=1.0)),
        initial,
    )

    clean_action, clean_state, clean_trace = policy.act(
        _observation(2, _metrics()),
        stressed_state,
    )
    cleaner_action, _, cleaner_trace = policy.act(
        _observation(3, _metrics()),
        clean_state,
    )

    assert clean_action.temperature_delta == pytest.approx(stressed_action.temperature_delta * 0.5)
    assert clean_action.top_p_delta == pytest.approx(stressed_action.top_p_delta * 0.5)
    assert clean_action.top_k_delta == 4
    assert clean_action.presence_penalty_delta == pytest.approx(
        stressed_action.presence_penalty_delta * 0.5
    )
    assert abs(cleaner_action.temperature_delta) < abs(clean_action.temperature_delta)
    assert abs(cleaner_action.top_k_delta) < abs(clean_action.top_k_delta)
    assert clean_trace.clean_decay_applied is True
    assert cleaner_trace.clean_decay_applied is True


@given(
    repetition=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    adherence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    response_length=st.integers(min_value=0, max_value=2_000),
)
def test_heuristic_actions_remain_bounded_for_finite_metric_inputs(
    repetition: float,
    adherence: float,
    response_length: int,
) -> None:
    bounds = ActionBounds(
        temperature_delta=(-0.04, 0.08),
        top_p_delta=(-0.03, 0.02),
        top_k_delta=(-4, 7),
        presence_penalty_delta=(-0.1, 0.15),
    )
    policy = HeuristicAdaptivePolicy()
    state = _advanced_heuristic_state(
        policy,
        _context(policy.policy_id, action_bounds=bounds),
    )

    action, _, _ = policy.act(
        _observation(
            1,
            _metrics(
                repetition=repetition,
                adherence=adherence,
                response_length=response_length,
            ),
        ),
        state,
    )

    assert bounds.contains(action)


@pytest.mark.parametrize(
    "metrics",
    [
        _metrics(repetition=None),
        _metrics(adherence=None),
        _metrics(response_length=None),
    ],
)
def test_heuristic_requires_each_feedback_metric(metrics: ResponseMetrics) -> None:
    policy = HeuristicAdaptivePolicy()
    state = _advanced_heuristic_state(policy, _context(policy.policy_id))

    with pytest.raises(ValueError, match="must be available"):
        policy.act(_observation(1, metrics), state)


def test_heuristic_rejects_missing_history_after_turn_zero() -> None:
    policy = HeuristicAdaptivePolicy()
    state = _advanced_heuristic_state(policy, _context(policy.policy_id))

    with pytest.raises(ValueError, match="requires previous-response metrics"):
        policy.act(_observation(1), state)


def test_policy_state_and_constructor_guards_fail_closed() -> None:
    random_policy = RandomMatchedPolicy()
    heuristic = HeuristicAdaptivePolicy()

    with pytest.raises(TypeError, match="RandomMatchedPolicySpec"):
        RandomMatchedPolicy(HeuristicAdaptivePolicySpec())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HeuristicAdaptivePolicySpec"):
        HeuristicAdaptivePolicy(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BestStaticPolicySpec"):
        BestStaticPolicy(object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="another policy"):
        random_policy.initial_state(_context("best_static"))
    with pytest.raises(ValueError, match="another policy"):
        heuristic.initial_state(_context("best_static"))
    with pytest.raises(ValueError, match="turn zero"):
        random_policy.initial_state(_context(random_policy.policy_id, turn_index=1))
    with pytest.raises(ValueError, match="turn zero"):
        heuristic.initial_state(_context(heuristic.policy_id, turn_index=1))

    random_state = _advanced_random_state(random_policy, _context(random_policy.policy_id))
    heuristic_state = _advanced_heuristic_state(heuristic, _context(heuristic.policy_id))
    with pytest.raises(ValueError, match="turn"):
        random_policy.act(_observation(2, _metrics()), random_state)
    with pytest.raises(ValueError, match="turn"):
        heuristic.act(_observation(2, _metrics()), heuristic_state)
    with pytest.raises(TypeError, match="RandomMatchedState"):
        random_policy.act(_observation(1, _metrics()), PolicyState())
    with pytest.raises(TypeError, match="HeuristicAdaptiveState"):
        heuristic.act(_observation(1, _metrics()), PolicyState())

    with pytest.raises(ValidationError, match="frozen"):
        random_state.next_turn_index = 99
    with pytest.raises(ValidationError, match="frozen"):
        heuristic_state.last_action = _ZERO_ACTION

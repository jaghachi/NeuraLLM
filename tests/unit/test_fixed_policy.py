"""Tests for the Phase 2 fixed kernel policy."""

import pytest

from neurallm.control import FixedPolicy, PolicyContext, PolicyState
from neurallm.domain.models import (
    ActionBounds,
    ControllerObservation,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
)
from neurallm.providers.fake import FakeProvider


def make_condition(policy_id: str = "kernel_fixed") -> ExperimentCondition:
    return ExperimentCondition(
        experiment_id="phase2",
        dataset_version="v1",
        prompt_sequence_id="sequence-1",
        turn_index=0,
        policy_id=policy_id,
        model_seed=1,
        controller_seed=2,
        provider_identity_id=FakeProvider().provider_identity.identity_id,
        base_decoding_profile_id="base-v1",
    )


def test_fixed_policy_returns_a_zero_action_through_shared_interface() -> None:
    policy = FixedPolicy()
    context = PolicyContext(
        condition=make_condition(),
        initial_decoding_parameters=DecodingParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=64,
            seed=1,
        ),
        action_bounds=ActionBounds(),
    )
    state = policy.initial_state(context)
    observation = ControllerObservation(
        turn_index=0,
        prompt_family="constrained",
        current_prompt_features=PromptFeatures({}),
        previous_response_metrics=None,
        has_previous_response=False,
    )

    action, next_state, trace = policy.act(observation, state)

    assert ActionBounds().contains(action)
    assert action.model_dump() == {
        "temperature_delta": 0.0,
        "top_p_delta": 0.0,
        "top_k_delta": 0,
        "presence_penalty_delta": 0.0,
    }
    assert next_state == PolicyState()
    assert trace.policy_id == policy.policy_id


def test_fixed_policy_rejects_mismatched_context() -> None:
    policy = FixedPolicy("kernel_fixed")
    context = PolicyContext(
        condition=make_condition("another-policy"),
        initial_decoding_parameters=DecodingParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=64,
            seed=1,
        ),
        action_bounds=ActionBounds(),
    )

    with pytest.raises(ValueError, match="another policy"):
        policy.initial_state(context)

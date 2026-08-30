"""Contract tests for the one shared controller policy interface."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neurallm.control.policy import (
    ControlPolicy,
    PolicyContext,
    PolicyState,
    PolicyTrace,
)
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
)
from neurallm.providers.fake import FakeProvider


def _decoding_parameters() -> DecodingParameters:
    return DecodingParameters(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=128,
        seed=17,
    )


def _zero_action() -> ControllerAction:
    return ControllerAction(
        temperature_delta=0.0,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )


def _policy_context() -> PolicyContext:
    provider_identity = FakeProvider().provider_identity
    condition = ExperimentCondition(
        experiment_id="phase-1-contract",
        dataset_version="development-v1",
        prompt_sequence_id="sequence-001",
        turn_index=0,
        policy_id="zero",
        model_seed=17,
        controller_seed=23,
        provider_identity_id=provider_identity.identity_id,
        base_decoding_profile_id="base-v1",
    )
    return PolicyContext(
        condition=condition,
        initial_decoding_parameters=_decoding_parameters(),
        action_bounds=ActionBounds(),
    )


class _ZeroPolicy:
    policy_id = "zero"

    def initial_state(self, context: PolicyContext) -> PolicyState:
        del context
        return PolicyState()

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, PolicyState, PolicyTrace]:
        action = _zero_action()
        return (
            action,
            state,
            PolicyTrace(
                policy_id=self.policy_id,
                turn_index=observation.turn_index,
                action=action,
            ),
        )


def test_one_structural_policy_contract_supports_a_policy_without_dispatch() -> None:
    policy = _ZeroPolicy()
    assert isinstance(policy, ControlPolicy)

    context = _policy_context()
    state = policy.initial_state(context)
    observation = ControllerObservation(
        turn_index=0,
        prompt_family="contract",
        current_prompt_features=PromptFeatures({"word_count": 4.0}),
        previous_response_metrics=None,
        has_previous_response=False,
    )

    action, next_state, trace = policy.act(observation, state)

    assert context.action_bounds.require(action) is action
    assert action == _zero_action()
    assert next_state is state
    assert trace == PolicyTrace(
        policy_id="zero",
        turn_index=0,
        action=action,
    )


def test_policy_boundary_models_are_strict_frozen_and_extra_forbid() -> None:
    context = _policy_context()
    trace = PolicyTrace(policy_id="zero", turn_index=0, action=_zero_action())

    with pytest.raises(ValidationError, match="frozen"):
        context.initial_decoding_parameters = _decoding_parameters()

    with pytest.raises(ValidationError):
        PolicyTrace.model_validate(
            {
                "policy_id": "zero",
                "turn_index": "0",
                "action": _zero_action(),
            }
        )

    with pytest.raises(ValidationError):
        PolicyState.model_validate({"untyped_state": True})

    assert trace.action == _zero_action()

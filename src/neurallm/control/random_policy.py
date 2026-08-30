"""Deterministic bounded random control policy."""

from __future__ import annotations

from pydantic import Field

from neurallm.control.policy import PolicyContext, PolicyState, PolicyTrace
from neurallm.control.specs import RandomMatchedPolicySpec
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
)
from neurallm.domain.serialization import canonical_sha256

_UINT64_RANGE = 2**64


def _zero_action() -> ControllerAction:
    return ControllerAction(
        temperature_delta=0.0,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )


class RandomMatchedState(PolicyState):
    """Immutable trajectory identity needed to replay each random draw."""

    initial_condition_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    controller_seed: int
    action_bounds: ActionBounds
    next_turn_index: int = Field(default=0, ge=0)


class RandomMatchedTrace(PolicyTrace):
    """Auditable identity of the deterministic draw used for one action."""

    draw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RandomMatchedPolicy:
    """Draw bounded actions from condition identity and controller seed only."""

    __slots__ = ("policy_id", "spec")

    def __init__(self, spec: RandomMatchedPolicySpec | None = None) -> None:
        resolved_spec = spec if spec is not None else RandomMatchedPolicySpec()
        if not isinstance(resolved_spec, RandomMatchedPolicySpec):
            raise TypeError("spec must be a RandomMatchedPolicySpec")
        self.spec = resolved_spec
        self.policy_id = resolved_spec.policy_id

    def initial_state(self, context: PolicyContext) -> RandomMatchedState:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if context.condition.policy_id != self.policy_id:
            raise ValueError("policy context targets another policy")
        if context.condition.turn_index != 0:
            raise ValueError("random_matched state must initialize at turn zero")
        return RandomMatchedState(
            initial_condition_id=context.condition.condition_id,
            controller_seed=context.condition.controller_seed,
            action_bounds=context.action_bounds,
        )

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, RandomMatchedState, RandomMatchedTrace]:
        if not isinstance(observation, ControllerObservation):
            raise TypeError("observation must be a ControllerObservation")
        if not isinstance(state, RandomMatchedState):
            raise TypeError("state must be a RandomMatchedState")
        if observation.turn_index != state.next_turn_index:
            raise ValueError("observation turn does not match random policy state")

        draw_sha256 = canonical_sha256(
            {
                "algorithm": self.spec.implementation_version,
                "controller_seed": state.controller_seed,
                "initial_condition_id": state.initial_condition_id,
                "turn_index": observation.turn_index,
            }
        )
        action = (
            _zero_action()
            if observation.turn_index == 0
            else self._action_from_digest(draw_sha256, state.action_bounds)
        )
        state.action_bounds.require(action)
        next_state = RandomMatchedState(
            initial_condition_id=state.initial_condition_id,
            controller_seed=state.controller_seed,
            action_bounds=state.action_bounds,
            next_turn_index=state.next_turn_index + 1,
        )
        return (
            action,
            next_state,
            RandomMatchedTrace(
                policy_id=self.policy_id,
                turn_index=observation.turn_index,
                action=action,
                draw_sha256=draw_sha256,
            ),
        )

    @staticmethod
    def _action_from_digest(
        draw_sha256: str,
        bounds: ActionBounds,
    ) -> ControllerAction:
        digest = bytes.fromhex(draw_sha256)
        draws = tuple(
            int.from_bytes(digest[offset : offset + 8], byteorder="big")
            for offset in range(0, 32, 8)
        )

        def draw_float(value: int, interval: tuple[float, float]) -> float:
            fraction = value / _UINT64_RANGE
            return interval[0] + fraction * (interval[1] - interval[0])

        top_k_span = bounds.top_k_delta[1] - bounds.top_k_delta[0] + 1
        return ControllerAction(
            temperature_delta=draw_float(draws[0], bounds.temperature_delta),
            top_p_delta=draw_float(draws[1], bounds.top_p_delta),
            top_k_delta=bounds.top_k_delta[0] + draws[2] % top_k_span,
            presence_penalty_delta=draw_float(draws[3], bounds.presence_penalty_delta),
        )


__all__ = [
    "RandomMatchedPolicy",
    "RandomMatchedState",
    "RandomMatchedTrace",
]

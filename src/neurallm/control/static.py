"""A fixed no-op policy used to exercise the Phase 2 experiment kernel."""

from __future__ import annotations

from neurallm.control.policy import PolicyContext, PolicyState, PolicyTrace
from neurallm.domain.models import ControllerAction, ControllerObservation


class FixedPolicy:
    """Preserve the plan's base decoding profile without claiming baseline selection."""

    __slots__ = ("policy_id",)

    def __init__(self, policy_id: str = "kernel_fixed") -> None:
        if not isinstance(policy_id, str):
            raise TypeError("policy_id must be a string")
        if not policy_id.strip():
            raise ValueError("policy_id must not be blank")
        self.policy_id = policy_id

    def initial_state(self, context: PolicyContext) -> PolicyState:
        if context.condition.policy_id != self.policy_id:
            raise ValueError("policy context targets another policy")
        return PolicyState()

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, PolicyState, PolicyTrace]:
        if not isinstance(state, PolicyState):
            raise TypeError("state must be a PolicyState")
        action = ControllerAction(
            temperature_delta=0.0,
            top_p_delta=0.0,
            top_k_delta=0,
            presence_penalty_delta=0.0,
        )
        return (
            action,
            state,
            PolicyTrace(
                policy_id=self.policy_id,
                turn_index=observation.turn_index,
                action=action,
            ),
        )


__all__ = ["FixedPolicy"]

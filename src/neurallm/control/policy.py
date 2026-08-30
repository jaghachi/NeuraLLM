"""Shared controller policy contract.

Every controller strategy implements :class:`ControlPolicy`.  Keeping one
structural interface lets the experiment runner invoke policies without
policy-specific dispatch branches.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    DecodingParameters,
    ExperimentCondition,
)


class _StrictFrozenModel(BaseModel):
    """Base configuration for immutable policy-boundary values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PolicyContext(_StrictFrozenModel):
    """Immutable inputs available when a policy initializes its state."""

    condition: ExperimentCondition
    initial_decoding_parameters: DecodingParameters
    action_bounds: ActionBounds


class PolicyState(_StrictFrozenModel):
    """Extensible immutable base for controller-owned state.

    Stateless policies may use this model directly.  Stateful policies define
    frozen subclasses with their own explicitly typed fields.
    """


class PolicyTrace(_StrictFrozenModel):
    """Fields common to every auditable policy decision."""

    policy_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    action: ControllerAction


@runtime_checkable
class ControlPolicy(Protocol):
    """One policy interface shared by every experiment arm."""

    policy_id: str

    def initial_state(self, context: PolicyContext) -> PolicyState:
        """Return this policy's immutable initial state."""
        ...

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, PolicyState, PolicyTrace]:
        """Select one bounded action and return the next state and trace."""
        ...


__all__ = ["ControlPolicy", "PolicyContext", "PolicyState", "PolicyTrace"]

"""Bounded decoding-action projection for the simulated neural state."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from neurallm.control.action_space import normalized_action_magnitude
from neurallm.control.neural.substrate import NeuralSubstrateState
from neurallm.domain.models import ActionBounds, ControllerAction


class DecoderActivation(BaseModel):
    """Normalized pre-action activations for all four controlled parameters."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    temperature: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    top_p: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    top_k: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    presence_penalty: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class DecodedAction(BaseModel):
    """Complete transparent decoder result before legal parameter clamping."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    decoder_version: Literal["four-surface-linear-decoder-v1"] = "four-surface-linear-decoder-v1"
    action_magnitude_version: Literal["rms-normalized-to-action-bounds-v1"] = (
        "rms-normalized-to-action-bounds-v1"
    )
    activation: DecoderActivation
    action: ControllerAction
    action_magnitude: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ActionDecoder:
    """Project neural state through four distinct fixed linear readouts."""

    __slots__ = ()

    def decode(
        self,
        state: NeuralSubstrateState,
        bounds: ActionBounds,
        *,
        action_enabled: bool,
    ) -> DecodedAction:
        if not isinstance(state, NeuralSubstrateState):
            raise TypeError("state must be a NeuralSubstrateState")
        if not isinstance(bounds, ActionBounds):
            raise TypeError("bounds must be ActionBounds")
        if not isinstance(action_enabled, bool):
            raise TypeError("action_enabled must be a bool")

        activation = DecoderActivation(
            temperature=self._clip(
                0.55 * state.excitation - 0.35 * state.inhibition + 0.25 * state.context
            ),
            top_p=self._clip(
                0.50 * state.excitation - 0.30 * state.adaptation - 0.20 * state.fatigue
            ),
            top_k=self._clip(
                0.45 * state.adaptation + 0.35 * state.context - 0.25 * state.inhibition
            ),
            presence_penalty=self._clip(
                0.60 * state.inhibition + 0.40 * state.fatigue - 0.20 * state.excitation
            ),
        )
        action = (
            self._action_from_activation(activation, bounds)
            if action_enabled
            else ControllerAction(
                temperature_delta=0.0,
                top_p_delta=0.0,
                top_k_delta=0,
                presence_penalty_delta=0.0,
            )
        )
        bounds.require(action)
        return DecodedAction(
            activation=activation,
            action=action,
            action_magnitude=normalized_action_magnitude(action, bounds),
        )

    @classmethod
    def _action_from_activation(
        cls,
        activation: DecoderActivation,
        bounds: ActionBounds,
    ) -> ControllerAction:
        top_k_scale = bounds.top_k_delta[1] if activation.top_k >= 0.0 else -bounds.top_k_delta[0]
        return ControllerAction(
            temperature_delta=cls._scale(
                activation.temperature,
                bounds.temperature_delta,
            ),
            top_p_delta=cls._scale(activation.top_p, bounds.top_p_delta),
            top_k_delta=round(activation.top_k * top_k_scale),
            presence_penalty_delta=cls._scale(
                activation.presence_penalty,
                bounds.presence_penalty_delta,
            ),
        )

    @staticmethod
    def _scale(value: float, interval: tuple[float, float]) -> float:
        scale = interval[1] if value >= 0.0 else -interval[0]
        return value * scale

    @staticmethod
    def _clip(value: float) -> float:
        return round(min(max(value, -1.0), 1.0), 12)


__all__ = ["ActionDecoder", "DecodedAction", "DecoderActivation"]

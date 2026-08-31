"""Auditable application of controller actions to decoding parameters."""

from __future__ import annotations

from math import sqrt

from pydantic import BaseModel, ConfigDict

from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    DecodingBounds,
    DecodingParameters,
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class SaturationIndicator(_StrictFrozenModel):
    """Whether one parameter was changed by either clamping stage."""

    step_clamped: bool
    legal_clamped: bool

    @property
    def any_saturation(self) -> bool:
        return self.step_clamped or self.legal_clamped


class ActionSaturation(_StrictFrozenModel):
    """Per-parameter saturation audit for one applied action."""

    temperature: SaturationIndicator
    top_p: SaturationIndicator
    top_k: SaturationIndicator
    presence_penalty: SaturationIndicator

    @property
    def any_saturation(self) -> bool:
        return any(
            indicator.any_saturation
            for indicator in (
                self.temperature,
                self.top_p,
                self.top_k,
                self.presence_penalty,
            )
        )


class ActionApplication(_StrictFrozenModel):
    """Every distinct stage of applying one controller action."""

    raw_action: ControllerAction
    step_clamped_action: ControllerAction
    final_decoding_parameters: DecodingParameters
    saturation: ActionSaturation


def _clamp_float(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _clamp_int(value: int, bounds: tuple[int, int]) -> int:
    return min(max(value, bounds[0]), bounds[1])


def apply_action(
    base_parameters: DecodingParameters,
    raw_action: ControllerAction,
    action_bounds: ActionBounds,
    decoding_bounds: DecodingBounds,
) -> ActionApplication:
    """Step-clamp an action, apply it, then clamp to legal decoding ranges."""

    if not isinstance(base_parameters, DecodingParameters):
        raise TypeError("base_parameters must be DecodingParameters")
    if not isinstance(raw_action, ControllerAction):
        raise TypeError("raw_action must be ControllerAction")
    if not isinstance(action_bounds, ActionBounds):
        raise TypeError("action_bounds must be ActionBounds")
    if not isinstance(decoding_bounds, DecodingBounds):
        raise TypeError("decoding_bounds must be DecodingBounds")

    step_clamped_action = ControllerAction(
        temperature_delta=_clamp_float(
            raw_action.temperature_delta,
            action_bounds.temperature_delta,
        ),
        top_p_delta=_clamp_float(
            raw_action.top_p_delta,
            action_bounds.top_p_delta,
        ),
        top_k_delta=_clamp_int(
            raw_action.top_k_delta,
            action_bounds.top_k_delta,
        ),
        presence_penalty_delta=_clamp_float(
            raw_action.presence_penalty_delta,
            action_bounds.presence_penalty_delta,
        ),
    )

    temperature_candidate = base_parameters.temperature + step_clamped_action.temperature_delta
    top_p_candidate = base_parameters.top_p + step_clamped_action.top_p_delta
    top_k_candidate = base_parameters.top_k + step_clamped_action.top_k_delta
    presence_penalty_candidate = (
        base_parameters.presence_penalty + step_clamped_action.presence_penalty_delta
    )

    final_temperature = _clamp_float(
        temperature_candidate,
        decoding_bounds.temperature,
    )
    final_top_p = _clamp_float(top_p_candidate, decoding_bounds.top_p)
    final_top_k = _clamp_int(top_k_candidate, decoding_bounds.top_k)
    final_presence_penalty = _clamp_float(
        presence_penalty_candidate,
        decoding_bounds.presence_penalty,
    )
    final_parameters = DecodingParameters(
        temperature=final_temperature,
        top_p=final_top_p,
        top_k=final_top_k,
        presence_penalty=final_presence_penalty,
        max_tokens=base_parameters.max_tokens,
        seed=base_parameters.seed,
    )

    return ActionApplication(
        raw_action=raw_action,
        step_clamped_action=step_clamped_action,
        final_decoding_parameters=final_parameters,
        saturation=ActionSaturation(
            temperature=SaturationIndicator(
                step_clamped=(
                    raw_action.temperature_delta != step_clamped_action.temperature_delta
                ),
                legal_clamped=temperature_candidate != final_temperature,
            ),
            top_p=SaturationIndicator(
                step_clamped=raw_action.top_p_delta != step_clamped_action.top_p_delta,
                legal_clamped=top_p_candidate != final_top_p,
            ),
            top_k=SaturationIndicator(
                step_clamped=raw_action.top_k_delta != step_clamped_action.top_k_delta,
                legal_clamped=top_k_candidate != final_top_k,
            ),
            presence_penalty=SaturationIndicator(
                step_clamped=(
                    raw_action.presence_penalty_delta != step_clamped_action.presence_penalty_delta
                ),
                legal_clamped=presence_penalty_candidate != final_presence_penalty,
            ),
        ),
    )


def normalized_action_magnitude(
    action: ControllerAction,
    bounds: ActionBounds,
) -> float:
    """Return RMS action magnitude normalized to the declared run bounds."""

    if not isinstance(action, ControllerAction):
        raise TypeError("action must be a ControllerAction")
    if not isinstance(bounds, ActionBounds):
        raise TypeError("bounds must be ActionBounds")
    components = (
        (float(action.temperature_delta), bounds.temperature_delta),
        (float(action.top_p_delta), bounds.top_p_delta),
        (float(action.top_k_delta), bounds.top_k_delta),
        (float(action.presence_penalty_delta), bounds.presence_penalty_delta),
    )
    normalized: list[float] = []
    for value, interval in components:
        scale = max(abs(float(interval[0])), abs(float(interval[1])))
        if scale == 0.0:
            if value != 0.0:
                raise ValueError("nonzero action uses a zero-width configured bound")
            normalized.append(0.0)
        else:
            normalized.append(value / scale)
    return sqrt(sum(value * value for value in normalized) / len(normalized))


__all__ = [
    "ActionApplication",
    "ActionSaturation",
    "SaturationIndicator",
    "apply_action",
    "normalized_action_magnitude",
]

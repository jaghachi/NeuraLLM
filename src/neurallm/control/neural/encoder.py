"""Causal observation encoding for the simulated neural controller."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from neurallm.domain.models import (
    ControllerObservation,
    CountMetricValue,
    UnitIntervalMetricValue,
)

_TARGET_RESPONSE_LENGTH_TOKENS = 64
_RESPONSE_LENGTH_SCALE_TOKENS = 64
_PROMPT_FEATURE_SCALE = 64.0


class EncodedObservation(BaseModel):
    """Bounded numeric inputs supplied to the neural substrate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    encoder_version: Literal["causal-five-signal-v1"] = "causal-five-signal-v1"
    turn_index: int = Field(ge=0)
    repetition_signal: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    adherence_deficit: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    task_deficit: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    length_deviation: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    prompt_signal: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    has_history: bool


class ObservationEncoder:
    """Convert only causally available observations into bounded signals.

    Null turn-zero history is converted to explicit calculation defaults after
    its absence is observed.  Those zero-valued signals are not represented as
    historical measurements.
    """

    __slots__ = ()

    def encode(self, observation: ControllerObservation) -> EncodedObservation:
        """Return the deterministic bounded encoding for one policy decision."""

        if not isinstance(observation, ControllerObservation):
            raise TypeError("observation must be a ControllerObservation")
        prompt_signal = self._prompt_signal(observation)
        metrics = observation.previous_response_metrics
        if metrics is None:
            if observation.turn_index != 0 or observation.has_previous_response:
                raise ValueError("only turn zero may use null neural history")
            return EncodedObservation(
                turn_index=observation.turn_index,
                repetition_signal=0.0,
                adherence_deficit=0.0,
                task_deficit=0.0,
                length_deviation=0.0,
                prompt_signal=prompt_signal,
                has_history=False,
            )

        if observation.turn_index == 0 or not observation.has_previous_response:
            raise ValueError("non-null neural history is invalid at turn zero")
        repetition = self._required_unit(metrics.repetition_ratio, "repetition_ratio")
        adherence = self._required_unit(
            metrics.instruction_adherence,
            "instruction_adherence",
        )
        task_score = self._required_unit(metrics.task_score, "task_score")
        response_length = self._required_count(
            metrics.response_length_tokens,
            "response_length_tokens",
        )
        length_deviation = self._clamp(
            (response_length - _TARGET_RESPONSE_LENGTH_TOKENS) / _RESPONSE_LENGTH_SCALE_TOKENS,
            -1.0,
            1.0,
        )
        return EncodedObservation(
            turn_index=observation.turn_index,
            repetition_signal=repetition,
            adherence_deficit=1.0 - adherence,
            task_deficit=1.0 - task_score,
            length_deviation=length_deviation,
            prompt_signal=prompt_signal,
            has_history=True,
        )

    def _prompt_signal(self, observation: ControllerObservation) -> float:
        values = tuple(observation.current_prompt_features.root.values())
        if not values:
            return 0.0
        normalized = tuple(
            self._clamp(value / _PROMPT_FEATURE_SCALE, -1.0, 1.0) for value in values
        )
        return self._clamp(sum(normalized) / len(normalized), -1.0, 1.0)

    @staticmethod
    def _required_unit(metric: UnitIntervalMetricValue, name: str) -> float:
        if not metric.availability or metric.value is None:
            raise ValueError(f"{name} must be available to the neural controller")
        return metric.value

    @staticmethod
    def _required_count(metric: CountMetricValue, name: str) -> int:
        if not metric.availability or metric.value is None:
            raise ValueError(f"{name} must be available to the neural controller")
        return metric.value

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


__all__ = ["EncodedObservation", "ObservationEncoder"]

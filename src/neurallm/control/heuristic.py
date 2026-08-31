"""Transparent non-neural adaptive controller."""

from __future__ import annotations

from pydantic import Field

from neurallm.control.policy import PolicyContext, PolicyState, PolicyTrace
from neurallm.control.specs import HeuristicAdaptivePolicySpec
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    MetricValue,
)


def _zero_action() -> ControllerAction:
    return ControllerAction(
        temperature_delta=0.0,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )


class HeuristicAdaptiveState(PolicyState):
    """Only the last declared action and trajectory position persist."""

    last_action: ControllerAction = Field(default_factory=_zero_action)
    action_bounds: ActionBounds
    next_turn_index: int = Field(default=0, ge=0)


class HeuristicAdaptiveTrace(PolicyTrace):
    """Typed explanation of the signals behind one heuristic action."""

    repetition_excess: float = Field(ge=0.0, le=1.0)
    adherence_deficit: float = Field(ge=0.0, le=1.0)
    length_deviation: float = Field(ge=-1.0, le=1.0)
    combined_drive: float = Field(ge=-1.0, le=1.0)
    clean_decay_applied: bool


class HeuristicAdaptivePolicy:
    """React to prior output problems and decay toward the base profile."""

    __slots__ = ("policy_id", "spec")

    def __init__(self, spec: HeuristicAdaptivePolicySpec | None = None) -> None:
        resolved_spec = spec if spec is not None else HeuristicAdaptivePolicySpec()
        if not isinstance(resolved_spec, HeuristicAdaptivePolicySpec):
            raise TypeError("spec must be a HeuristicAdaptivePolicySpec")
        self.spec = resolved_spec
        self.policy_id = resolved_spec.policy_id

    def initial_state(self, context: PolicyContext) -> HeuristicAdaptiveState:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if context.condition.policy_id != self.policy_id:
            raise ValueError("policy context targets another policy")
        if context.condition.turn_index != 0:
            raise ValueError("heuristic_adaptive state must initialize at turn zero")
        return HeuristicAdaptiveState(action_bounds=context.action_bounds)

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, HeuristicAdaptiveState, HeuristicAdaptiveTrace]:
        if not isinstance(observation, ControllerObservation):
            raise TypeError("observation must be a ControllerObservation")
        if not isinstance(state, HeuristicAdaptiveState):
            raise TypeError("state must be a HeuristicAdaptiveState")
        if observation.turn_index != state.next_turn_index:
            raise ValueError("observation turn does not match heuristic policy state")
        state.action_bounds.require(state.last_action)

        if observation.previous_response_metrics is None:
            if observation.turn_index != 0:
                raise ValueError("nonzero heuristic observation requires previous-response metrics")
            action = _zero_action()
            repetition_excess = 0.0
            adherence_deficit = 0.0
            length_deviation = 0.0
            combined_drive = 0.0
            clean_decay_applied = False
        else:
            metrics = observation.previous_response_metrics
            repetition = self._required_float(metrics.repetition_ratio, "repetition_ratio")
            adherence = self._required_float(
                metrics.instruction_adherence,
                "instruction_adherence",
            )
            response_length = self._required_int(
                metrics.response_length_tokens,
                "response_length_tokens",
            )
            repetition_excess = self._repetition_excess(repetition)
            adherence_deficit = self._adherence_deficit(adherence)
            length_deviation = self._length_deviation(response_length)
            clean_decay_applied = (
                repetition_excess == 0.0 and adherence_deficit == 0.0 and length_deviation == 0.0
            )
            if clean_decay_applied:
                action = self._decay_action(state.last_action, state.action_bounds)
                combined_drive = 0.0
            else:
                combined_drive = max(
                    -1.0,
                    min(
                        1.0,
                        self.spec.repetition_reaction_fraction * repetition_excess
                        - self.spec.adherence_reaction_fraction * adherence_deficit
                        + self.spec.length_reaction_fraction * length_deviation,
                    ),
                )
                action = self._action_from_drive(combined_drive, state.action_bounds)

        state.action_bounds.require(action)
        next_state = HeuristicAdaptiveState(
            last_action=action,
            action_bounds=state.action_bounds,
            next_turn_index=state.next_turn_index + 1,
        )
        return (
            action,
            next_state,
            HeuristicAdaptiveTrace(
                policy_id=self.policy_id,
                turn_index=observation.turn_index,
                action=action,
                repetition_excess=repetition_excess,
                adherence_deficit=adherence_deficit,
                length_deviation=length_deviation,
                combined_drive=combined_drive,
                clean_decay_applied=clean_decay_applied,
            ),
        )

    @staticmethod
    def _required_float(metric: MetricValue[float], name: str) -> float:
        if not metric.availability or metric.value is None:
            raise ValueError(f"{name} must be available to heuristic_adaptive")
        return metric.value

    @staticmethod
    def _required_int(metric: MetricValue[int], name: str) -> int:
        if not metric.availability or metric.value is None:
            raise ValueError(f"{name} must be available to heuristic_adaptive")
        return metric.value

    def _repetition_excess(self, repetition: float) -> float:
        threshold = self.spec.high_repetition_threshold
        return max(0.0, min(1.0, (repetition - threshold) / (1.0 - threshold)))

    def _adherence_deficit(self, adherence: float) -> float:
        threshold = self.spec.minimum_instruction_adherence
        return max(0.0, min(1.0, (threshold - adherence) / threshold))

    def _length_deviation(self, response_length: int) -> float:
        minimum = self.spec.minimum_response_length_tokens
        maximum = self.spec.maximum_response_length_tokens
        if response_length < minimum:
            return min(1.0, (minimum - response_length) / max(minimum, 1))
        if response_length > maximum:
            return -min(1.0, (response_length - maximum) / maximum)
        return 0.0

    @staticmethod
    def _scaled_float(drive: float, interval: tuple[float, float]) -> float:
        return drive * (interval[1] if drive >= 0.0 else -interval[0])

    @classmethod
    def _action_from_drive(
        cls,
        drive: float,
        bounds: ActionBounds,
    ) -> ControllerAction:
        top_k_limit = bounds.top_k_delta[1] if drive >= 0.0 else -bounds.top_k_delta[0]
        return ControllerAction(
            temperature_delta=cls._scaled_float(drive, bounds.temperature_delta),
            top_p_delta=cls._scaled_float(drive, bounds.top_p_delta),
            top_k_delta=round(drive * top_k_limit),
            presence_penalty_delta=cls._scaled_float(
                drive,
                bounds.presence_penalty_delta,
            ),
        )

    def _decay_action(
        self,
        previous: ControllerAction,
        bounds: ActionBounds,
    ) -> ControllerAction:
        retained_fraction = 1.0 - self.spec.clean_decay_fraction
        decayed = ControllerAction(
            temperature_delta=previous.temperature_delta * retained_fraction,
            top_p_delta=previous.top_p_delta * retained_fraction,
            top_k_delta=round(previous.top_k_delta * retained_fraction),
            presence_penalty_delta=previous.presence_penalty_delta * retained_fraction,
        )
        return bounds.require(decayed)


__all__ = [
    "HeuristicAdaptivePolicy",
    "HeuristicAdaptiveState",
    "HeuristicAdaptiveTrace",
]

"""Persistent and matched-history-reset simulated neural policies."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from neurallm.control.neural.decoder import ActionDecoder, DecoderActivation
from neurallm.control.neural.encoder import EncodedObservation, ObservationEncoder
from neurallm.control.neural.substrate import (
    NeuralSubstrate,
    NeuralSubstrateState,
    NeuralTransition,
)
from neurallm.control.policy import PolicyContext, PolicyState, PolicyTrace
from neurallm.control.specs import (
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
)
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    Sha256Hex,
    SqliteInt64,
)
from neurallm.domain.serialization import canonical_sha256

NeuralPolicySpec = NeuralPersistentPolicySpec | NeuralMatchedHistoryStateResetPolicySpec


class NeuralPolicyState(PolicyState):
    """Complete persisted controller envelope and its declared neural state."""

    state_schema_version: Literal["simulated-neural-state-v1"] = "simulated-neural-state-v1"
    substrate: NeuralSubstrateState
    controller_seed: SqliteInt64
    action_bounds: ActionBounds
    next_turn_index: int = Field(default=0, ge=0)


class NeuralPolicyTrace(PolicyTrace):
    """Complete causal mechanism trace for one simulated-neural decision."""

    trace_schema_version: Literal["simulated-neural-policy-trace-v1"] = (
        "simulated-neural-policy-trace-v1"
    )
    observation_encoding: EncodedObservation
    stored_substrate_state: NeuralSubstrateState
    effective_substrate_state: NeuralSubstrateState
    substrate_transition: NeuralTransition
    decoder_version: Literal["four-surface-linear-decoder-v1"]
    action_magnitude_version: Literal["rms-normalized-to-action-bounds-v1"]
    decoder_activation: DecoderActivation
    action_magnitude: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    state_reset_applied: bool
    mechanism_sha256: Sha256Hex

    @model_validator(mode="after")
    def _validate_transition_boundary(self) -> NeuralPolicyTrace:
        if self.substrate_transition.state_before != self.effective_substrate_state:
            raise ValueError("substrate transition does not start from effective state")
        expected = canonical_sha256(
            {
                "turn_index": self.turn_index,
                "observation_encoding": self.observation_encoding,
                "stored_substrate_state": self.stored_substrate_state,
                "effective_substrate_state": self.effective_substrate_state,
                "substrate_transition": self.substrate_transition,
                "decoder_version": self.decoder_version,
                "action_magnitude_version": self.action_magnitude_version,
                "decoder_activation": self.decoder_activation,
                "action": self.action,
                "action_magnitude": self.action_magnitude,
                "state_reset_applied": self.state_reset_applied,
            }
        )
        if self.mechanism_sha256 != expected:
            raise ValueError("mechanism_sha256 does not match the neural decision")
        return self


class SimulatedNeuralPolicy:
    """One fixed neural mechanism with persistent and substrate-reset roles."""

    __slots__ = ("_spec",)
    _spec: NeuralPolicySpec

    def __init__(self, spec: NeuralPolicySpec) -> None:
        if not isinstance(
            spec,
            (NeuralPersistentPolicySpec, NeuralMatchedHistoryStateResetPolicySpec),
        ):
            raise TypeError("spec must be a simulated neural policy specification")
        object.__setattr__(self, "_spec", spec)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    @property
    def spec(self) -> NeuralPolicySpec:
        return self._spec

    @property
    def policy_id(self) -> str:
        return self._spec.policy_id

    @property
    def encoder(self) -> ObservationEncoder:
        return ObservationEncoder()

    @property
    def substrate(self) -> NeuralSubstrate:
        return NeuralSubstrate()

    @property
    def decoder(self) -> ActionDecoder:
        return ActionDecoder()

    def initial_state(self, context: PolicyContext) -> NeuralPolicyState:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if context.condition.policy_id != self.policy_id:
            raise ValueError("policy context targets another policy")
        if context.condition.turn_index != 0:
            raise ValueError("simulated neural state must initialize at turn zero")
        return NeuralPolicyState(
            substrate=self.substrate.initial_state(context.condition.controller_seed),
            controller_seed=context.condition.controller_seed,
            action_bounds=context.action_bounds,
        )

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, NeuralPolicyState, NeuralPolicyTrace]:
        if not isinstance(observation, ControllerObservation):
            raise TypeError("observation must be a ControllerObservation")
        if not isinstance(state, NeuralPolicyState):
            raise TypeError("state must be a NeuralPolicyState")
        if observation.turn_index != state.next_turn_index:
            raise ValueError("observation turn does not match simulated neural state")
        if observation.turn_index > 0 and observation.previous_response_metrics is None:
            raise ValueError("nonzero neural observation requires previous-response metrics")

        state_reset_applied = (
            isinstance(
                self.spec,
                NeuralMatchedHistoryStateResetPolicySpec,
            )
            and observation.turn_index > 0
        )
        effective_substrate = (
            self.substrate.initial_state(state.controller_seed)
            if state_reset_applied
            else state.substrate
        )
        encoded = self.encoder.encode(observation)
        transition = self.substrate.step(
            effective_substrate,
            encoded,
            state.controller_seed,
        )
        decoded = self.decoder.decode(
            transition.state_after,
            state.action_bounds,
            action_enabled=observation.turn_index > 0,
        )
        next_state = NeuralPolicyState(
            substrate=transition.state_after,
            controller_seed=state.controller_seed,
            action_bounds=state.action_bounds,
            next_turn_index=state.next_turn_index + 1,
        )
        mechanism_payload = {
            "turn_index": observation.turn_index,
            "observation_encoding": encoded,
            "stored_substrate_state": state.substrate,
            "effective_substrate_state": effective_substrate,
            "substrate_transition": transition,
            "decoder_version": decoded.decoder_version,
            "action_magnitude_version": decoded.action_magnitude_version,
            "decoder_activation": decoded.activation,
            "action": decoded.action,
            "action_magnitude": decoded.action_magnitude,
            "state_reset_applied": state_reset_applied,
        }
        trace = NeuralPolicyTrace(
            policy_id=self.policy_id,
            turn_index=observation.turn_index,
            action=decoded.action,
            observation_encoding=encoded,
            stored_substrate_state=state.substrate,
            effective_substrate_state=effective_substrate,
            substrate_transition=transition,
            decoder_version=decoded.decoder_version,
            action_magnitude_version=decoded.action_magnitude_version,
            decoder_activation=decoded.activation,
            action_magnitude=decoded.action_magnitude,
            state_reset_applied=state_reset_applied,
            mechanism_sha256=canonical_sha256(mechanism_payload),
        )
        return decoded.action, next_state, trace


class NeuralPersistentPolicy(SimulatedNeuralPolicy):
    """End-to-end policy that carries its five-variable neural state."""

    __slots__ = ()

    def __init__(self, spec: NeuralPersistentPolicySpec | None = None) -> None:
        super().__init__(spec if spec is not None else NeuralPersistentPolicySpec())


class NeuralMatchedHistoryStateResetPolicy(SimulatedNeuralPolicy):
    """Attribution-only policy that resets only the neural substrate."""

    __slots__ = ()

    def __init__(
        self,
        spec: NeuralMatchedHistoryStateResetPolicySpec | None = None,
    ) -> None:
        super().__init__(spec if spec is not None else NeuralMatchedHistoryStateResetPolicySpec())


__all__ = [
    "NeuralMatchedHistoryStateResetPolicy",
    "NeuralPersistentPolicy",
    "NeuralPolicyState",
    "NeuralPolicyTrace",
    "SimulatedNeuralPolicy",
]

"""Small deterministic and interpretable simulated neural substrate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from neurallm.control.neural.encoder import EncodedObservation
from neurallm.domain.serialization import canonical_sha256

_UINT32_RANGE = 2**32


class NeuralSubstrateState(BaseModel):
    """The complete five-variable neural state; no hidden state exists."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    excitation: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    inhibition: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    adaptation: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    fatigue: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    context: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class NeuralStateSaturation(BaseModel):
    """Per-variable evidence that a raw update reached a state bound."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    excitation: bool
    inhibition: bool
    adaptation: bool
    fatigue: bool
    context: bool

    @property
    def any_saturation(self) -> bool:
        return any(self.model_dump().values())


class NeuralTransition(BaseModel):
    """Auditable inputs and outputs of one explicit substrate update."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    equation_version: Literal["bounded-five-state-v1"] = "bounded-five-state-v1"
    seed_drive: float = Field(ge=-0.02, le=0.02, allow_inf_nan=False)
    state_before: NeuralSubstrateState
    state_after: NeuralSubstrateState
    saturation: NeuralStateSaturation


class NeuralSubstrate:
    """Advance five bounded states with fixed, explicit, unlearned equations.

    The equations are intentionally linear before clipping and 12-decimal
    quantization.  They use no learned parameters, global RNG, I/O, or mutable
    object state.  ``seed_drive`` is a recorded deterministic function of the
    controller seed and logical turn index.
    """

    __slots__ = ()

    def initial_state(self, controller_seed: int) -> NeuralSubstrateState:
        """Derive a small deterministic baseline state from the declared seed."""

        if not isinstance(controller_seed, int):
            raise TypeError("controller_seed must be an integer")
        digest = bytes.fromhex(
            canonical_sha256(
                {
                    "algorithm": "simulated-neural-initial-state-v1",
                    "controller_seed": controller_seed,
                }
            )
        )
        centered = tuple(
            self._quantize(
                ((int.from_bytes(digest[index * 4 : index * 4 + 4], "big") / _UINT32_RANGE) * 0.10)
                - 0.05,
                -0.05,
                0.05,
            )
            for index in range(4)
        )
        fatigue = self._quantize(
            int.from_bytes(digest[16:20], "big") / _UINT32_RANGE * 0.05,
            0.0,
            0.05,
        )
        return NeuralSubstrateState(
            excitation=centered[0],
            inhibition=centered[1],
            adaptation=centered[2],
            fatigue=fatigue,
            context=centered[3],
        )

    def step(
        self,
        state: NeuralSubstrateState,
        encoded: EncodedObservation,
        controller_seed: int,
    ) -> NeuralTransition:
        """Apply the five documented equations and return the complete trace."""

        if not isinstance(state, NeuralSubstrateState):
            raise TypeError("state must be a NeuralSubstrateState")
        if not isinstance(encoded, EncodedObservation):
            raise TypeError("encoded must be an EncodedObservation")
        if not isinstance(controller_seed, int):
            raise TypeError("controller_seed must be an integer")
        seed_drive = self._seed_drive(controller_seed, encoded.turn_index)
        length_high = max(0.0, encoded.length_deviation)

        raw_excitation = (
            0.55 * state.excitation
            + 0.35 * encoded.repetition_signal
            + 0.20 * encoded.task_deficit
            - 0.15 * state.inhibition
            + 0.10 * encoded.prompt_signal
            + seed_drive
        )
        raw_inhibition = (
            0.60 * state.inhibition
            + 0.30 * encoded.adherence_deficit
            + 0.15 * length_high
            + 0.10 * state.fatigue
            - 0.10 * state.excitation
            - 0.50 * seed_drive
        )
        raw_adaptation = (
            0.65 * state.adaptation
            + 0.25 * encoded.length_deviation
            + 0.15 * (encoded.repetition_signal - encoded.adherence_deficit)
            + 0.10 * state.context
        )
        raw_fatigue = (
            0.70 * state.fatigue
            + 0.15 * abs(encoded.length_deviation)
            + 0.10 * encoded.repetition_signal
            + 0.05 * encoded.task_deficit
        )
        raw_context = (
            0.50 * state.context + 0.35 * encoded.prompt_signal + 0.15 * encoded.length_deviation
        )
        excitation, excitation_saturated = self._bounded(raw_excitation, -1.0, 1.0)
        inhibition, inhibition_saturated = self._bounded(raw_inhibition, -1.0, 1.0)
        adaptation, adaptation_saturated = self._bounded(raw_adaptation, -1.0, 1.0)
        fatigue, fatigue_saturated = self._bounded(raw_fatigue, 0.0, 1.0)
        context, context_saturated = self._bounded(raw_context, -1.0, 1.0)
        return NeuralTransition(
            seed_drive=seed_drive,
            state_before=state,
            state_after=NeuralSubstrateState(
                excitation=excitation,
                inhibition=inhibition,
                adaptation=adaptation,
                fatigue=fatigue,
                context=context,
            ),
            saturation=NeuralStateSaturation(
                excitation=excitation_saturated,
                inhibition=inhibition_saturated,
                adaptation=adaptation_saturated,
                fatigue=fatigue_saturated,
                context=context_saturated,
            ),
        )

    @staticmethod
    def _seed_drive(controller_seed: int, turn_index: int) -> float:
        digest = bytes.fromhex(
            canonical_sha256(
                {
                    "algorithm": "simulated-neural-seed-drive-v1",
                    "controller_seed": controller_seed,
                    "turn_index": turn_index,
                }
            )
        )
        fraction = int.from_bytes(digest[:4], "big") / _UINT32_RANGE
        return NeuralSubstrate._quantize(fraction * 0.04 - 0.02, -0.02, 0.02)

    @staticmethod
    def _quantize(value: float, lower: float, upper: float) -> float:
        return round(min(max(value, lower), upper), 12)

    @classmethod
    def _bounded(
        cls,
        value: float,
        lower: float,
        upper: float,
    ) -> tuple[float, bool]:
        return cls._quantize(value, lower, upper), not lower <= value <= upper


__all__ = [
    "NeuralStateSaturation",
    "NeuralSubstrate",
    "NeuralSubstrateState",
    "NeuralTransition",
]

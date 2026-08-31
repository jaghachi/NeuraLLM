"""Turn-level evidence and aggregation for development-pilot selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from neurallm.domain.models import (
    DecodingParameters,
    ExperimentCondition,
    Sha256Hex,
    UnitIntervalMetricValue,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import MatchedUnitKey
from neurallm.providers.base import GenerationMetadata
from neurallm.storage.models import TurnInputEvidence


class DevelopmentPilotTurnEvidence(BaseModel):
    """One committed ``best_static`` turn used by the selector."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    condition: ExperimentCondition
    request_sha256: Sha256Hex
    response_sha256: Sha256Hex
    generation_metadata: GenerationMetadata
    decoding_parameters: DecodingParameters
    turn_input: TurnInputEvidence
    task_score: UnitIntervalMetricValue

    @model_validator(mode="after")
    def _validate_turn_binding(self) -> Self:
        if self.condition.policy_id != "best_static":
            raise ValueError("pilot static-selection evidence accepts best_static turns only")
        if self.turn_input.condition_id != self.condition.condition_id:
            raise ValueError("pilot turn input targets another condition")
        if self.decoding_parameters.seed != self.condition.model_seed:
            raise ValueError("pilot best_static request seed differs from its model seed")
        if (
            self.generation_metadata.generation_method != "llama_cpp_completion_http_v1"
            or self.generation_metadata.request_sha256 != self.request_sha256
        ):
            raise ValueError(
                "pilot best_static turn requires request-bound llama_cpp protocol evidence"
            )
        if not self.task_score.availability or self.task_score.value is None:
            raise ValueError("pilot best_static selection requires every task score")
        return self


def pilot_turn_sort_key(
    turn: DevelopmentPilotTurnEvidence,
) -> tuple[str, int, int, int]:
    condition = turn.condition
    return (
        condition.prompt_sequence_id,
        condition.model_seed,
        condition.controller_seed,
        condition.turn_index,
    )


def prompt_input_sha256(turn: DevelopmentPilotTurnEvidence) -> str:
    evidence = turn.turn_input
    return canonical_sha256(
        {
            "prompt_case_id": evidence.prompt_case_id,
            "prompt_family": evidence.prompt_family,
            "prompt_features": evidence.prompt_features,
            "validator": evidence.validator,
        }
    )


def aggregate_pilot_unit_scores(
    turns: tuple[DevelopmentPilotTurnEvidence, ...],
) -> tuple[tuple[MatchedUnitKey, ...], tuple[float, ...]]:
    by_unit_and_controller: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for turn in turns:
        condition = turn.condition
        assert turn.task_score.value is not None
        by_unit_and_controller[(condition.prompt_sequence_id, condition.model_seed)][
            condition.controller_seed
        ].append(turn.task_score.value)

    unit_keys: list[MatchedUnitKey] = []
    unit_scores: list[float] = []
    for (sequence_id, model_seed), controller_scores in sorted(by_unit_and_controller.items()):
        unit_keys.append(MatchedUnitKey(prompt_sequence_id=sequence_id, model_seed=model_seed))
        controller_means = tuple(
            sum(scores) / len(scores) for _, scores in sorted(controller_scores.items())
        )
        unit_scores.append(sum(controller_means) / len(controller_means))
    return tuple(unit_keys), tuple(unit_scores)


__all__ = [
    "DevelopmentPilotTurnEvidence",
    "aggregate_pilot_unit_scores",
    "pilot_turn_sort_key",
    "prompt_input_sha256",
]

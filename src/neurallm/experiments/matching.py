"""Exact Phase 3 pairing and coverage identities at the declared analysis unit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import ExperimentCondition, Sha256Hex


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class MatchedUnitKey(_StrictFrozenModel):
    """One primary analysis unit; controller seeds and turns remain nested within it."""

    experiment_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    prompt_sequence_id: str = Field(min_length=1)
    model_seed: int


class MatchedCoverageExpectation(_StrictFrozenModel):
    """Exact complete Cartesian condition coverage required for one matched unit."""

    unit_key: MatchedUnitKey
    policy_ids: tuple[str, ...]
    controller_seeds: tuple[int, ...]
    turn_indexes: tuple[int, ...]
    expected_condition_ids: tuple[Sha256Hex, ...]
    expected_condition_count: int = Field(gt=0)

    @field_validator("policy_ids", "expected_condition_ids")
    @classmethod
    def _require_sorted_unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("matched coverage dimensions must be nonempty, sorted, and unique")
        return values

    @field_validator("controller_seeds", "turn_indexes")
    @classmethod
    def _require_sorted_unique_integers(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("matched coverage dimensions must be nonempty, sorted, and unique")
        return values

    @model_validator(mode="after")
    def _validate_expected_count(self) -> MatchedCoverageExpectation:
        cartesian_count = len(self.policy_ids) * len(self.controller_seeds) * len(self.turn_indexes)
        if self.expected_condition_count != cartesian_count:
            raise ValueError("matched coverage count does not equal the Cartesian schedule")
        if len(self.expected_condition_ids) != self.expected_condition_count:
            raise ValueError("matched condition IDs do not equal the expected count")
        return self


def materialize_matched_coverage(
    conditions: Sequence[ExperimentCondition],
    *,
    experiment_id: str,
    dataset_version: str,
    sequence_turn_indexes: Mapping[str, tuple[int, ...]],
    policy_ids: tuple[str, ...],
    model_seeds: tuple[int, ...],
    controller_seeds: tuple[int, ...],
) -> tuple[MatchedCoverageExpectation, ...]:
    """Materialize and validate every sequence-by-model-seed matched unit."""

    canonical_policy_ids = tuple(sorted(policy_ids))
    canonical_model_seeds = tuple(sorted(model_seeds))
    canonical_controller_seeds = tuple(sorted(controller_seeds))
    if not canonical_policy_ids or len(canonical_policy_ids) != len(set(canonical_policy_ids)):
        raise ValueError("matched policy IDs must be nonempty and unique")
    if not canonical_model_seeds or len(canonical_model_seeds) != len(set(canonical_model_seeds)):
        raise ValueError("matched model seeds must be nonempty and unique")
    if not canonical_controller_seeds or len(canonical_controller_seeds) != len(
        set(canonical_controller_seeds)
    ):
        raise ValueError("matched controller seeds must be nonempty and unique")

    indexed: dict[tuple[str, int], list[ExperimentCondition]] = {}
    for condition in conditions:
        if condition.experiment_id != experiment_id or condition.dataset_version != dataset_version:
            raise ValueError("condition belongs to another experiment or dataset")
        key = (condition.prompt_sequence_id, condition.model_seed)
        indexed.setdefault(key, []).append(condition)

    expected_unit_keys = tuple(
        (sequence_id, model_seed)
        for sequence_id, model_seed in product(
            sorted(sequence_turn_indexes),
            canonical_model_seeds,
        )
    )
    if set(indexed) != set(expected_unit_keys):
        raise ValueError("planned conditions do not cover every matched analysis unit")

    expectations: list[MatchedCoverageExpectation] = []
    for sequence_id, model_seed in expected_unit_keys:
        turn_indexes = tuple(sorted(sequence_turn_indexes[sequence_id]))
        expected_cells = set(
            product(canonical_policy_ids, canonical_controller_seeds, turn_indexes)
        )
        unit_conditions = indexed[(sequence_id, model_seed)]
        actual_cells = {
            (condition.policy_id, condition.controller_seed, condition.turn_index)
            for condition in unit_conditions
        }
        if actual_cells != expected_cells or len(unit_conditions) != len(expected_cells):
            raise ValueError("matched unit does not have exact Cartesian policy coverage")
        expectations.append(
            MatchedCoverageExpectation(
                unit_key=MatchedUnitKey(
                    experiment_id=experiment_id,
                    dataset_version=dataset_version,
                    prompt_sequence_id=sequence_id,
                    model_seed=model_seed,
                ),
                policy_ids=canonical_policy_ids,
                controller_seeds=canonical_controller_seeds,
                turn_indexes=turn_indexes,
                expected_condition_ids=tuple(
                    sorted(condition.condition_id for condition in unit_conditions)
                ),
                expected_condition_count=len(expected_cells),
            )
        )
    return tuple(expectations)


__all__ = [
    "MatchedCoverageExpectation",
    "MatchedUnitKey",
    "materialize_matched_coverage",
]

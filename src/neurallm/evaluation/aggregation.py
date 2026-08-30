"""Exact coverage and matched-unit aggregation for Phase 3."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.models import (
    CoverageResult,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    MatchedUnitKey,
    SequencePolicyOutcome,
    TurnEvaluationRecord,
    TurnRecordKey,
)


def turn_key_sort_key(key: TurnRecordKey) -> tuple[str, int, str, int, int]:
    """Return the canonical ordering key for a turn condition."""

    return (
        key.prompt_sequence_id,
        key.turn_index,
        key.policy_id,
        key.model_seed,
        key.controller_seed,
    )


def record_sort_key(record: TurnEvaluationRecord) -> tuple[str, int, str, int, int, str]:
    """Order records deterministically, including duplicate-key evidence."""

    return (*turn_key_sort_key(record.key), canonical_json(record))


def expected_turn_keys(design: ExpectedEvaluationDesign) -> tuple[TurnRecordKey, ...]:
    """Expand the frozen expected design into its exact condition grid."""

    keys = (
        TurnRecordKey(
            prompt_sequence_id=sequence.prompt_sequence_id,
            turn_index=turn_index,
            policy_id=policy_id,
            model_seed=model_seed,
            controller_seed=controller_seed,
        )
        for sequence in design.sequences
        for turn_index in range(sequence.turn_count)
        for policy_id in design.policy_ids
        for model_seed in design.model_seeds
        for controller_seed in design.controller_seeds
    )
    return tuple(sorted(keys, key=turn_key_sort_key))


def validate_exact_coverage(
    records: Sequence[TurnEvaluationRecord],
    design: ExpectedEvaluationDesign,
) -> CoverageResult:
    """Compare observed keys with the exact grid; never silently repair evidence."""

    expected = expected_turn_keys(design)
    expected_set = {turn_key_sort_key(key): key for key in expected}
    counts = Counter(turn_key_sort_key(record.key) for record in records)
    observed_keys: dict[tuple[str, int, str, int, int], TurnRecordKey] = {}
    for record in sorted(records, key=record_sort_key):
        observed_keys.setdefault(turn_key_sort_key(record.key), record.key)

    missing = tuple(expected_set[key] for key in sorted(expected_set.keys() - counts.keys()))
    unexpected = tuple(observed_keys[key] for key in sorted(counts.keys() - expected_set.keys()))
    duplicates = tuple(
        observed_keys[key] for key in sorted(key for key, count in counts.items() if count > 1)
    )
    return CoverageResult(
        exact=not missing and not unexpected and not duplicates and len(records) == len(expected),
        expected_count=len(expected),
        observed_count=len(records),
        missing_keys=missing,
        unexpected_keys=unexpected,
        duplicate_keys=duplicates,
    )


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty evidence group")
    return sum(materialized) / len(materialized)


def _required_metrics(record: TurnEvaluationRecord) -> tuple[float, float, float, float]:
    if (
        record.task_score is None
        or record.instruction_adherence is None
        or record.response_length_tokens is None
        or record.repetition_ratio is None
    ):
        raise ValueError("required evaluator metric is unavailable")
    return (
        record.task_score,
        record.instruction_adherence,
        float(record.response_length_tokens),
        record.repetition_ratio,
    )


def aggregate_matched_units(
    records: Sequence[TurnEvaluationRecord],
) -> tuple[SequencePolicyOutcome, ...]:
    """Average turns within controller seed, then controllers within each unit.

    Only the resulting ``prompt_sequence_id x model_seed`` values enter paired
    statistics.  Neither turns nor controller seeds inflate the sample size.
    """

    controller_groups: dict[tuple[str, int, str, int], list[TurnEvaluationRecord]] = defaultdict(
        list
    )
    for record in records:
        controller_groups[
            (
                record.prompt_sequence_id,
                record.model_seed,
                record.policy_id,
                record.controller_seed,
            )
        ].append(record)

    controller_means: dict[
        tuple[str, int, str],
        list[tuple[float, float, float, float, float, float, int]],
    ] = defaultdict(list)
    for (sequence_id, model_seed, policy_id, _), group in sorted(controller_groups.items()):
        ordered = sorted(group, key=record_sort_key)
        metrics = tuple(_required_metrics(record) for record in ordered)
        controller_means[(sequence_id, model_seed, policy_id)].append(
            (
                _mean(metric[0] for metric in metrics),
                _mean(metric[1] for metric in metrics),
                _mean(metric[2] for metric in metrics),
                _mean(metric[3] for metric in metrics),
                _mean(record.action_magnitude for record in ordered),
                _mean(float(record.action_saturated) for record in ordered),
                len(ordered),
            )
        )

    outcomes: list[SequencePolicyOutcome] = []
    for (sequence_id, model_seed, policy_id), controller_values in sorted(controller_means.items()):
        outcomes.append(
            SequencePolicyOutcome(
                unit_key=MatchedUnitKey(
                    prompt_sequence_id=sequence_id,
                    model_seed=model_seed,
                ),
                policy_id=policy_id,
                controller_seed_count=len(controller_values),
                turn_observation_count=sum(value[6] for value in controller_values),
                task_score=_mean(value[0] for value in controller_values),
                instruction_adherence=_mean(value[1] for value in controller_values),
                response_length_tokens=_mean(value[2] for value in controller_values),
                repetition_ratio=_mean(value[3] for value in controller_values),
                action_magnitude=_mean(value[4] for value in controller_values),
                action_saturation_rate=_mean(value[5] for value in controller_values),
            )
        )
    return tuple(outcomes)


def evaluation_input_sha256(
    records: Sequence[TurnEvaluationRecord],
    design: ExpectedEvaluationDesign,
    spec: EvaluationSpec,
) -> str:
    """Hash canonical, order-independent evaluator inputs."""

    return canonical_sha256(
        {
            "design": design,
            "evaluation_spec": spec,
            "records": tuple(sorted(records, key=record_sort_key)),
        }
    )


__all__ = [
    "aggregate_matched_units",
    "evaluation_input_sha256",
    "expected_turn_keys",
    "record_sort_key",
    "turn_key_sort_key",
    "validate_exact_coverage",
]

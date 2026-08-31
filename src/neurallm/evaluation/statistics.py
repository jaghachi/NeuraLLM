"""Deterministic paired statistics used by the Phase 3 evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from math import floor, isfinite
from random import Random

from neurallm.evaluation.models import (
    BootstrapResult,
    HolmAdjustedPValue,
    PermutationTestResult,
)

_EXACT_SIGN_FLIP_MAX_UNITS = 20
_FLOAT_COMPARISON_TOLERANCE = 1e-15


def _validated_differences(differences: Sequence[float]) -> tuple[float, ...]:
    values = tuple(differences)
    if not values:
        raise ValueError("paired statistics require at least one matched unit")
    if not all(isfinite(value) for value in values):
        raise ValueError("paired differences must be finite")
    return values


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("a quantile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int,
) -> BootstrapResult:
    """Return a percentile CI by resampling matched differences as units."""

    values = _validated_differences(differences)
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between zero and one")
    generator = Random(seed)
    sample_size = len(values)
    resampled_means = sorted(
        sum(values[generator.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(resamples)
    )
    alpha = 1.0 - confidence_level
    return BootstrapResult(
        seed=seed,
        resamples=resamples,
        confidence_level=confidence_level,
        sample_size=sample_size,
        estimate=sum(values) / sample_size,
        lower=_linear_quantile(resampled_means, alpha / 2.0),
        upper=_linear_quantile(resampled_means, 1.0 - alpha / 2.0),
    )


def paired_sign_flip_permutation_test(
    differences: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int,
) -> PermutationTestResult:
    """Return a two-sided paired sign-flip permutation p-value.

    Families of at most twenty units are enumerated exactly.  Larger families
    use a deterministic Monte Carlo stream and an add-one correction.
    """

    values = _validated_differences(differences)
    if resamples < 1:
        raise ValueError("permutation resamples must be positive")
    sample_size = len(values)
    observed = sum(values) / sample_size
    observed_magnitude = abs(observed)

    total_sign_patterns = 2**sample_size
    if sample_size <= _EXACT_SIGN_FLIP_MAX_UNITS and total_sign_patterns <= resamples:
        performed = total_sign_patterns
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=sample_size):
            statistic = sum(value * sign for value, sign in zip(values, signs, strict=True))
            statistic /= sample_size
            if abs(statistic) + _FLOAT_COMPARISON_TOLERANCE >= observed_magnitude:
                extreme += 1
        p_value = extreme / performed
        exact = True
    else:
        generator = Random(seed)
        extreme = 0
        for _ in range(resamples):
            statistic = (
                sum(value * (-1.0 if generator.getrandbits(1) == 0 else 1.0) for value in values)
                / sample_size
            )
            if abs(statistic) + _FLOAT_COMPARISON_TOLERANCE >= observed_magnitude:
                extreme += 1
        performed = resamples
        p_value = (extreme + 1) / (performed + 1)
        exact = False

    return PermutationTestResult(
        seed=seed,
        requested_resamples=resamples,
        performed_permutations=performed,
        exact=exact,
        sample_size=sample_size,
        observed_mean=observed,
        p_value=p_value,
    )


def holm_adjust(
    raw_p_values: Mapping[str, float],
) -> tuple[HolmAdjustedPValue, ...]:
    """Apply the explicitly versioned Holm step-down family correction."""

    if not raw_p_values:
        raise ValueError("Holm correction requires a nonempty comparator family")
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in raw_p_values.values()):
        raise ValueError("raw p-values must be finite values in [0, 1]")

    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_maximum = 0.0
    adjusted: list[HolmAdjustedPValue] = []
    for zero_based_rank, (comparator_id, raw_p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - zero_based_rank) * raw_p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted.append(
            HolmAdjustedPValue(
                comparator_policy_id=comparator_id,
                raw_p_value=raw_p_value,
                adjusted_p_value=running_maximum,
                rank=zero_based_rank + 1,
                family_size=family_size,
            )
        )
    return tuple(sorted(adjusted, key=lambda result: result.comparator_policy_id))


__all__ = [
    "holm_adjust",
    "paired_bootstrap_ci",
    "paired_sign_flip_permutation_test",
]

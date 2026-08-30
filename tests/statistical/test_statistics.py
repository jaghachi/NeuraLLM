"""Golden cases for versioned paired statistical routines."""

from __future__ import annotations

import pytest

from neurallm.evaluation import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_permutation_test,
)


def test_bootstrap_and_exact_sign_flip_golden_cases_record_their_seeds() -> None:
    bootstrap = paired_bootstrap_ci(
        (0.25, 0.25, 0.25, 0.25),
        resamples=100,
        confidence_level=0.95,
        seed=17,
    )
    permutation = paired_sign_flip_permutation_test(
        (1.0, 1.0),
        resamples=999,
        seed=23,
    )

    assert bootstrap.seed == 17
    assert bootstrap.resamples == 100
    assert bootstrap.estimate == pytest.approx(0.25)
    assert bootstrap.lower == pytest.approx(0.25)
    assert bootstrap.upper == pytest.approx(0.25)
    assert permutation.seed == 23
    assert permutation.exact is True
    assert permutation.performed_permutations == 4
    assert permutation.p_value == pytest.approx(0.5)


def test_holm_v1_matches_golden_vector_and_returns_comparator_order() -> None:
    adjusted = holm_adjust({"b": 0.04, "a": 0.01, "c": 0.03})

    assert tuple(result.comparator_policy_id for result in adjusted) == ("a", "b", "c")
    assert tuple(result.adjusted_p_value for result in adjusted) == pytest.approx(
        (0.03, 0.06, 0.06)
    )
    assert {result.method_version for result in adjusted} == {"holm-v1"}


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: paired_bootstrap_ci((), seed=1), "at least one"),
        (lambda: paired_bootstrap_ci((float("nan"),), seed=1), "finite"),
        (lambda: paired_bootstrap_ci((0.1,), resamples=0, seed=1), "positive"),
        (
            lambda: paired_bootstrap_ci((0.1,), confidence_level=1.0, seed=1),
            "between zero and one",
        ),
        (
            lambda: paired_sign_flip_permutation_test((0.1,), resamples=0, seed=1),
            "positive",
        ),
        (lambda: holm_adjust({}), "nonempty"),
        (lambda: holm_adjust({"bad": 1.1}), r"in \[0, 1\]"),
    ],
)
def test_statistical_routines_fail_closed_on_invalid_inputs(
    call: object,
    message: str,
) -> None:
    callable_operation = call
    assert callable(callable_operation)
    with pytest.raises(ValueError, match=message):
        callable_operation()

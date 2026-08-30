"""Tests for deterministic repetition and diversity metrics."""

import pytest

from neurallm.metrics.repetition import (
    distinct_ngram_ratio,
    late_window_repetition_ratio,
    repeated_ngram_ratio,
    repetition_ratio,
    tokenize,
)


def test_tokenization_is_nfkc_casefolded_and_whitespace_based() -> None:
    assert tokenize("ＡLPHA\tBeta  alpha") == ("alpha", "beta", "alpha")


def test_repetition_and_distinct_ratios_have_declared_formulas() -> None:
    tokens = ("a", "b", "a", "b", "a", "b")

    assert repetition_ratio(tokens) == pytest.approx(4 / 6)
    assert repeated_ngram_ratio(tokens, 3) == pytest.approx(2 / 4)
    assert repeated_ngram_ratio(tokens, 4) == pytest.approx(1 / 3)
    assert distinct_ngram_ratio(tokens, 2) == pytest.approx(2 / 5)
    assert distinct_ngram_ratio(tokens, 3) == pytest.approx(2 / 4)


def test_short_outputs_have_finite_zero_ngram_ratios() -> None:
    assert repetition_ratio(()) == 0.0
    assert repeated_ngram_ratio(("one",), 3) == 0.0
    assert distinct_ngram_ratio(("one",), 2) == 0.0
    assert late_window_repetition_ratio(("one",)) == 0.0


def test_late_window_counts_tokens_seen_in_the_prefix() -> None:
    tokens = ("a", "b", "c", "d", "e", "f", "a", "b")

    assert late_window_repetition_ratio(tokens) == 1.0


@pytest.mark.parametrize("function", [repeated_ngram_ratio, distinct_ngram_ratio])
def test_ngram_order_must_be_positive(function: object) -> None:
    with pytest.raises(ValueError, match="positive"):
        function(("token",), 0)  # type: ignore[operator]

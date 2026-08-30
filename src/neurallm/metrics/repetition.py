"""Pure deterministic repetition and diversity metrics."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

TOKENIZATION_VERSION = "unicode-nfkc-casefold-whitespace-v1"


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize and split on Unicode whitespace without external models."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(normalized.split())


def repetition_ratio(tokens: Sequence[str]) -> float:
    """Fraction of token occurrences beyond their first occurrence."""

    if not tokens:
        return 0.0
    return (len(tokens) - len(set(tokens))) / len(tokens)


def repeated_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    """Fraction of n-gram occurrences beyond their first occurrence."""

    if n < 1:
        raise ValueError("n must be positive")
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    ngrams = tuple(tuple(tokens[index : index + n]) for index in range(total))
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def distinct_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    """Fraction of distinct n-grams among all observed n-grams."""

    if n < 1:
        raise ValueError("n must be positive")
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    ngrams = tuple(tuple(tokens[index : index + n]) for index in range(total))
    return len(set(ngrams)) / len(ngrams)


def late_window_repetition_ratio(tokens: Sequence[str]) -> float:
    """Fraction of final-quarter tokens already seen in the response."""

    if len(tokens) < 2:
        return 0.0
    window_size = max(1, len(tokens) // 4)
    start = len(tokens) - window_size
    seen = set(tokens[:start])
    repeated = 0
    for token in tokens[start:]:
        if token in seen:
            repeated += 1
        seen.add(token)
    return repeated / window_size


__all__ = [
    "TOKENIZATION_VERSION",
    "distinct_ngram_ratio",
    "late_window_repetition_ratio",
    "repeated_ngram_ratio",
    "repetition_ratio",
    "tokenize",
]

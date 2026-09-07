"""Canonical Phase 2 response-metric computation."""

from __future__ import annotations

from collections.abc import Mapping

from neurallm.domain.models import MetricValue, ResponseMetrics
from neurallm.domain.serialization import canonical_sha256
from neurallm.metrics.answer_channel import final_answer_text
from neurallm.metrics.base import MetricContext
from neurallm.metrics.repetition import (
    TOKENIZATION_VERSION,
    distinct_ngram_ratio,
    late_window_repetition_ratio,
    repeated_ngram_ratio,
    repetition_ratio,
    tokenize,
)
from neurallm.metrics.validators import validate_response

METRIC_VERSIONS = {
    "task_score": "validator-v1",
    "instruction_adherence": "validator-v1",
    "response_length_tokens": TOKENIZATION_VERSION,
    "repetition_ratio": f"token-repetition-{TOKENIZATION_VERSION}",
    "repeated_3_gram_ratio": f"repeated-3gram-{TOKENIZATION_VERSION}",
    "repeated_4_gram_ratio": f"repeated-4gram-{TOKENIZATION_VERSION}",
    "distinct_2": f"distinct-2gram-{TOKENIZATION_VERSION}",
    "distinct_3": f"distinct-3gram-{TOKENIZATION_VERSION}",
    "late_window_repetition_ratio": f"late-quarter-{TOKENIZATION_VERSION}",
    "format_validity": "validator-v1",
    "semantic_similarity": "semantic-unavailable-v1",
}

FINAL_ANSWER_METRIC_VERSIONS = {
    name: (
        version
        if name == "semantic_similarity"
        else "validator-final-answer-v2"
        if version == "validator-v1"
        else f"final-answer-v2-{version}"
    )
    for name, version in METRIC_VERSIONS.items()
}


def validate_metric_versions(metric_versions: Mapping[str, str]) -> None:
    """Require one complete implemented version set; mixed sets are invalid."""

    if dict(metric_versions) not in (METRIC_VERSIONS, FINAL_ANSWER_METRIC_VERSIONS):
        raise ValueError("configured metric versions do not match a supported implementation")


def _input_hash(
    context: MetricContext, metric_name: str, metric_versions: Mapping[str, str]
) -> str:
    return canonical_sha256(
        {
            "metric_name": metric_name,
            "metric_version": metric_versions[metric_name],
            "prompt_case_id": context.prompt_case_id,
            "prompt_family": context.prompt_family,
            "prompt": context.prompt,
            "response_text": context.response_text,
            "validator": context.validator,
        }
    )


def _float_metric(
    context: MetricContext, name: str, value: float, metric_versions: Mapping[str, str]
) -> MetricValue[float]:
    return MetricValue[float](
        value=value,
        availability=True,
        metric_version=metric_versions[name],
        input_hash=_input_hash(context, name, metric_versions),
    )


def compute_response_metrics(
    context: MetricContext,
    *,
    metric_versions: Mapping[str, str] | None = None,
) -> ResponseMetrics:
    """Compute the declared tuple, defaulting to frozen legacy v1 semantics.

    V2 computes all available response-derived values on the final answer.
    Missing/malformed answer framing yields zero-length input and zero validator
    scores, not unavailable metrics or a transport error. Hashes always bind the
    complete raw response so reasoning content remains auditable.
    """

    if not isinstance(context, MetricContext):
        raise TypeError("context must be a MetricContext")
    versions = METRIC_VERSIONS if metric_versions is None else metric_versions
    validate_metric_versions(versions)
    response_text = context.response_text
    if dict(versions) == FINAL_ANSWER_METRIC_VERSIONS:
        response_text = final_answer_text(response_text) or ""
    validation = validate_response(response_text, context.validator)
    tokens = tokenize(response_text)

    return ResponseMetrics(
        task_score=_float_metric(context, "task_score", validation.task_score, versions),
        instruction_adherence=_float_metric(
            context,
            "instruction_adherence",
            validation.instruction_adherence,
            versions,
        ),
        response_length_tokens=MetricValue[int](
            value=len(tokens),
            availability=True,
            metric_version=versions["response_length_tokens"],
            input_hash=_input_hash(context, "response_length_tokens", versions),
        ),
        repetition_ratio=_float_metric(
            context,
            "repetition_ratio",
            repetition_ratio(tokens),
            versions,
        ),
        repeated_3_gram_ratio=_float_metric(
            context,
            "repeated_3_gram_ratio",
            repeated_ngram_ratio(tokens, 3),
            versions,
        ),
        repeated_4_gram_ratio=_float_metric(
            context,
            "repeated_4_gram_ratio",
            repeated_ngram_ratio(tokens, 4),
            versions,
        ),
        distinct_2=_float_metric(context, "distinct_2", distinct_ngram_ratio(tokens, 2), versions),
        distinct_3=_float_metric(context, "distinct_3", distinct_ngram_ratio(tokens, 3), versions),
        late_window_repetition_ratio=_float_metric(
            context,
            "late_window_repetition_ratio",
            late_window_repetition_ratio(tokens),
            versions,
        ),
        format_validity=_float_metric(
            context,
            "format_validity",
            validation.format_validity,
            versions,
        ),
        semantic_similarity=MetricValue[float](
            value=None,
            availability=False,
            metric_version=versions["semantic_similarity"],
            input_hash=_input_hash(context, "semantic_similarity", versions),
        ),
    )


__all__ = [
    "FINAL_ANSWER_METRIC_VERSIONS",
    "METRIC_VERSIONS",
    "compute_response_metrics",
    "validate_metric_versions",
]

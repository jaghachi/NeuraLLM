"""Canonical Phase 2 response-metric computation."""

from __future__ import annotations

from neurallm.domain.models import MetricValue, ResponseMetrics
from neurallm.domain.serialization import canonical_sha256
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


def _input_hash(context: MetricContext, metric_name: str) -> str:
    return canonical_sha256(
        {
            "metric_name": metric_name,
            "metric_version": METRIC_VERSIONS[metric_name],
            "prompt_case_id": context.prompt_case_id,
            "prompt_family": context.prompt_family,
            "prompt": context.prompt,
            "response_text": context.response_text,
            "validator": context.validator,
        }
    )


def _float_metric(context: MetricContext, name: str, value: float) -> MetricValue[float]:
    return MetricValue[float](
        value=value,
        availability=True,
        metric_version=METRIC_VERSIONS[name],
        input_hash=_input_hash(context, name),
    )


def compute_response_metrics(context: MetricContext) -> ResponseMetrics:
    """Compute the complete deterministic Phase 2 metric tuple."""

    if not isinstance(context, MetricContext):
        raise TypeError("context must be a MetricContext")
    validation = validate_response(context.response_text, context.validator)
    tokens = tokenize(context.response_text)

    return ResponseMetrics(
        task_score=_float_metric(context, "task_score", validation.task_score),
        instruction_adherence=_float_metric(
            context,
            "instruction_adherence",
            validation.instruction_adherence,
        ),
        response_length_tokens=MetricValue[int](
            value=len(tokens),
            availability=True,
            metric_version=METRIC_VERSIONS["response_length_tokens"],
            input_hash=_input_hash(context, "response_length_tokens"),
        ),
        repetition_ratio=_float_metric(
            context,
            "repetition_ratio",
            repetition_ratio(tokens),
        ),
        repeated_3_gram_ratio=_float_metric(
            context,
            "repeated_3_gram_ratio",
            repeated_ngram_ratio(tokens, 3),
        ),
        repeated_4_gram_ratio=_float_metric(
            context,
            "repeated_4_gram_ratio",
            repeated_ngram_ratio(tokens, 4),
        ),
        distinct_2=_float_metric(context, "distinct_2", distinct_ngram_ratio(tokens, 2)),
        distinct_3=_float_metric(context, "distinct_3", distinct_ngram_ratio(tokens, 3)),
        late_window_repetition_ratio=_float_metric(
            context,
            "late_window_repetition_ratio",
            late_window_repetition_ratio(tokens),
        ),
        format_validity=_float_metric(
            context,
            "format_validity",
            validation.format_validity,
        ),
        semantic_similarity=MetricValue[float](
            value=None,
            availability=False,
            metric_version=METRIC_VERSIONS["semantic_similarity"],
            input_hash=_input_hash(context, "semantic_similarity"),
        ),
    )


__all__ = ["METRIC_VERSIONS", "compute_response_metrics"]

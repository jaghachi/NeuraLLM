"""Tests for the canonical Phase 2 metric tuple and plugin registry."""

from dataclasses import dataclass

import pytest

from neurallm.domain.models import MetricValue
from neurallm.metrics import (
    METRIC_VERSIONS,
    MetricContext,
    MetricRegistry,
    ValidatorSpec,
    compute_response_metrics,
)


def make_context(response_text: str = "alpha beta alpha beta alpha beta") -> MetricContext:
    return MetricContext(
        prompt_case_id="case-1",
        prompt_family="constrained",
        prompt="Include alpha and beta.",
        response_text=response_text,
        validator=ValidatorSpec(
            kind="contains_all",
            required_terms=("alpha", "beta"),
        ),
    )


def test_complete_metric_tuple_is_deterministic_and_provenance_bearing() -> None:
    context = make_context()

    first = compute_response_metrics(context)
    second = compute_response_metrics(context)

    assert first == second
    assert first.task_score.value == 1.0
    assert first.response_length_tokens.value == 6
    assert first.repeated_3_gram_ratio.value == pytest.approx(0.5)
    assert first.repeated_4_gram_ratio.value == pytest.approx(1 / 3)
    assert first.semantic_similarity.value is None
    assert first.semantic_similarity.availability is False
    dumped = first.model_dump(mode="json")
    for metric_name, version in METRIC_VERSIONS.items():
        assert dumped[metric_name]["metric_version"] == version
        assert len(dumped[metric_name]["input_hash"]) == 64


def test_metric_hash_changes_when_response_changes() -> None:
    first = compute_response_metrics(make_context("alpha beta"))
    second = compute_response_metrics(make_context("alpha beta beta"))

    assert first.task_score.input_hash != second.task_score.input_hash
    assert first.repetition_ratio.input_hash != second.repetition_ratio.input_hash


@dataclass(frozen=True)
class ConstantPlugin:
    metric_name: str
    metric_version: str = "constant-v1"

    def compute(self, context: MetricContext) -> MetricValue[float]:
        return MetricValue[float](
            value=0.5,
            availability=True,
            metric_version=self.metric_version,
            input_hash="0" * 64,
        )


def test_metric_registry_is_ordered_and_rejects_duplicate_names() -> None:
    registry = MetricRegistry((ConstantPlugin("z"), ConstantPlugin("a")))

    assert tuple(registry.plugins) == ("a", "z")
    assert tuple(registry.compute(make_context())) == ("a", "z")
    with pytest.raises(ValueError, match="duplicate"):
        MetricRegistry((ConstantPlugin("a"), ConstantPlugin("a")))

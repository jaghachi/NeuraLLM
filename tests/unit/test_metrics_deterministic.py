"""Tests for the canonical Phase 2 metric tuple and plugin registry."""

from dataclasses import dataclass

import pytest

from neurallm.domain.models import MetricValue
from neurallm.domain.serialization import canonical_sha256
from neurallm.metrics import (
    FINAL_ANSWER_METRIC_VERSIONS,
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


def test_legacy_metric_tuple_hash_is_preserved_including_reasoning_text() -> None:
    context = make_context("<think>alpha beta alpha</think>\nfinal")
    legacy = compute_response_metrics(context)
    assert legacy == compute_response_metrics(context, metric_versions=METRIC_VERSIONS)
    assert canonical_sha256(legacy) == (
        "a67395bc389aee8cdad2c6402a13e85b1e6a22f2edaa9ce5165c022e83eca2af"
    )
    assert legacy.task_score.value == 1.0


@pytest.mark.parametrize("response", ("<think>alpha beta", "<think>alpha beta</think>"))
def test_v2_unfinished_or_absent_final_answer_cannot_score_reasoning(response: str) -> None:
    metrics = compute_response_metrics(
        make_context(response), metric_versions=FINAL_ANSWER_METRIC_VERSIONS
    )
    assert metrics.task_score.value == 0.0
    assert metrics.instruction_adherence.value == 0.0
    assert metrics.format_validity.value == 0.0
    assert metrics.response_length_tokens.value == 0
    assert metrics.repetition_ratio.value == 0.0
    assert metrics.semantic_similarity.availability is False


def test_v2_keywords_and_repetition_in_reasoning_cannot_score_the_answer() -> None:
    context = make_context("<think>alpha beta alpha beta alpha beta</think>\nfinal answer")
    metrics = compute_response_metrics(context, metric_versions=FINAL_ANSWER_METRIC_VERSIONS)
    assert metrics.task_score.value == 0.0
    assert metrics.instruction_adherence.value == 0.0
    assert metrics.format_validity.value == 1.0
    assert metrics.response_length_tokens.value == 2
    assert metrics.repetition_ratio.value == 0.0
    assert metrics.distinct_2.value == 1.0
    for name, value in metrics.model_dump(mode="json").items():
        assert value["metric_version"] == FINAL_ANSWER_METRIC_VERSIONS[name]


@pytest.mark.parametrize(
    ("answer", "validator", "score"),
    (
        ("RECOVERED", ValidatorSpec(kind="exact_match", expected_text="RECOVERED"), 1.0),
        ("RECOVERED\n", ValidatorSpec(kind="exact_match", expected_text="RECOVERED"), 0.0),
        ('{"answer": 1}', ValidatorSpec(kind="json_object", required_json_keys=("answer",)), 1.0),
        ("answer", ValidatorSpec(kind="non_empty"), 1.0),
    ),
)
def test_v2_validators_use_the_final_answer_without_trailing_whitespace_repair(
    answer: str, validator: ValidatorSpec, score: float
) -> None:
    context = make_context().model_copy(
        update={"response_text": f"<think>reason</think>\n\n{answer}", "validator": validator}
    )
    metrics = compute_response_metrics(context, metric_versions=FINAL_ANSWER_METRIC_VERSIONS)
    assert metrics.task_score.value == score


def test_v2_provenance_still_binds_raw_reasoning_and_preserves_context() -> None:
    first = make_context("<think>one reason</think>alpha beta")
    second = make_context("<think>another reason</think>alpha beta")
    first_metrics = compute_response_metrics(first, metric_versions=FINAL_ANSWER_METRIC_VERSIONS)
    second_metrics = compute_response_metrics(second, metric_versions=FINAL_ANSWER_METRIC_VERSIONS)
    assert first_metrics.task_score.value == second_metrics.task_score.value == 1.0
    assert first_metrics.task_score.input_hash != second_metrics.task_score.input_hash
    assert first.response_text == "<think>one reason</think>alpha beta"


@pytest.mark.parametrize(
    "versions",
    (
        {},
        {**METRIC_VERSIONS, "task_score": "unknown"},
        {**METRIC_VERSIONS, "task_score": FINAL_ANSWER_METRIC_VERSIONS["task_score"]},
        {**FINAL_ANSWER_METRIC_VERSIONS, "extra": "v1"},
    ),
)
def test_computation_rejects_unknown_incomplete_or_mixed_metric_versions(
    versions: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="metric versions"):
        compute_response_metrics(make_context(), metric_versions=versions)


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

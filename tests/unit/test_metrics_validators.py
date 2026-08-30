"""Tests for strict deterministic prompt validators."""

import pytest
from pydantic import ValidationError

from neurallm.metrics.validators import ValidatorSpec, validate_response


def test_contains_all_scores_fraction_without_case_sensitivity() -> None:
    spec = ValidatorSpec(
        kind="contains_all",
        required_terms=("Alpha", "beta", "gamma"),
    )

    result = validate_response("ALPHA and beta", spec)

    assert result.task_score == pytest.approx(2 / 3)
    assert result.instruction_adherence == pytest.approx(2 / 3)
    assert result.format_validity == 1.0


def test_exact_match_honors_case_sensitive_setting() -> None:
    insensitive = ValidatorSpec(kind="exact_match", expected_text="Answer")
    sensitive = ValidatorSpec(
        kind="exact_match",
        expected_text="Answer",
        case_sensitive=True,
    )

    assert validate_response("answer", insensitive).task_score == 1.0
    assert validate_response("answer", sensitive).task_score == 0.0


def test_json_object_scores_required_keys_and_rejects_malformed_json() -> None:
    spec = ValidatorSpec(
        kind="json_object",
        required_json_keys=("answer", "reason"),
    )

    partial = validate_response('{"answer": 42}', spec)
    malformed = validate_response('{"answer": NaN}', spec)
    duplicate = validate_response('{"answer": 1, "answer": 2}', spec)

    assert partial.task_score == 0.5
    assert partial.format_validity == 1.0
    assert malformed.task_score == 0.0
    assert malformed.format_validity == 0.0
    assert duplicate.format_validity == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "contains_all", "required_terms": ()},
        {"kind": "exact_match", "expected_text": " "},
        {"kind": "json_object", "required_json_keys": ()},
        {"kind": "non_empty", "required_terms": ("unexpected",)},
        {"kind": "contains_all", "required_terms": ("x", "x")},
    ],
)
def test_malformed_validator_configuration_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ValidatorSpec.model_validate(payload)


def test_non_string_response_is_rejected() -> None:
    spec = ValidatorSpec(kind="non_empty")

    with pytest.raises(TypeError, match="response_text"):
        validate_response(42, spec)  # type: ignore[arg-type]

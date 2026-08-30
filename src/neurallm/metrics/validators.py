"""Strict deterministic objective validators for prompt cases."""

from __future__ import annotations

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ValidatorSpec(_StrictFrozenModel):
    """Configuration for one deterministic prompt objective validator."""

    kind: Literal["non_empty", "contains_all", "exact_match", "json_object"]
    expected_text: str | None = None
    required_terms: tuple[str, ...] = ()
    required_json_keys: tuple[str, ...] = ()
    case_sensitive: bool = False

    @field_validator("required_terms", "required_json_keys", mode="before")
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("required_terms", "required_json_keys")
    @classmethod
    def _validate_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("validator values must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("validator values must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _validate_kind_configuration(self) -> Self:
        if self.kind == "non_empty":
            if self.expected_text is not None or self.required_terms or self.required_json_keys:
                raise ValueError("non_empty validator does not accept objective parameters")
        elif self.kind == "contains_all":
            if not self.required_terms:
                raise ValueError("contains_all validator requires required_terms")
            if self.expected_text is not None or self.required_json_keys:
                raise ValueError("contains_all validator received incompatible parameters")
        elif self.kind == "exact_match":
            if self.expected_text is None or not self.expected_text.strip():
                raise ValueError("exact_match validator requires non-blank expected_text")
            if self.required_terms or self.required_json_keys:
                raise ValueError("exact_match validator received incompatible parameters")
        elif self.kind == "json_object":
            if not self.required_json_keys:
                raise ValueError("json_object validator requires required_json_keys")
            if self.expected_text is not None or self.required_terms:
                raise ValueError("json_object validator received incompatible parameters")
        return self


class ValidationResult(_StrictFrozenModel):
    """Normalized output of a deterministic objective validator."""

    task_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    instruction_adherence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    format_validity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def validate_response(response_text: str, spec: ValidatorSpec) -> ValidationResult:
    """Validate one response without model calls or heuristic repair."""

    if not isinstance(response_text, str):
        raise TypeError("response_text must be a string")
    if not isinstance(spec, ValidatorSpec):
        raise TypeError("spec must be a ValidatorSpec")

    non_empty = bool(response_text.strip())
    if spec.kind == "non_empty":
        score = float(non_empty)
        return ValidationResult(
            task_score=score,
            instruction_adherence=score,
            format_validity=score,
        )

    if spec.kind == "contains_all":
        candidate = response_text if spec.case_sensitive else response_text.casefold()
        terms = spec.required_terms
        matched = sum(
            1 for term in terms if (term if spec.case_sensitive else term.casefold()) in candidate
        )
        score = matched / len(terms)
        return ValidationResult(
            task_score=score,
            instruction_adherence=score,
            format_validity=float(non_empty),
        )

    if spec.kind == "exact_match":
        assert spec.expected_text is not None
        candidate = response_text if spec.case_sensitive else response_text.casefold()
        expected = spec.expected_text if spec.case_sensitive else spec.expected_text.casefold()
        score = float(candidate == expected)
        return ValidationResult(
            task_score=score,
            instruction_adherence=score,
            format_validity=float(non_empty),
        )

    try:
        parsed = json.loads(
            response_text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return ValidationResult(
            task_score=0.0,
            instruction_adherence=0.0,
            format_validity=0.0,
        )
    if not isinstance(parsed, dict):
        return ValidationResult(
            task_score=0.0,
            instruction_adherence=0.0,
            format_validity=0.0,
        )
    matched_keys = sum(key in parsed for key in spec.required_json_keys)
    score = matched_keys / len(spec.required_json_keys)
    return ValidationResult(
        task_score=score,
        instruction_adherence=score,
        format_validity=1.0,
    )


__all__ = ["ValidationResult", "ValidatorSpec", "validate_response"]

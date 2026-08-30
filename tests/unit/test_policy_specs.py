"""Tests for strict discriminated Phase 3 policy configuration."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from neurallm.control import (
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    PolicySpec,
    RandomMatchedPolicySpec,
)


def test_policy_spec_union_is_discriminated_by_kind() -> None:
    adapter = TypeAdapter(PolicySpec)

    assert adapter.validate_python({"kind": "best_static"}) == BestStaticPolicySpec()
    assert adapter.validate_python({"kind": "random_matched"}) == RandomMatchedPolicySpec()
    assert adapter.validate_python({"kind": "heuristic_adaptive"}) == HeuristicAdaptivePolicySpec()

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python({"kind": "unknown"})


def test_spec_defaults_bind_algorithm_state_and_history_contracts() -> None:
    assert BestStaticPolicySpec().model_dump() == {
        "kind": "best_static",
        "policy_id": "best_static",
        "implementation_version": "best-static-v1",
        "state_schema_version": "stateless-v1",
        "history_access": "none",
    }
    assert RandomMatchedPolicySpec().model_dump() == {
        "kind": "random_matched",
        "policy_id": "random_matched",
        "implementation_version": "random-matched-sha256-v1",
        "state_schema_version": "random-matched-state-v1",
        "history_access": "none",
    }
    heuristic = HeuristicAdaptivePolicySpec()
    assert heuristic.history_access == "own_previous_response"
    assert heuristic.minimum_response_length_tokens < heuristic.maximum_response_length_tokens
    assert heuristic.repetition_reaction_fraction == 0.75
    assert heuristic.adherence_reaction_fraction == 0.50
    assert heuristic.length_reaction_fraction == 0.25
    assert heuristic.clean_decay_fraction == 0.50


def test_specs_are_strict_frozen_and_extra_forbid() -> None:
    adapter = TypeAdapter(PolicySpec)
    spec = HeuristicAdaptivePolicySpec()

    with pytest.raises(ValidationError, match="frozen"):
        spec.high_repetition_threshold = 0.5

    with pytest.raises(ValidationError, match="extra_forbidden"):
        adapter.validate_python({"kind": "best_static", "mode": "legacy"})

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "heuristic_adaptive",
                "minimum_response_length_tokens": "8",
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "minimum_response_length_tokens": 8,
            "maximum_response_length_tokens": 8,
        },
        {
            "minimum_response_length_tokens": 9,
            "maximum_response_length_tokens": 8,
        },
        {"high_repetition_threshold": 1.0},
        {"minimum_instruction_adherence": 0.0},
        {"repetition_reaction_fraction": 0.0},
        {"clean_decay_fraction": 1.1},
    ],
)
def test_invalid_heuristic_configuration_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        HeuristicAdaptivePolicySpec.model_validate(payload)

"""Unit and property tests for auditable action application."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from neurallm.control import ActionApplication, apply_action
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    DecodingBounds,
    DecodingParameters,
)


def _base_parameters(
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    presence_penalty: float = 0.0,
    max_tokens: int = 64,
    seed: int = 11,
) -> DecodingParameters:
    return DecodingParameters(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )


def _action(
    *,
    temperature_delta: float = 0.0,
    top_p_delta: float = 0.0,
    top_k_delta: int = 0,
    presence_penalty_delta: float = 0.0,
) -> ControllerAction:
    return ControllerAction(
        temperature_delta=temperature_delta,
        top_p_delta=top_p_delta,
        top_k_delta=top_k_delta,
        presence_penalty_delta=presence_penalty_delta,
    )


def test_apply_action_transmits_all_four_controlled_deltas() -> None:
    raw = _action(
        temperature_delta=0.05,
        top_p_delta=-0.04,
        top_k_delta=7,
        presence_penalty_delta=0.15,
    )

    applied = apply_action(
        _base_parameters(),
        raw,
        ActionBounds(),
        DecodingBounds(),
    )

    assert applied.raw_action is raw
    assert applied.step_clamped_action == raw
    assert applied.final_decoding_parameters == _base_parameters(
        temperature=0.75,
        top_p=0.86,
        top_k=47,
        presence_penalty=0.15,
    )
    assert applied.saturation.any_saturation is False


def test_step_clamp_is_recorded_separately_from_legal_clamp() -> None:
    applied = apply_action(
        _base_parameters(),
        _action(
            temperature_delta=0.5,
            top_p_delta=-0.5,
            top_k_delta=50,
            presence_penalty_delta=-0.5,
        ),
        ActionBounds(),
        DecodingBounds(),
    )

    assert applied.step_clamped_action == _action(
        temperature_delta=0.1,
        top_p_delta=-0.05,
        top_k_delta=10,
        presence_penalty_delta=-0.2,
    )
    assert applied.final_decoding_parameters.temperature == pytest.approx(0.8)
    assert applied.final_decoding_parameters.top_p == pytest.approx(0.85)
    assert applied.final_decoding_parameters.top_k == 50
    assert applied.final_decoding_parameters.presence_penalty == pytest.approx(-0.2)
    assert applied.final_decoding_parameters.max_tokens == 64
    assert applied.final_decoding_parameters.seed == 11
    for indicator in (
        applied.saturation.temperature,
        applied.saturation.top_p,
        applied.saturation.top_k,
        applied.saturation.presence_penalty,
    ):
        assert indicator.step_clamped is True
        assert indicator.legal_clamped is False


def test_final_legal_clamp_preserves_fixed_budget_and_seed() -> None:
    applied = apply_action(
        _base_parameters(
            temperature=1.98,
            top_p=0.02,
            top_k=198,
            presence_penalty=1.95,
            max_tokens=257,
            seed=-19,
        ),
        _action(
            temperature_delta=0.1,
            top_p_delta=-0.05,
            top_k_delta=10,
            presence_penalty_delta=0.2,
        ),
        ActionBounds(),
        DecodingBounds(),
    )

    assert applied.final_decoding_parameters == _base_parameters(
        temperature=2.0,
        top_p=0.01,
        top_k=200,
        presence_penalty=2.0,
        max_tokens=257,
        seed=-19,
    )
    for indicator in (
        applied.saturation.temperature,
        applied.saturation.top_p,
        applied.saturation.top_k,
        applied.saturation.presence_penalty,
    ):
        assert indicator.step_clamped is False
        assert indicator.legal_clamped is True


def test_action_application_is_strict_frozen_and_json_serializable() -> None:
    applied = apply_action(
        _base_parameters(),
        _action(),
        ActionBounds(),
        DecodingBounds(),
    )

    restored = ActionApplication.model_validate_json(applied.model_dump_json())

    assert restored == applied
    with pytest.raises(ValidationError, match="frozen"):
        applied.raw_action = _action(temperature_delta=0.1)


@given(
    temperature=st.floats(
        min_value=0.0001,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_p=st.floats(
        min_value=0.0001,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_k=st.integers(min_value=0, max_value=1_000),
    presence_penalty=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_tokens=st.integers(min_value=1, max_value=16_384),
    seed=st.integers(min_value=-(2**63), max_value=2**63 - 1),
    temperature_delta=st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_p_delta=st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_k_delta=st.integers(min_value=-1_000, max_value=1_000),
    presence_penalty_delta=st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_apply_action_is_finite_bounded_and_preserves_fixed_fields(
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    max_tokens: int,
    seed: int,
    temperature_delta: float,
    top_p_delta: float,
    top_k_delta: int,
    presence_penalty_delta: float,
) -> None:
    base = _base_parameters(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )
    raw = _action(
        temperature_delta=temperature_delta,
        top_p_delta=top_p_delta,
        top_k_delta=top_k_delta,
        presence_penalty_delta=presence_penalty_delta,
    )
    action_bounds = ActionBounds()
    decoding_bounds = DecodingBounds()

    applied = apply_action(base, raw, action_bounds, decoding_bounds)
    step = applied.step_clamped_action
    final = applied.final_decoding_parameters

    assert action_bounds.contains(step)
    assert decoding_bounds.temperature[0] <= final.temperature <= decoding_bounds.temperature[1]
    assert decoding_bounds.top_p[0] <= final.top_p <= decoding_bounds.top_p[1]
    assert decoding_bounds.top_k[0] <= final.top_k <= decoding_bounds.top_k[1]
    assert (
        decoding_bounds.presence_penalty[0]
        <= final.presence_penalty
        <= decoding_bounds.presence_penalty[1]
    )
    assert final.max_tokens == max_tokens
    assert final.seed == seed
    assert math.isfinite(final.temperature)
    assert math.isfinite(final.top_p)
    assert math.isfinite(final.presence_penalty)

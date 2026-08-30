"""Property-based checks for domain serialization and identifiers."""

from __future__ import annotations

import json
import math

from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from neurallm.domain.identifiers import condition_id
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    DecodingParameters,
    ExperimentCondition,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256

ZERO_HASH = "0" * 64

finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
json_scalars = st.none() | st.booleans() | st.integers() | finite_floats | st.text()
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5)
    ),
    max_leaves=20,
)


@given(json_values)
def test_canonical_serialization_round_trips_json_values(value: object) -> None:
    serialized = canonical_json(value)
    round_tripped = json.loads(serialized)

    assert canonical_json(round_tripped) == serialized
    assert canonical_sha256(round_tripped) == canonical_sha256(value)


@given(
    temperature=st.floats(
        min_value=math.nextafter(0.0, math.inf),
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_p=st.floats(
        min_value=math.nextafter(0.0, math.inf),
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_k=st.integers(min_value=0),
    presence_penalty=finite_floats,
    max_tokens=st.integers(min_value=1),
    seed=st.integers(),
)
def test_finite_decoding_parameters_remain_finite_and_round_trip(
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    max_tokens: int,
    seed: int,
) -> None:
    parameters = DecodingParameters(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
    )

    restored = DecodingParameters.model_validate_json(parameters.model_dump_json())
    assert restored == parameters
    assert math.isfinite(restored.temperature)
    assert math.isfinite(restored.top_p)
    assert math.isfinite(restored.presence_penalty)


@given(
    temperature_delta=st.floats(
        min_value=-0.10,
        max_value=0.10,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_p_delta=st.floats(
        min_value=-0.05,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    top_k_delta=st.integers(min_value=-10, max_value=10),
    presence_penalty_delta=st.floats(
        min_value=-0.20,
        max_value=0.20,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_controller_action_stays_within_shared_bounds(
    temperature_delta: float,
    top_p_delta: float,
    top_k_delta: int,
    presence_penalty_delta: float,
) -> None:
    action = ControllerAction(
        temperature_delta=temperature_delta,
        top_p_delta=top_p_delta,
        top_k_delta=top_k_delta,
        presence_penalty_delta=presence_penalty_delta,
    )

    assert ActionBounds().require(action) is action
    assert -0.10 <= action.temperature_delta <= 0.10
    assert -0.05 <= action.top_p_delta <= 0.05
    assert -10 <= action.top_k_delta <= 10
    assert -0.20 <= action.presence_penalty_delta <= 0.20


@given(
    experiment_id=st.text(min_size=1).filter(str.strip),
    dataset_version=st.text(min_size=1).filter(str.strip),
    prompt_sequence_id=st.text(min_size=1).filter(str.strip),
    turn_index=st.integers(min_value=0),
    policy_id=st.text(min_size=1).filter(str.strip),
    model_seed=st.integers(),
    controller_seed=st.integers(),
    base_decoding_profile_id=st.text(min_size=1).filter(str.strip),
)
def test_condition_identifiers_are_deterministic(
    experiment_id: str,
    dataset_version: str,
    prompt_sequence_id: str,
    turn_index: int,
    policy_id: str,
    model_seed: int,
    controller_seed: int,
    base_decoding_profile_id: str,
) -> None:
    condition = ExperimentCondition(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        prompt_sequence_id=prompt_sequence_id,
        turn_index=turn_index,
        policy_id=policy_id,
        model_seed=model_seed,
        controller_seed=controller_seed,
        provider_identity_id=ZERO_HASH,
        base_decoding_profile_id=base_decoding_profile_id,
    )
    restored = ExperimentCondition.model_validate_json(condition.model_dump_json())

    assert condition_id(condition) == condition_id(restored)
    assert len(condition_id(condition)) == 64


@given(nonfinite=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_nonfinite_controller_actions_always_fail(nonfinite: float) -> None:
    try:
        ControllerAction(
            temperature_delta=nonfinite,
            top_p_delta=0.0,
            top_k_delta=0,
            presence_penalty_delta=0.0,
        )
    except ValidationError:
        return
    raise AssertionError("non-finite controller action was accepted")

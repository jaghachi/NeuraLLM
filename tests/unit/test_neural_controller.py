"""Causal, deterministic, and bounded simulated-neural controller tests."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from neurallm.control import (
    ActionDecoder,
    HeuristicAdaptivePolicy,
    NeuralMatchedHistoryStateResetPolicy,
    NeuralPersistentPolicy,
    NeuralPolicyState,
    NeuralSubstrate,
    NeuralSubstrateState,
    ObservationEncoder,
    PolicyContext,
)
from neurallm.control.neural.encoder import EncodedObservation
from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    CountMetricValue,
    DecodingParameters,
    ExperimentCondition,
    PromptFeatures,
    ResponseMetrics,
    UnitIntervalMetricValue,
)
from neurallm.domain.serialization import canonical_json

_INPUT_HASH = "0" * 64
_PROVIDER_ID = "1" * 64
_ZERO_ACTION = ControllerAction(
    temperature_delta=0.0,
    top_p_delta=0.0,
    top_k_delta=0,
    presence_penalty_delta=0.0,
)


def _context(policy_id: str, controller_seed: int = 23) -> PolicyContext:
    return PolicyContext(
        condition=ExperimentCondition(
            experiment_id="phase4-neural-tests",
            dataset_version="development-v1",
            prompt_sequence_id="sequence-a",
            turn_index=0,
            policy_id=policy_id,
            model_seed=17,
            controller_seed=controller_seed,
            provider_identity_id=_PROVIDER_ID,
            base_decoding_profile_id="base-v1",
        ),
        initial_decoding_parameters=DecodingParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=128,
            seed=17,
        ),
        action_bounds=ActionBounds(),
    )


def _unit(value: float | None) -> UnitIntervalMetricValue:
    return UnitIntervalMetricValue(
        value=value,
        availability=value is not None,
        metric_version="phase4-test-v1",
        input_hash=_INPUT_HASH,
    )


def _count(value: int | None) -> CountMetricValue:
    return CountMetricValue(
        value=value,
        availability=value is not None,
        metric_version="phase4-test-v1",
        input_hash=_INPUT_HASH,
    )


def _metrics(
    *,
    task_score: float | None = 1.0,
    adherence: float | None = 1.0,
    repetition: float | None = 0.0,
    response_length: int | None = 64,
) -> ResponseMetrics:
    return ResponseMetrics(
        task_score=_unit(task_score),
        instruction_adherence=_unit(adherence),
        response_length_tokens=_count(response_length),
        repetition_ratio=_unit(repetition),
        repeated_3_gram_ratio=_unit(0.0),
        repeated_4_gram_ratio=_unit(0.0),
        distinct_2=_unit(1.0),
        distinct_3=_unit(1.0),
        late_window_repetition_ratio=_unit(0.0),
        format_validity=_unit(1.0),
        semantic_similarity=_unit(None),
    )


def _observation(
    turn_index: int,
    metrics: ResponseMetrics | None = None,
    *,
    prompt_features: dict[str, float] | None = None,
) -> ControllerObservation:
    return ControllerObservation(
        turn_index=turn_index,
        prompt_family="designed-stimulus",
        current_prompt_features=PromptFeatures(
            prompt_features or {"constraint_count": 2.0, "target_length": 64.0}
        ),
        previous_response_metrics=metrics,
        has_previous_response=metrics is not None,
    )


def test_encoder_marks_turn_zero_calculation_defaults_without_fake_history() -> None:
    encoded = ObservationEncoder().encode(_observation(0))

    assert encoded.has_history is False
    assert encoded.repetition_signal == 0.0
    assert encoded.adherence_deficit == 0.0
    assert encoded.task_deficit == 0.0
    assert encoded.length_deviation == 0.0
    assert encoded.prompt_signal > 0.0


@pytest.mark.parametrize(
    "metrics",
    (
        _metrics(task_score=None),
        _metrics(adherence=None),
        _metrics(repetition=None),
        _metrics(response_length=None),
    ),
)
def test_encoder_fails_closed_when_a_used_metric_is_unavailable(
    metrics: ResponseMetrics,
) -> None:
    with pytest.raises(ValueError, match="must be available"):
        ObservationEncoder().encode(_observation(1, metrics))


def test_substrate_equations_are_explicit_deterministic_and_bounded() -> None:
    substrate = NeuralSubstrate()
    state = NeuralSubstrateState(
        excitation=0.2,
        inhibition=-0.1,
        adaptation=0.3,
        fatigue=0.4,
        context=-0.2,
    )
    encoded = EncodedObservation(
        turn_index=3,
        repetition_signal=0.8,
        adherence_deficit=0.25,
        task_deficit=0.5,
        length_deviation=-0.75,
        prompt_signal=0.4,
        has_history=True,
    )

    first = substrate.step(state, encoded, controller_seed=91)
    repeated = substrate.step(state, encoded, controller_seed=91)

    assert first == repeated
    assert first.state_after.excitation == round(
        0.55 * 0.2 + 0.35 * 0.8 + 0.20 * 0.5 - 0.15 * -0.1 + 0.10 * 0.4 + first.seed_drive,
        12,
    )
    assert first.state_after.inhibition == round(
        0.60 * -0.1 + 0.30 * 0.25 + 0.10 * 0.4 - 0.10 * 0.2 - 0.50 * first.seed_drive,
        12,
    )
    assert all(
        -1.0 <= value <= 1.0
        for name, value in first.state_after.model_dump().items()
        if name != "fatigue"
    )
    assert 0.0 <= first.state_after.fatigue <= 1.0


def test_substrate_reports_exact_per_variable_saturation() -> None:
    transition = NeuralSubstrate().step(
        NeuralSubstrateState(
            excitation=1.0,
            inhibition=-1.0,
            adaptation=1.0,
            fatigue=1.0,
            context=1.0,
        ),
        EncodedObservation(
            turn_index=1,
            repetition_signal=1.0,
            adherence_deficit=0.0,
            task_deficit=1.0,
            length_deviation=1.0,
            prompt_signal=1.0,
            has_history=True,
        ),
        controller_seed=19,
    )

    assert transition.state_after.excitation == 1.0
    assert transition.state_after.adaptation == 1.0
    assert transition.saturation.model_dump() == {
        "excitation": True,
        "inhibition": False,
        "adaptation": True,
        "fatigue": False,
        "context": False,
    }
    assert transition.saturation.any_saturation is True


@given(
    excitation=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    inhibition=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    adaptation=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    fatigue=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    context=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    repetition=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    adherence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    task=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    length=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    prompt=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    controller_seed=st.integers(min_value=-(2**63), max_value=2**63 - 1),
)
def test_substrate_property_keeps_all_state_finite_and_bounded(
    excitation: float,
    inhibition: float,
    adaptation: float,
    fatigue: float,
    context: float,
    repetition: float,
    adherence: float,
    task: float,
    length: float,
    prompt: float,
    controller_seed: int,
) -> None:
    transition = NeuralSubstrate().step(
        NeuralSubstrateState(
            excitation=excitation,
            inhibition=inhibition,
            adaptation=adaptation,
            fatigue=fatigue,
            context=context,
        ),
        EncodedObservation(
            turn_index=7,
            repetition_signal=repetition,
            adherence_deficit=adherence,
            task_deficit=task,
            length_deviation=length,
            prompt_signal=prompt,
            has_history=True,
        ),
        controller_seed,
    )
    values = transition.state_after.model_dump()
    assert all(-1.0 <= value <= 1.0 for value in values.values())
    assert 0.0 <= transition.state_after.fatigue <= 1.0


def test_decoder_uses_four_distinct_readouts_and_reports_bounded_magnitude() -> None:
    bounds = ActionBounds()
    state = NeuralSubstrateState(
        excitation=0.8,
        inhibition=0.1,
        adaptation=-0.4,
        fatigue=0.3,
        context=0.6,
    )
    decoded = ActionDecoder().decode(
        state,
        bounds,
        action_enabled=True,
    )

    assert bounds.contains(decoded.action)
    assert 0.0 < decoded.action_magnitude <= 1.0
    assert len(set(decoded.activation.model_dump().values())) == 4
    gated = ActionDecoder().decode(
        state,
        bounds,
        action_enabled=False,
    )
    assert gated.action == _ZERO_ACTION
    assert gated.action_magnitude == 0.0


def test_turn_zero_mechanism_and_state_are_byte_equivalent_between_neural_arms() -> None:
    persistent = NeuralPersistentPolicy()
    reset = NeuralMatchedHistoryStateResetPolicy()
    observation = _observation(0)
    persistent_initial = persistent.initial_state(_context(persistent.policy_id))
    reset_initial = reset.initial_state(_context(reset.policy_id))

    persistent_action, persistent_next, persistent_trace = persistent.act(
        observation,
        persistent_initial,
    )
    reset_action, reset_next, reset_trace = reset.act(observation, reset_initial)

    assert canonical_json(persistent_initial) == canonical_json(reset_initial)
    assert persistent_action == reset_action == _ZERO_ACTION
    assert canonical_json(persistent_next) == canonical_json(reset_next)
    assert persistent_trace.mechanism_sha256 == reset_trace.mechanism_sha256
    assert persistent_trace.state_reset_applied is False
    assert reset_trace.state_reset_applied is False


def test_reset_intervention_changes_only_declared_substrate_and_preserves_turn() -> None:
    persistent = NeuralPersistentPolicy()
    reset = NeuralMatchedHistoryStateResetPolicy()
    initial = persistent.initial_state(_context(persistent.policy_id, controller_seed=41))
    _, focal_state, _ = persistent.act(_observation(0), initial)
    observation = _observation(
        1,
        _metrics(task_score=0.2, adherence=0.4, repetition=0.9, response_length=160),
    )

    persistent_action, persistent_next, persistent_trace = persistent.act(
        observation,
        focal_state,
    )
    reset_action, reset_next, reset_trace = reset.act(observation, focal_state)

    assert persistent_trace.stored_substrate_state == reset_trace.stored_substrate_state
    assert persistent_trace.effective_substrate_state == focal_state.substrate
    assert reset_trace.effective_substrate_state == NeuralSubstrate().initial_state(
        focal_state.controller_seed
    )
    assert reset_trace.effective_substrate_state != focal_state.substrate
    assert persistent_trace.observation_encoding == reset_trace.observation_encoding
    assert persistent_trace.substrate_transition.seed_drive == (
        reset_trace.substrate_transition.seed_drive
    )
    assert persistent_action != reset_action
    assert persistent_next.next_turn_index == reset_next.next_turn_index == 2
    assert persistent_next.controller_seed == reset_next.controller_seed
    assert persistent_next.action_bounds == reset_next.action_bounds
    assert persistent_trace.state_reset_applied is False
    assert reset_trace.state_reset_applied is True


def test_neural_state_and_trace_round_trip_without_hidden_state() -> None:
    policy = NeuralPersistentPolicy()
    _, state, trace = policy.act(
        _observation(0),
        policy.initial_state(_context(policy.policy_id)),
    )

    assert NeuralPolicyState.model_validate_json(canonical_json(state)) == state
    assert type(state).model_validate_json(canonical_json(state)) == state
    assert trace.model_validate_json(canonical_json(trace)) == trace
    assert trace.decoder_version == "four-surface-linear-decoder-v1"
    assert trace.action_magnitude_version == "rms-normalized-to-action-bounds-v1"
    with pytest.raises(ValidationError, match="frozen"):
        state.next_turn_index = 99
    with pytest.raises(AttributeError, match="immutable"):
        policy.spec = policy.spec  # type: ignore[misc]
    encoder = policy.encoder
    with pytest.raises(AttributeError):
        encoder.target_response_length_tokens = 1  # type: ignore[attr-defined]

    invalid_trace = trace.model_dump(mode="python")
    invalid_trace["state_reset_applied"] = True
    with pytest.raises(ValidationError, match="mechanism_sha256"):
        type(trace).model_validate(invalid_trace)


@pytest.mark.parametrize(
    "metrics",
    (
        _metrics(task_score=0.1, adherence=1.0, repetition=0.9, response_length=64),
        _metrics(task_score=0.7, adherence=0.2, repetition=0.1, response_length=8),
        _metrics(task_score=0.4, adherence=0.8, repetition=0.4, response_length=256),
    ),
)
def test_designed_neural_actions_do_not_alias_heuristic(metrics: ResponseMetrics) -> None:
    neural = NeuralPersistentPolicy()
    heuristic = HeuristicAdaptivePolicy()
    _, neural_state, _ = neural.act(
        _observation(0),
        neural.initial_state(_context(neural.policy_id)),
    )
    _, heuristic_state, _ = heuristic.act(
        _observation(0),
        heuristic.initial_state(_context(heuristic.policy_id)),
    )

    neural_action, _, _ = neural.act(_observation(1, metrics), neural_state)
    heuristic_action, _, _ = heuristic.act(_observation(1, metrics), heuristic_state)

    assert neural_action != heuristic_action
    assert neural_state.action_bounds.contains(neural_action)


def test_neural_policy_rejects_wrong_state_and_missing_nonzero_history() -> None:
    policy = NeuralPersistentPolicy()
    state = policy.initial_state(_context(policy.policy_id))
    _, state, _ = policy.act(_observation(0), state)

    with pytest.raises(ValueError, match="requires previous-response metrics"):
        policy.act(_observation(1), state)
    with pytest.raises(TypeError, match="NeuralPolicyState"):
        policy.act(_observation(1, _metrics()), object())  # type: ignore[arg-type]

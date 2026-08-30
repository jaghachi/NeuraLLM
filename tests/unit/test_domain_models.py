"""Unit tests for strict, immutable domain contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from neurallm.domain.models import (
    ActionBounds,
    ControllerAction,
    ControllerObservation,
    DecodingBounds,
    DecodingParameters,
    ExperimentCondition,
    MetricValue,
    PromptFeatures,
    ProviderIdentity,
    ResponseMetrics,
    RunManifest,
    SeedSchedule,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256

ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64
PROVIDER_CONFIG_JSON = canonical_json({"provider": "test"})
PROVIDER_CONFIG_HASH = canonical_sha256({"provider": "test"})


def make_metric(value: float = 0.5) -> MetricValue[float]:
    return MetricValue[float](
        value=value,
        availability=True,
        metric_version="metric-v1",
        input_hash=ZERO_HASH,
    )


def make_response_metrics() -> ResponseMetrics:
    return ResponseMetrics(
        task_score=make_metric(),
        instruction_adherence=make_metric(),
        response_length_tokens=MetricValue[int](
            value=42,
            availability=True,
            metric_version="tokens-v1",
            input_hash=ZERO_HASH,
        ),
        repetition_ratio=make_metric(),
        repeated_3_gram_ratio=make_metric(),
        repeated_4_gram_ratio=make_metric(),
        distinct_2=make_metric(),
        distinct_3=make_metric(),
        late_window_repetition_ratio=make_metric(),
        format_validity=make_metric(1.0),
        semantic_similarity=MetricValue[float](
            value=None,
            availability=False,
            metric_version="semantic-v1",
            input_hash=ZERO_HASH,
        ),
    )


def make_provider_identity() -> ProviderIdentity:
    return ProviderIdentity(
        provider_type="fake",
        implementation_version="1.0",
        model_alias="deterministic-fake",
        build_id="builtin",
        provider_config_hash=PROVIDER_CONFIG_HASH,
    )


def valid_decoding_parameters() -> dict[str, Any]:
    return {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "presence_penalty": 0.0,
        "max_tokens": 256,
        "seed": 7,
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("temperature", 0.0),
        ("temperature", float("nan")),
        ("top_p", 0.0),
        ("top_p", 1.01),
        ("top_p", float("inf")),
        ("top_k", -1),
        ("presence_penalty", float("-inf")),
        ("max_tokens", 0),
    ],
)
def test_decoding_parameters_reject_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    values = valid_decoding_parameters()
    values[field_name] = invalid_value

    with pytest.raises(ValidationError):
        DecodingParameters.model_validate(values)


@pytest.mark.parametrize(
    ("field_name", "coerced_value"),
    [
        ("temperature", "0.7"),
        ("top_k", 40.0),
        ("max_tokens", True),
        ("seed", "7"),
    ],
)
def test_decoding_parameters_do_not_coerce_types(
    field_name: str,
    coerced_value: object,
) -> None:
    values = valid_decoding_parameters()
    values[field_name] = coerced_value

    with pytest.raises(ValidationError):
        DecodingParameters.model_validate(values)


def test_domain_models_are_frozen_and_forbid_extras() -> None:
    parameters = DecodingParameters(**valid_decoding_parameters())

    with pytest.raises(ValidationError):
        parameters.temperature = 0.2

    with pytest.raises(ValidationError):
        DecodingParameters(**valid_decoding_parameters(), unknown=True)


def test_prompt_features_are_deeply_immutable_and_finite() -> None:
    features = PromptFeatures({"length": 12.0, "question_marks": 1.0})

    with pytest.raises(TypeError):
        features.root["length"] = 99.0  # type: ignore[index]

    with pytest.raises(ValidationError):
        PromptFeatures({"bad": float("nan")})


def test_controller_action_accepts_inclusive_pilot_bounds() -> None:
    action = ControllerAction(
        temperature_delta=-0.10,
        top_p_delta=0.05,
        top_k_delta=10,
        presence_penalty_delta=-0.20,
    )

    assert action.temperature_delta == -0.10
    assert action.top_p_delta == 0.05
    assert action.top_k_delta == 10
    assert action.presence_penalty_delta == -0.20


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("temperature_delta", float("nan")),
        ("top_p_delta", float("inf")),
        ("presence_penalty_delta", float("-inf")),
    ],
)
def test_controller_action_rejects_nonfinite_deltas(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "temperature_delta": 0.0,
        "top_p_delta": 0.0,
        "top_k_delta": 0,
        "presence_penalty_delta": 0.0,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValidationError):
        ControllerAction.model_validate(values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("temperature_delta", 0.1000001),
        ("top_p_delta", -0.0500001),
        ("top_k_delta", 11),
        ("presence_penalty_delta", -0.2000001),
    ],
)
def test_default_action_bounds_fail_closed_on_out_of_range_actions(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "temperature_delta": 0.0,
        "top_p_delta": 0.0,
        "top_k_delta": 0,
        "presence_penalty_delta": 0.0,
    }
    values[field_name] = invalid_value
    action = ControllerAction.model_validate(values)

    with pytest.raises(ValueError, match="configured run bounds"):
        ActionBounds().require(action)


def test_action_bounds_are_configurable_and_manifest_bound() -> None:
    action = ControllerAction(
        temperature_delta=0.20,
        top_p_delta=0.0,
        top_k_delta=0,
        presence_penalty_delta=0.0,
    )
    custom_bounds = ActionBounds(temperature_delta=(-0.25, 0.25))

    assert not ActionBounds().contains(action)
    assert custom_bounds.require(action) is action


@pytest.mark.parametrize(
    ("field_name", "invalid_bounds"),
    [
        ("temperature", (1.0, 0.5)),
        ("top_p", (0.9, 0.8)),
        ("top_k", (10, 9)),
        ("presence_penalty", (1.0, -1.0)),
    ],
)
def test_decoding_bounds_reject_inverted_intervals(
    field_name: str,
    invalid_bounds: tuple[float, float] | tuple[int, int],
) -> None:
    with pytest.raises(ValidationError, match="lower bound exceeds upper bound"):
        DecodingBounds.model_validate({field_name: invalid_bounds})


def test_controller_action_cannot_control_generation_budget() -> None:
    values = {
        "temperature_delta": 0.0,
        "top_p_delta": 0.0,
        "top_k_delta": 0,
        "presence_penalty_delta": 0.0,
        "max_tokens": 1,
    }

    assert "max_tokens" not in ControllerAction.model_fields
    with pytest.raises(ValidationError):
        ControllerAction.model_validate(values)


def test_metric_availability_exactly_matches_value_presence() -> None:
    unavailable = MetricValue[float](
        value=None,
        availability=False,
        metric_version="metric-v1",
        input_hash=ZERO_HASH,
    )
    assert unavailable.value is None

    for value, availability in ((None, True), (0.5, False)):
        with pytest.raises(ValidationError):
            MetricValue[float](
                value=value,
                availability=availability,
                metric_version="metric-v1",
                input_hash=ZERO_HASH,
            )


def test_optional_semantic_metric_keeps_explicit_unavailable_provenance() -> None:
    metrics = make_response_metrics()

    assert metrics.semantic_similarity.value is None
    assert metrics.semantic_similarity.availability is False
    assert metrics.semantic_similarity.metric_version == "semantic-v1"

    with pytest.raises(ValidationError):
        ResponseMetrics.model_validate({**metrics.model_dump(), "semantic_similarity": None})


@pytest.mark.parametrize("field_name", ["task_score", "instruction_adherence"])
def test_normalized_metrics_reject_values_outside_unit_interval(field_name: str) -> None:
    metrics = make_response_metrics().model_dump()
    metrics[field_name]["value"] = 1.01

    with pytest.raises(ValidationError):
        ResponseMetrics.model_validate(metrics)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_metric_values_reject_nonfinite_numbers(nonfinite: float) -> None:
    with pytest.raises(ValidationError):
        MetricValue[float](
            value=nonfinite,
            availability=True,
            metric_version="metric-v1",
            input_hash=ZERO_HASH,
        )


@pytest.mark.parametrize(
    "invalid_hash",
    ["abc", "A" * 64, "g" * 64, "0" * 63],
)
def test_metric_input_hash_is_lowercase_sha256(invalid_hash: str) -> None:
    with pytest.raises(ValidationError):
        MetricValue[float](
            value=1.0,
            availability=True,
            metric_version="metric-v1",
            input_hash=invalid_hash,
        )


def test_turn_zero_requires_explicit_null_history() -> None:
    observation = ControllerObservation(
        turn_index=0,
        prompt_family="instruction",
        current_prompt_features={"length": 12.0},
        previous_response_metrics=None,
        has_previous_response=False,
    )

    assert observation.previous_response_metrics is None
    assert observation.has_previous_response is False

    with pytest.raises(ValidationError):
        ControllerObservation(
            turn_index=0,
            prompt_family="instruction",
            current_prompt_features={},
            previous_response_metrics=make_response_metrics(),
            has_previous_response=True,
        )


@pytest.mark.parametrize(
    ("metrics_factory", "has_previous_response"),
    [
        (lambda: make_response_metrics(), False),
        (lambda: None, True),
    ],
)
def test_observation_rejects_history_flag_mismatch(
    metrics_factory: Callable[[], ResponseMetrics | None],
    has_previous_response: bool,
) -> None:
    with pytest.raises(ValidationError):
        ControllerObservation(
            turn_index=1,
            prompt_family="instruction",
            current_prompt_features={},
            previous_response_metrics=metrics_factory(),
            has_previous_response=has_previous_response,
        )


def test_condition_identity_covers_every_condition_field() -> None:
    base = ExperimentCondition(
        experiment_id="experiment-a",
        dataset_version="dataset-v1",
        prompt_sequence_id="sequence-1",
        turn_index=0,
        policy_id="static",
        model_seed=11,
        controller_seed=22,
        provider_identity_id=ZERO_HASH,
        base_decoding_profile_id="base-v1",
    )
    changed = base.model_copy(update={"controller_seed": 23})

    assert base.condition_id == base.condition_id
    assert base.condition_id != changed.condition_id
    assert len(base.condition_id) == 64
    assert base.condition_id == base.condition_id.lower()


def test_provider_identity_is_stable_and_not_an_input_field() -> None:
    identity = make_provider_identity()

    assert identity.identity_id == make_provider_identity().identity_id
    assert "identity_id" not in ProviderIdentity.model_fields
    with pytest.raises(ValidationError):
        ProviderIdentity(
            **identity.model_dump(),
            identity_id=ONE_HASH,
        )


def test_run_manifest_binds_and_freezes_phase_one_provenance() -> None:
    manifest = RunManifest(
        source_commit="a" * 40,
        working_tree_clean=True,
        experiment_config_hash=ONE_HASH,
        dataset_hash=ONE_HASH,
        provider_config_hash=PROVIDER_CONFIG_HASH,
        provider_identity=make_provider_identity(),
        provider_effective_configuration_json=PROVIDER_CONFIG_JSON,
        policy_config_hashes={"static": ONE_HASH},
        metric_versions={"task_score": "v1"},
        seed_schedule=SeedSchedule(model_seeds=(1,), controller_seeds=(2,)),
        action_bounds=ActionBounds(),
        decision_rule_version="decision-v1",
        database_schema_version=1,
    )

    with pytest.raises(TypeError):
        manifest.policy_config_hashes["random"] = ZERO_HASH  # type: ignore[index]

    assert manifest.action_bounds.temperature_delta == (-0.10, 0.10)
    assert manifest.decoding_bounds == DecodingBounds()


def test_run_manifest_rejects_provider_config_hash_mismatch() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            source_commit="a" * 40,
            working_tree_clean=True,
            experiment_config_hash=ONE_HASH,
            dataset_hash=ONE_HASH,
            provider_config_hash=ONE_HASH,
            provider_identity=make_provider_identity(),
            provider_effective_configuration_json=PROVIDER_CONFIG_JSON,
            policy_config_hashes={"static": ONE_HASH},
            metric_versions={"task_score": "v1"},
            seed_schedule=SeedSchedule(model_seeds=(1,), controller_seeds=(2,)),
            action_bounds=ActionBounds(),
            decision_rule_version="decision-v1",
            database_schema_version=1,
        )

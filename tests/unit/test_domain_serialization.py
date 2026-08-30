"""Tests for canonical JSON, hashes, and deterministic identifiers."""

from __future__ import annotations

import json

import pytest

from neurallm.domain.identifiers import (
    condition_id,
    condition_identifier,
    deterministic_identifier,
)
from neurallm.domain.models import ExperimentCondition, ProviderIdentity
from neurallm.domain.serialization import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
)

ZERO_HASH = "0" * 64


def make_condition(**updates: object) -> ExperimentCondition:
    values: dict[str, object] = {
        "experiment_id": "experiment-a",
        "dataset_version": "dataset-v1",
        "prompt_sequence_id": "sequence-1",
        "turn_index": 0,
        "policy_id": "static",
        "model_seed": 11,
        "controller_seed": 22,
        "provider_identity_id": ZERO_HASH,
        "base_decoding_profile_id": "base-v1",
    }
    values.update(updates)
    return ExperimentCondition.model_validate(values)


def test_canonical_json_is_sorted_compact_utf8_and_reproducible() -> None:
    value = {"z": 1, "a": "µ"}

    assert canonical_json(value) == '{"a":"µ","z":1}'
    assert canonical_json_bytes(value) == b'{"a":"\xc2\xb5","z":1}'
    assert (
        canonical_sha256(value)
        == "eacbaec97a8fd81fe45be99bba3b690d7f9e905aab34e4717712940f092c4476"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_numbers_at_any_depth(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"outer": [0, {"value": value}]})


def test_canonical_json_rejects_non_string_object_keys() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json({1: "not canonical"})


def test_model_canonicalization_is_independent_of_input_field_order() -> None:
    forwards = make_condition()
    backwards = ExperimentCondition.model_validate(
        dict(reversed(list(forwards.model_dump().items())))
    )

    assert canonical_json(forwards) == canonical_json(backwards)
    assert canonical_sha256(forwards) == canonical_sha256(backwards)


def test_condition_identifier_is_the_direct_canonical_condition_hash() -> None:
    condition = make_condition()

    assert condition_id(condition) == canonical_sha256(condition)
    assert condition_identifier(condition) == condition_id(condition)
    assert condition.condition_id == condition_id(condition)


def test_namespaced_identifier_prevents_cross_domain_aliases() -> None:
    payload = {"same": "payload"}

    assert deterministic_identifier("condition", payload) != deterministic_identifier(
        "provider",
        payload,
    )
    assert deterministic_identifier("condition", payload) == deterministic_identifier(
        "condition",
        payload,
    )


def test_provider_identity_hash_includes_explicit_nulls() -> None:
    identity = ProviderIdentity(
        provider_type="fake",
        implementation_version="1.0",
        model_alias="deterministic-fake",
        build_id="builtin",
        provider_config_hash=ZERO_HASH,
    )
    serialized = json.loads(canonical_json(identity))

    assert serialized["model_path"] is None
    assert serialized["model_sha256"] is None
    assert serialized["chat_template_sha256"] is None
    assert identity.identity_id == canonical_sha256(identity)

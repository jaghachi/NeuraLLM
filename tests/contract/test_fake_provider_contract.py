"""Contract tests for the deterministic, zero-network fake provider."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from neurallm.domain.models import DecodingParameters, ExperimentCondition, ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import (
    GenerationMetadata,
    GenerationProvider,
    GenerationRequest,
    ProviderIdentityMismatchError,
)
from neurallm.providers.fake import FakeProvider


def _request(
    *,
    prompt: str = "Explain why deterministic tests matter.",
    decoding_parameters: DecodingParameters | None = None,
    provider_identity_id: str | None = None,
) -> GenerationRequest:
    bound_identity_id = provider_identity_id or FakeProvider().provider_identity.identity_id
    return GenerationRequest(
        prompt=prompt,
        decoding_parameters=(
            decoding_parameters
            or DecodingParameters(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.1,
                max_tokens=128,
                seed=17,
            )
        ),
        condition=ExperimentCondition(
            experiment_id="phase-1-contract",
            dataset_version="development-v1",
            prompt_sequence_id="sequence-001",
            turn_index=0,
            policy_id="fake-test",
            model_seed=17,
            controller_seed=23,
            provider_identity_id=bound_identity_id,
            base_decoding_profile_id="base-v1",
        ),
    )


def test_fake_provider_satisfies_the_shared_provider_protocol() -> None:
    provider = FakeProvider()

    assert isinstance(provider, GenerationProvider)
    assert provider.provider_identity.provider_type == "fake"
    assert provider.provider_identity.implementation_version == "2.0.0"
    assert provider.provider_identity.model_alias == "deterministic-fake"
    effective_configuration = json.loads(provider.effective_configuration_json)
    assert effective_configuration["generation_method"] == "fake_provider_visible_sha256_v2"
    assert provider.provider_identity.provider_config_hash == canonical_sha256(
        effective_configuration
    )


def test_identical_typed_inputs_produce_identical_outputs_without_live_state() -> None:
    request = _request()
    provider = FakeProvider()

    first = provider.generate(request)
    second = FakeProvider().generate(request.model_copy(deep=True))
    provider_visible_inputs = {
        "prompt": request.prompt,
        "decoding_parameters": request.decoding_parameters,
        "provider_identity": provider.provider_identity,
        "provider_configuration": json.loads(provider.effective_configuration_json),
    }

    assert first == second
    assert first.text == f"fake-response-v2:{canonical_sha256(provider_visible_inputs)}"
    assert first.provider_identity == FakeProvider().provider_identity
    assert first.effective_parameters == request.decoding_parameters
    assert first.raw_metadata.request_sha256 == canonical_sha256(request)
    assert first.raw_metadata.generation_method == "fake_provider_visible_sha256_v2"


@pytest.mark.parametrize(
    ("condition_field", "changed_value"),
    [
        ("experiment_id", "another-experiment"),
        ("dataset_version", "another-dataset-v1"),
        ("prompt_sequence_id", "sequence-002"),
        ("turn_index", 1),
        ("policy_id", "another-policy"),
        ("model_seed", 18),
        ("controller_seed", 24),
        ("base_decoding_profile_id", "another-base-v1"),
    ],
)
def test_orchestration_metadata_does_not_change_response_bytes(
    condition_field: str,
    changed_value: str | int,
) -> None:
    provider = FakeProvider()
    baseline_request = _request()
    changed_condition = ExperimentCondition.model_validate(
        {
            **baseline_request.condition.model_dump(mode="python"),
            condition_field: changed_value,
        }
    )
    changed_request = GenerationRequest(
        prompt=baseline_request.prompt,
        decoding_parameters=baseline_request.decoding_parameters,
        condition=changed_condition,
    )

    baseline = provider.generate(baseline_request)
    changed = provider.generate(changed_request)

    assert changed.text.encode("utf-8") == baseline.text.encode("utf-8")
    assert changed.raw_metadata.request_sha256 != baseline.raw_metadata.request_sha256


@pytest.mark.parametrize(
    "changed_request",
    [
        _request(prompt="A different prompt."),
        _request(
            decoding_parameters=DecodingParameters(
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.1,
                max_tokens=128,
                seed=17,
            )
        ),
        _request(
            decoding_parameters=DecodingParameters(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                presence_penalty=0.1,
                max_tokens=128,
                seed=18,
            )
        ),
    ],
)
def test_provider_visible_input_changes_change_response_bytes(
    changed_request: GenerationRequest,
) -> None:
    provider = FakeProvider()

    baseline = provider.generate(_request())
    changed = provider.generate(changed_request)

    assert changed.text.encode("utf-8") != baseline.text.encode("utf-8")
    assert changed.raw_metadata.request_sha256 != baseline.raw_metadata.request_sha256


def test_provider_identity_change_changes_response_bytes() -> None:
    baseline_provider = FakeProvider()
    changed_identity = baseline_provider.provider_identity.model_copy(
        update={"build_id": "alternate-builtin"}
    )
    changed_provider = FakeProvider(changed_identity)

    baseline = baseline_provider.generate(_request())
    changed = changed_provider.generate(_request(provider_identity_id=changed_identity.identity_id))

    assert changed.text.encode("utf-8") != baseline.text.encode("utf-8")


def test_response_metadata_and_effective_parameters_are_immutable_and_typed() -> None:
    response = FakeProvider().generate(_request())

    with pytest.raises(ValidationError, match="frozen"):
        response.raw_metadata.request_sha256 = "0" * 64

    with pytest.raises(ValidationError, match="frozen"):
        response.effective_parameters.temperature = 0.8

    with pytest.raises(ValidationError):
        GenerationMetadata.model_validate(
            {
                **response.raw_metadata.model_dump(),
                "untyped": "value",
            }
        )


def test_fake_provider_fails_closed_on_request_identity_mismatch() -> None:
    provider = FakeProvider()
    request = _request(provider_identity_id="f" * 64)

    assert request.provider_identity_id != provider.provider_identity.identity_id
    with pytest.raises(ProviderIdentityMismatchError, match="does not match"):
        provider.generate(request)


def test_request_derives_condition_and_provider_ids_from_one_typed_condition() -> None:
    request = _request()

    assert request.condition_id == request.condition.condition_id
    assert request.provider_identity_id == request.condition.provider_identity_id
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate({**request.model_dump(), "provider_identity_id": "f" * 64})


def test_fake_provider_cannot_stamp_a_non_fake_identity() -> None:
    non_fake_identity = ProviderIdentity(
        provider_type="llama_cpp",
        implementation_version="1.0",
        model_alias="not-a-fake",
        build_id="external",
        provider_config_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="type 'fake'"):
        FakeProvider(non_fake_identity)


def test_generation_metadata_rejects_partial_or_tampered_protocol_evidence() -> None:
    payload = {"content": "raw"}

    with pytest.raises(ValidationError, match="must be complete"):
        GenerationMetadata(
            request_sha256="0" * 64,
            generation_method="llama_cpp_completion_http_v1",
            provider_request_json=canonical_json(payload),
        )

    with pytest.raises(ValidationError, match="SHA-256"):
        GenerationMetadata(
            request_sha256="0" * 64,
            generation_method="llama_cpp_completion_http_v1",
            provider_request_json=canonical_json(payload),
            provider_request_sha256="0" * 64,
            provider_response_json=canonical_json(payload),
            provider_response_sha256=canonical_sha256(payload),
        )

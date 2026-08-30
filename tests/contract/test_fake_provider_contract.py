"""Contract tests for the deterministic, zero-network fake provider."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neurallm.domain.models import DecodingParameters, ExperimentCondition, ProviderIdentity
from neurallm.domain.serialization import canonical_sha256
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
    controller_seed: int = 23,
    provider_identity_id: str | None = None,
) -> GenerationRequest:
    bound_identity_id = provider_identity_id or FakeProvider().provider_identity.identity_id
    return GenerationRequest(
        prompt=prompt,
        decoding_parameters=DecodingParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.1,
            max_tokens=128,
            seed=17,
        ),
        condition=ExperimentCondition(
            experiment_id="phase-1-contract",
            dataset_version="development-v1",
            prompt_sequence_id="sequence-001",
            turn_index=0,
            policy_id="fake-test",
            model_seed=17,
            controller_seed=controller_seed,
            provider_identity_id=bound_identity_id,
            base_decoding_profile_id="base-v1",
        ),
    )


def test_fake_provider_satisfies_the_shared_provider_protocol() -> None:
    provider = FakeProvider()

    assert isinstance(provider, GenerationProvider)
    assert provider.provider_identity.provider_type == "fake"
    assert provider.provider_identity.model_alias == "deterministic-fake"


def test_identical_typed_inputs_produce_identical_outputs_without_live_state() -> None:
    request = _request()

    first = FakeProvider().generate(request)
    second = FakeProvider().generate(request.model_copy(deep=True))

    assert first == second
    assert first.text == f"fake-response:{canonical_sha256(request)}"
    assert first.provider_identity == FakeProvider().provider_identity
    assert first.effective_parameters == request.decoding_parameters
    assert first.raw_metadata.request_sha256 == canonical_sha256(request)
    assert first.raw_metadata.generation_method == "request_sha256_v1"


@pytest.mark.parametrize(
    "changed_request",
    [
        _request(prompt="A different prompt."),
        _request(controller_seed=24),
    ],
)
def test_scientific_input_changes_change_the_fake_response(
    changed_request: GenerationRequest,
) -> None:
    baseline = FakeProvider().generate(_request())
    changed = FakeProvider().generate(changed_request)

    assert changed.text != baseline.text
    assert changed.raw_metadata.request_sha256 != baseline.raw_metadata.request_sha256


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

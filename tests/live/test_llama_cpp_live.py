"""Explicitly configured live smoke test for a local llama.cpp server."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from neurallm.domain.models import DecodingParameters, ExperimentCondition
from neurallm.providers import GenerationRequest, LlamaCppProvider, LlamaCppProviderConfig


class _LiveSmokeConfiguration(BaseModel):
    """Complete provider and request inputs for one deliberate live dispatch."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    provider: LlamaCppProviderConfig
    prompt: str = Field(min_length=1)
    decoding_parameters: DecodingParameters
    experiment_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    prompt_sequence_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    controller_seed: int
    base_decoding_profile_id: str = Field(min_length=1)


@pytest.mark.live
def test_explicit_live_llama_cpp_preflight_and_single_generation() -> None:
    """Run only after both the live marker and complete JSON config are supplied."""

    raw_configuration = os.environ.get("NEURALLM_LIVE_LLAMA_CONFIG_JSON")
    if raw_configuration is None:
        pytest.skip(
            "set NEURALLM_LIVE_LLAMA_CONFIG_JSON to a complete live smoke payload; "
            "no HTTP request was attempted"
        )
    try:
        configuration = _LiveSmokeConfiguration.model_validate_json(raw_configuration)
    except ValidationError as error:
        pytest.fail(
            "NEURALLM_LIVE_LLAMA_CONFIG_JSON is present but invalid; "
            f"no HTTP request was attempted: {error}"
        )

    with LlamaCppProvider(configuration.provider) as provider:
        request = GenerationRequest(
            prompt=configuration.prompt,
            decoding_parameters=configuration.decoding_parameters,
            condition=ExperimentCondition(
                experiment_id=configuration.experiment_id,
                dataset_version=configuration.dataset_version,
                prompt_sequence_id=configuration.prompt_sequence_id,
                turn_index=0,
                policy_id=configuration.policy_id,
                model_seed=configuration.decoding_parameters.seed,
                controller_seed=configuration.controller_seed,
                provider_identity_id=provider.provider_identity.identity_id,
                base_decoding_profile_id=configuration.base_decoding_profile_id,
            ),
        )
        response = provider.generate(request)

    assert response.provider_identity == provider.provider_identity
    assert response.effective_parameters.max_tokens == configuration.decoding_parameters.max_tokens
    assert response.effective_parameters.seed == configuration.decoding_parameters.seed
    assert response.text.strip()
    assert provider.last_raw_response_sha256 is not None

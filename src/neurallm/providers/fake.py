"""Deterministic, dependency-free generation provider for tests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from neurallm.domain.models import DecodingParameters, ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import (
    GenerationMetadata,
    GenerationRequest,
    GenerationResponse,
    ProviderIdentityMismatchError,
)


class _FakeProviderConfiguration(BaseModel):
    """Scientific inputs defining the built-in fake implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider_type: Literal["fake"] = "fake"
    implementation_version: Literal["2.0.0"] = "2.0.0"
    generation_method: Literal["fake_provider_visible_sha256_v2"] = (
        "fake_provider_visible_sha256_v2"
    )


class _FakeProviderVisibleInputs(BaseModel):
    """Only inputs visible to the provider when generating response content."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt: str
    decoding_parameters: DecodingParameters
    provider_identity: ProviderIdentity
    provider_configuration: _FakeProviderConfiguration


class FakeProvider:
    """Return a stable response derived solely from a typed request.

    The provider performs no I/O and has no environment or network dependency.
    A custom identity may be injected for contract tests; otherwise an explicit
    built-in identity is created when the instance is constructed.
    """

    __slots__ = ("_provider_identity",)

    def __init__(self, provider_identity: ProviderIdentity | None = None) -> None:
        identity = provider_identity or fake_provider_identity()
        if identity.provider_type != "fake":
            raise ValueError("FakeProvider requires a provider identity of type 'fake'")
        self._provider_identity = identity

    @property
    def provider_identity(self) -> ProviderIdentity:
        """Return the immutable identity bound to this fake instance."""
        return self._provider_identity

    @property
    def effective_configuration_json(self) -> str:
        """Return the exact canonical configuration bound by the identity hash."""

        return fake_provider_effective_configuration_json()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate from provider-visible inputs and retain full request evidence."""

        if request.provider_identity_id != self.provider_identity.identity_id:
            raise ProviderIdentityMismatchError(
                "generation request provider identity does not match FakeProvider"
            )
        request_sha256 = canonical_sha256(request)
        configuration = _FakeProviderConfiguration()
        visible_inputs_sha256 = canonical_sha256(
            _FakeProviderVisibleInputs(
                prompt=request.prompt,
                decoding_parameters=request.decoding_parameters,
                provider_identity=self.provider_identity,
                provider_configuration=configuration,
            )
        )
        return GenerationResponse(
            text=f"fake-response-v2:{visible_inputs_sha256}",
            provider_identity=self.provider_identity,
            effective_parameters=request.decoding_parameters,
            raw_metadata=GenerationMetadata(
                request_sha256=request_sha256,
                generation_method=configuration.generation_method,
            ),
        )


def fake_provider_effective_configuration_json() -> str:
    """Return fake-provider configuration evidence without constructing a provider."""

    return canonical_json(_FakeProviderConfiguration())


def fake_provider_identity() -> ProviderIdentity:
    """Return the canonical built-in identity without constructing a provider."""

    configuration = _FakeProviderConfiguration()
    return ProviderIdentity(
        provider_type=configuration.provider_type,
        implementation_version=configuration.implementation_version,
        model_alias="deterministic-fake",
        build_id="builtin",
        provider_config_hash=canonical_sha256(configuration),
    )


__all__ = [
    "FakeProvider",
    "fake_provider_effective_configuration_json",
    "fake_provider_identity",
]

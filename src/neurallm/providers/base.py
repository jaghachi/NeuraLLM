"""Typed request and response boundary for text-generation providers."""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from neurallm.domain.models import DecodingParameters, ExperimentCondition, ProviderIdentity


class ProviderIdentityMismatchError(ValueError):
    """Raised before generation when a request targets another provider identity."""


class _StrictFrozenModel(BaseModel):
    """Base configuration for immutable provider-boundary values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GenerationRequest(_StrictFrozenModel):
    """One fully specified generation request."""

    prompt: str = Field(min_length=1)
    decoding_parameters: DecodingParameters
    condition: ExperimentCondition

    @property
    def condition_id(self) -> str:
        """Return the identifier derived from the typed condition."""

        return self.condition.condition_id

    @property
    def provider_identity_id(self) -> str:
        """Return the only provider identity permitted for this request."""

        return self.condition.provider_identity_id


class GenerationMetadata(_StrictFrozenModel):
    """Typed provider metadata required to reproduce a fake response."""

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_method: Literal["request_sha256_v1"] = "request_sha256_v1"


class GenerationResponse(_StrictFrozenModel):
    """Provider output with identity and observed effective settings."""

    text: str = Field(min_length=1)
    provider_identity: ProviderIdentity
    effective_parameters: DecodingParameters
    raw_metadata: GenerationMetadata


@runtime_checkable
class GenerationProvider(Protocol):
    """Structural contract implemented by every generation provider."""

    @property
    def provider_identity(self) -> ProviderIdentity:
        """Return the explicit identity bound to this provider instance."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one response without changing the requested condition."""
        ...


__all__ = [
    "GenerationMetadata",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "ProviderIdentityMismatchError",
]

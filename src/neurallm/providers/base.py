"""Typed request and response boundary for text-generation providers."""

import json
from math import isclose
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neurallm.domain.models import (
    DecodingParameters,
    ExperimentCondition,
    ProviderIdentity,
    Sha256Hex,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256

DECODING_FLOAT_RELATIVE_TOLERANCE = 1e-6
DECODING_FLOAT_ABSOLUTE_TOLERANCE = 1e-8


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
    """Logical request identity plus retained provider-protocol payloads."""

    request_sha256: Sha256Hex
    generation_method: Literal[
        "request_sha256_v1",
        "fake_provider_visible_sha256_v2",
        "llama_cpp_completion_http_v1",
    ] = "request_sha256_v1"
    provider_request_json: str | None = None
    provider_request_sha256: Sha256Hex | None = None
    provider_response_json: str | None = None
    provider_response_sha256: Sha256Hex | None = None

    @staticmethod
    def _validate_protocol_payload(value: str, digest: str, name: str) -> None:
        try:
            parsed: object = json.loads(value)
            if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
                raise ValueError(f"{name} must encode a JSON object")
            if canonical_json(parsed) != value:
                raise ValueError(f"{name} must be canonical JSON")
            if canonical_sha256(parsed) != digest:
                raise ValueError(f"{name} SHA-256 does not match its payload")
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{name} must be finite canonical JSON") from exc

    @model_validator(mode="after")
    def _validate_protocol_evidence(self) -> Self:
        values = (
            self.provider_request_json,
            self.provider_request_sha256,
            self.provider_response_json,
            self.provider_response_sha256,
        )
        has_protocol_evidence = all(value is not None for value in values)
        if any(value is not None for value in values) != has_protocol_evidence:
            raise ValueError("provider request/response payloads and hashes must be complete")
        requires_protocol_evidence = self.generation_method == "llama_cpp_completion_http_v1"
        if has_protocol_evidence != requires_protocol_evidence:
            raise ValueError("generation method and provider protocol evidence disagree")
        if has_protocol_evidence:
            assert self.provider_request_json is not None
            assert self.provider_request_sha256 is not None
            assert self.provider_response_json is not None
            assert self.provider_response_sha256 is not None
            self._validate_protocol_payload(
                self.provider_request_json,
                self.provider_request_sha256,
                "provider_request_json",
            )
            self._validate_protocol_payload(
                self.provider_response_json,
                self.provider_response_sha256,
                "provider_response_json",
            )
        return self


class GenerationResponse(_StrictFrozenModel):
    """Provider output with identity and observed effective settings."""

    text: str = Field(min_length=1)
    provider_identity: ProviderIdentity
    effective_parameters: DecodingParameters
    raw_metadata: GenerationMetadata

    @model_validator(mode="after")
    def _validate_provider_protocol_pair(self) -> Self:
        llama_identity = self.provider_identity.provider_type == "llama_cpp"
        llama_protocol = self.raw_metadata.generation_method == "llama_cpp_completion_http_v1"
        if llama_identity != llama_protocol:
            raise ValueError("llama_cpp provider identity and generation protocol must agree")
        return self


def effective_parameters_match_request(
    effective: DecodingParameters,
    requested: DecodingParameters,
) -> bool:
    """Match provider echoes using the one narrow cross-layer float tolerance."""

    return (
        isclose(
            effective.temperature,
            requested.temperature,
            rel_tol=DECODING_FLOAT_RELATIVE_TOLERANCE,
            abs_tol=DECODING_FLOAT_ABSOLUTE_TOLERANCE,
        )
        and isclose(
            effective.top_p,
            requested.top_p,
            rel_tol=DECODING_FLOAT_RELATIVE_TOLERANCE,
            abs_tol=DECODING_FLOAT_ABSOLUTE_TOLERANCE,
        )
        and effective.top_k == requested.top_k
        and isclose(
            effective.presence_penalty,
            requested.presence_penalty,
            rel_tol=DECODING_FLOAT_RELATIVE_TOLERANCE,
            abs_tol=DECODING_FLOAT_ABSOLUTE_TOLERANCE,
        )
        and effective.max_tokens == requested.max_tokens
        and effective.seed == requested.seed
    )


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
    "DECODING_FLOAT_ABSOLUTE_TOLERANCE",
    "DECODING_FLOAT_RELATIVE_TOLERANCE",
    "GenerationMetadata",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "ProviderIdentityMismatchError",
    "effective_parameters_match_request",
]

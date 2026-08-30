"""Generation provider contracts and implementations."""

from neurallm.providers.base import (
    DECODING_FLOAT_ABSOLUTE_TOLERANCE,
    DECODING_FLOAT_RELATIVE_TOLERANCE,
    GenerationMetadata,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderIdentityMismatchError,
    effective_parameters_match_request,
)
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.providers.llama_cpp import (
    LLAMA_CPP_IMPLEMENTATION_VERSION,
    LlamaCppEffectiveConfiguration,
    LlamaCppIdentityDriftError,
    LlamaCppProtocolError,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    LlamaCppProviderError,
    LlamaCppTransportError,
    llama_cpp_provider_identity,
)

__all__ = [
    "DECODING_FLOAT_ABSOLUTE_TOLERANCE",
    "DECODING_FLOAT_RELATIVE_TOLERANCE",
    "FakeProvider",
    "fake_provider_effective_configuration_json",
    "fake_provider_identity",
    "GenerationMetadata",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "LlamaCppEffectiveConfiguration",
    "LLAMA_CPP_IMPLEMENTATION_VERSION",
    "LlamaCppIdentityDriftError",
    "LlamaCppProtocolError",
    "LlamaCppProvider",
    "LlamaCppProviderConfig",
    "LlamaCppProviderError",
    "LlamaCppTransportError",
    "llama_cpp_provider_identity",
    "ProviderIdentityMismatchError",
    "effective_parameters_match_request",
]

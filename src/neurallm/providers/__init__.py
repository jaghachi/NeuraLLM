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
    require_llama_cpp_provider_binding,
)
from neurallm.providers.llama_cpp_evidence import (
    LlamaCppGenerationBindingError,
    reconstruct_llama_cpp_generation_binding,
    require_llama_cpp_generation_binding,
)
from neurallm.providers.preflight import LlamaCppPreflightResult, preflight_llama_cpp

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
    "LlamaCppGenerationBindingError",
    "LlamaCppProtocolError",
    "LlamaCppProvider",
    "LlamaCppProviderConfig",
    "LlamaCppProviderError",
    "LlamaCppPreflightResult",
    "LlamaCppTransportError",
    "llama_cpp_provider_identity",
    "require_llama_cpp_provider_binding",
    "reconstruct_llama_cpp_generation_binding",
    "require_llama_cpp_generation_binding",
    "preflight_llama_cpp",
    "ProviderIdentityMismatchError",
    "effective_parameters_match_request",
]

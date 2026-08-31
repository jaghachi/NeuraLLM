"""Cross-object binding for retained llama.cpp request/response evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import isfinite

from neurallm.domain.models import (
    DecodingParameters,
    ExperimentCondition,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.providers.base import (
    GenerationMetadata,
    GenerationRequest,
    GenerationResponse,
    effective_parameters_match_request,
)


class LlamaCppGenerationBindingError(ValueError):
    """Raised when retained llama.cpp wire evidence crosses domain objects."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise LlamaCppGenerationBindingError(f"{name} must be a JSON object")
    return value


def _finite_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlamaCppGenerationBindingError(f"generation_settings.{key} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise LlamaCppGenerationBindingError(f"generation_settings.{key} must be finite")
    return result


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LlamaCppGenerationBindingError(f"generation_settings.{key} must be an integer")
    return value


def _parse_protocol_json(value: str | None, name: str) -> Mapping[str, object]:
    if value is None:
        raise LlamaCppGenerationBindingError(f"llama.cpp {name} evidence is missing")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LlamaCppGenerationBindingError(f"llama.cpp {name} evidence is invalid JSON") from exc
    return _mapping(parsed, f"llama.cpp {name} evidence")


def require_llama_cpp_generation_binding(
    request: GenerationRequest,
    response: GenerationResponse,
) -> None:
    """Require exact wire/domain identity for one retained llama.cpp generation."""

    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    if not isinstance(response, GenerationResponse):
        raise TypeError("response must be a GenerationResponse")
    metadata = response.raw_metadata
    if metadata.generation_method != "llama_cpp_completion_http_v1":
        raise LlamaCppGenerationBindingError(
            "llama.cpp generation requires completion-protocol metadata"
        )
    if response.provider_identity.provider_type != "llama_cpp":
        raise LlamaCppGenerationBindingError("llama.cpp response has another provider type")
    if response.provider_identity.identity_id != request.provider_identity_id:
        raise LlamaCppGenerationBindingError("llama.cpp response targets another request")
    if metadata.request_sha256 != canonical_sha256(request):
        raise LlamaCppGenerationBindingError(
            "llama.cpp metadata does not bind the canonical request"
        )

    parameters = request.decoding_parameters
    expected_request: Mapping[str, object] = {
        "prompt": request.prompt,
        "model": response.provider_identity.model_alias,
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
        "top_k": parameters.top_k,
        "presence_penalty": parameters.presence_penalty,
        "n_predict": parameters.max_tokens,
        "seed": parameters.seed,
        "stream": False,
        "cache_prompt": False,
    }
    provider_request = _parse_protocol_json(
        metadata.provider_request_json,
        "request",
    )
    if provider_request != expected_request:
        raise LlamaCppGenerationBindingError(
            "llama.cpp provider request does not exactly match GenerationRequest"
        )

    provider_response = _parse_protocol_json(
        metadata.provider_response_json,
        "response",
    )
    if not response.text.strip() or provider_response.get("content") != response.text:
        raise LlamaCppGenerationBindingError(
            "llama.cpp response content differs from GenerationResponse.text"
        )
    if provider_response.get("stop") is not True:
        raise LlamaCppGenerationBindingError("llama.cpp response does not report stop=true")
    if provider_response.get("model") != response.provider_identity.model_alias:
        raise LlamaCppGenerationBindingError("llama.cpp response model alias differs")
    settings = _mapping(
        provider_response.get("generation_settings"),
        "llama.cpp generation_settings",
    )
    observed = DecodingParameters(
        temperature=_finite_float(settings, "temperature"),
        top_p=_finite_float(settings, "top_p"),
        top_k=_integer(settings, "top_k"),
        presence_penalty=_finite_float(settings, "presence_penalty"),
        max_tokens=_integer(settings, "n_predict"),
        seed=_integer(settings, "seed"),
    )
    if _integer(settings, "max_tokens") != observed.max_tokens:
        raise LlamaCppGenerationBindingError("llama.cpp response generation budgets disagree")
    if not effective_parameters_match_request(observed, parameters):
        raise LlamaCppGenerationBindingError(
            "llama.cpp response settings differ from GenerationRequest"
        )
    if not effective_parameters_match_request(observed, response.effective_parameters):
        raise LlamaCppGenerationBindingError(
            "llama.cpp response settings differ from GenerationResponse"
        )


def reconstruct_llama_cpp_generation_binding(
    *,
    condition: ExperimentCondition,
    decoding_parameters: DecodingParameters,
    provider_identity: ProviderIdentity,
    metadata: GenerationMetadata,
) -> tuple[GenerationRequest, GenerationResponse]:
    """Reconstruct typed domain objects from retained canonical wire payloads."""

    provider_request = _parse_protocol_json(metadata.provider_request_json, "request")
    prompt = provider_request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise LlamaCppGenerationBindingError("llama.cpp provider request prompt is invalid")
    provider_response = _parse_protocol_json(metadata.provider_response_json, "response")
    content = provider_response.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LlamaCppGenerationBindingError("llama.cpp provider response content is invalid")
    settings = _mapping(
        provider_response.get("generation_settings"),
        "llama.cpp generation_settings",
    )
    effective_parameters = DecodingParameters(
        temperature=_finite_float(settings, "temperature"),
        top_p=_finite_float(settings, "top_p"),
        top_k=_integer(settings, "top_k"),
        presence_penalty=_finite_float(settings, "presence_penalty"),
        max_tokens=_integer(settings, "n_predict"),
        seed=_integer(settings, "seed"),
    )
    request = GenerationRequest(
        prompt=prompt,
        decoding_parameters=decoding_parameters,
        condition=condition,
    )
    response = GenerationResponse(
        text=content,
        provider_identity=provider_identity,
        effective_parameters=effective_parameters,
        raw_metadata=metadata,
    )
    require_llama_cpp_generation_binding(request, response)
    return request, response


__all__ = [
    "LlamaCppGenerationBindingError",
    "reconstruct_llama_cpp_generation_binding",
    "require_llama_cpp_generation_binding",
]

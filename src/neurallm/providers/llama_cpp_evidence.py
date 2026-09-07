"""Cross-object binding for retained llama.cpp request/response evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from math import isfinite

from neurallm.domain.models import (
    DecodingParameters,
    ExperimentCondition,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import (
    GenerationMetadata,
    GenerationRequest,
    GenerationResponse,
    effective_parameters_match_request,
)
from neurallm.providers.prompt_rendering import render_no_thinking_prompt


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


def _template_input_prompt(provider_request: Mapping[str, object]) -> str:
    template_request = _mapping(
        provider_request.get("template_request"), "llama.cpp template request"
    )
    messages = template_request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise LlamaCppGenerationBindingError("llama.cpp template request needs one user message")
    message = _mapping(messages[0], "llama.cpp template user message")
    prompt = message.get("content")
    if not isinstance(prompt, str) or not prompt:
        raise LlamaCppGenerationBindingError("llama.cpp template input prompt is invalid")
    return prompt


def _template_completion_request(
    provider_request: Mapping[str, object],
    request: GenerationRequest,
    identity: ProviderIdentity,
) -> tuple[Mapping[str, object], str]:
    if set(provider_request) != {
        "chat_template",
        "template_request",
        "template_response",
        "completion_request",
    }:
        raise LlamaCppGenerationBindingError("llama.cpp template evidence envelope differs")
    chat_template = provider_request.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or sha256(chat_template.encode("utf-8")).hexdigest() != identity.chat_template_sha256
    ):
        raise LlamaCppGenerationBindingError("llama.cpp template text does not bind its identity")
    expected_template_request = {
        "model": identity.model_alias,
        "messages": [{"role": "user", "content": request.prompt}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    template_request = _mapping(
        provider_request.get("template_request"), "llama.cpp template request"
    )
    if canonical_json(template_request) != canonical_json(expected_template_request):
        raise LlamaCppGenerationBindingError(
            "llama.cpp template request does not exactly match GenerationRequest"
        )
    template_response = _mapping(
        provider_request.get("template_response"), "llama.cpp template response"
    )
    prompt = template_response.get("prompt")
    if set(template_response) != {"prompt"} or not isinstance(prompt, str) or not prompt.strip():
        raise LlamaCppGenerationBindingError("llama.cpp rendered template prompt is invalid")
    try:
        expected_prompt = render_no_thinking_prompt(
            request.prompt, sha256(chat_template.encode("utf-8")).hexdigest()
        )
    except ValueError as exc:
        raise LlamaCppGenerationBindingError("llama.cpp chat template is unsupported") from exc
    if prompt != expected_prompt:
        raise LlamaCppGenerationBindingError(
            "llama.cpp rendered prompt differs from the pinned no-thinking template"
        )
    return (
        _mapping(provider_request.get("completion_request"), "llama.cpp completion request"),
        prompt,
    )


def _require_template_completion_termination(
    provider_response: Mapping[str, object], max_tokens: int
) -> None:
    if provider_response.get("stop_type") not in ("eos", "word", "limit"):
        raise LlamaCppGenerationBindingError("llama.cpp response stop_type is invalid")
    tokens_predicted = provider_response.get("tokens_predicted")
    if (
        isinstance(tokens_predicted, bool)
        or not isinstance(tokens_predicted, int)
        or not 0 <= tokens_predicted <= max_tokens
    ):
        raise LlamaCppGenerationBindingError("llama.cpp response tokens_predicted is invalid")
    if not isinstance(provider_response.get("truncated"), bool):
        raise LlamaCppGenerationBindingError("llama.cpp response truncated must be a boolean")


def _require_template_metadata_hashes(metadata: GenerationMetadata) -> None:
    for payload, digest, name in (
        (metadata.provider_request_json, metadata.provider_request_sha256, "request"),
        (metadata.provider_response_json, metadata.provider_response_sha256, "response"),
    ):
        parsed = _parse_protocol_json(payload, name)
        if canonical_json(parsed) != payload or canonical_sha256(parsed) != digest:
            raise LlamaCppGenerationBindingError(
                f"llama.cpp {name} evidence is not canonical hash-bound JSON"
            )


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
    if metadata.generation_method not in (
        "llama_cpp_completion_http_v1",
        "llama_cpp_chat_template_http_v2",
    ):
        raise LlamaCppGenerationBindingError(
            "llama.cpp generation requires completion-protocol metadata"
        )
    if response.provider_identity.provider_type != "llama_cpp":
        raise LlamaCppGenerationBindingError("llama.cpp response has another provider type")
    template_protocol = metadata.generation_method == "llama_cpp_chat_template_http_v2"
    template_identity = (
        response.provider_identity.implementation_version == "llama-cpp-chat-template-http-v2"
    )
    if template_protocol != template_identity:
        raise LlamaCppGenerationBindingError("llama.cpp template identity and protocol disagree")
    if response.provider_identity.identity_id != request.provider_identity_id:
        raise LlamaCppGenerationBindingError("llama.cpp response targets another request")
    if metadata.request_sha256 != canonical_sha256(request):
        raise LlamaCppGenerationBindingError(
            "llama.cpp metadata does not bind the canonical request"
        )

    provider_request = _parse_protocol_json(metadata.provider_request_json, "request")
    completion_request = provider_request
    completion_prompt = request.prompt
    if template_protocol:
        completion_request, completion_prompt = _template_completion_request(
            provider_request, request, response.provider_identity
        )
    parameters = request.decoding_parameters
    expected_request: Mapping[str, object] = {
        "prompt": completion_prompt,
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
    if canonical_json(completion_request) != canonical_json(expected_request):
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
    if template_protocol:
        _require_template_completion_termination(provider_response, parameters.max_tokens)
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
    if template_protocol:
        _require_template_metadata_hashes(metadata)


def reconstruct_llama_cpp_generation_binding(
    *,
    condition: ExperimentCondition,
    decoding_parameters: DecodingParameters,
    provider_identity: ProviderIdentity,
    metadata: GenerationMetadata,
) -> tuple[GenerationRequest, GenerationResponse]:
    """Reconstruct typed domain objects from retained canonical wire payloads."""

    provider_request = _parse_protocol_json(metadata.provider_request_json, "request")
    prompt = (
        _template_input_prompt(provider_request)
        if metadata.generation_method == "llama_cpp_chat_template_http_v2"
        else provider_request.get("prompt")
    )
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

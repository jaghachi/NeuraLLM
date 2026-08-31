"""Storage-level regression tests for llama.cpp wire/domain cross-binding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurallm.domain.models import ActionBounds, ProviderIdentity, RunManifest, SeedSchedule
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import GenerationMetadata, GenerationRequest, GenerationResponse
from neurallm.providers.llama_cpp_evidence import (
    LlamaCppGenerationBindingError,
    reconstruct_llama_cpp_generation_binding,
    require_llama_cpp_generation_binding,
)
from neurallm.storage import CURRENT_SCHEMA_VERSION, SQLiteRunStore, StoreInvariantError
from tests.storage.helpers import make_request


def _identity(provider_type: str) -> tuple[ProviderIdentity, str]:
    effective = {"provider_type": provider_type, "wire-binding-test": True}
    return (
        ProviderIdentity(
            provider_type=provider_type,
            implementation_version=(
                "llama-cpp-completion-http-v1" if provider_type == "llama_cpp" else "test-v1"
            ),
            model_alias="wire-binding-model",
            build_id="wire-binding-build",
            provider_config_hash=canonical_sha256(effective),
            model_path=("C:/models/wire-binding.gguf" if provider_type == "llama_cpp" else None),
            model_sha256=("a" * 64 if provider_type == "llama_cpp" else None),
            chat_template_sha256=("b" * 64 if provider_type == "llama_cpp" else None),
        ),
        canonical_json(effective),
    )


def _manifest(identity: ProviderIdentity, effective_json: str) -> RunManifest:
    return RunManifest(
        source_commit="0" * 40,
        working_tree_clean=True,
        experiment_config_hash=canonical_sha256("wire-binding-config"),
        dataset_hash=canonical_sha256("wire-binding-dataset"),
        provider_config_hash=identity.provider_config_hash,
        provider_identity=identity,
        provider_effective_configuration_json=effective_json,
        policy_config_hashes={"test-policy": canonical_sha256("test-policy")},
        metric_versions={"test-metric": "v1"},
        seed_schedule=SeedSchedule(model_seeds=(7,), controller_seeds=(11,)),
        action_bounds=ActionBounds(),
        decision_rule_version="wire-binding-test-v1",
        database_schema_version=CURRENT_SCHEMA_VERSION,
    )


def _llama_metadata(
    request: GenerationRequest,
    request_payload: dict[str, object],
    response_payload: dict[str, object],
) -> GenerationMetadata:
    return GenerationMetadata(
        request_sha256=canonical_sha256(request),
        generation_method="llama_cpp_completion_http_v1",
        provider_request_json=canonical_json(request_payload),
        provider_request_sha256=canonical_sha256(request_payload),
        provider_response_json=canonical_json(response_payload),
        provider_response_sha256=canonical_sha256(response_payload),
    )


def _wire_payloads(
    request: GenerationRequest,
    identity: ProviderIdentity,
) -> tuple[dict[str, object], dict[str, object]]:
    parameters = request.decoding_parameters
    request_payload = {
        "prompt": request.prompt,
        "model": identity.model_alias,
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
        "top_k": parameters.top_k,
        "presence_penalty": parameters.presence_penalty,
        "n_predict": parameters.max_tokens,
        "seed": parameters.seed,
        "stream": False,
        "cache_prompt": False,
    }
    response_payload = {
        "content": "bound response",
        "stop": True,
        "model": identity.model_alias,
        "generation_settings": {
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "top_k": parameters.top_k,
            "presence_penalty": parameters.presence_penalty,
            "n_predict": parameters.max_tokens,
            "max_tokens": parameters.max_tokens,
            "seed": parameters.seed,
        },
    }
    return request_payload, response_payload


def _raw_bound_response(
    request: GenerationRequest,
    identity: ProviderIdentity,
    *,
    request_payload: dict[str, object] | None = None,
    response_payload: dict[str, object] | None = None,
    metadata_updates: dict[str, object] | None = None,
    response_identity: ProviderIdentity | None = None,
    effective_parameters: object | None = None,
    text: str = "bound response",
) -> GenerationResponse:
    resolved_request, resolved_response = _wire_payloads(request, identity)
    if request_payload is not None:
        resolved_request = request_payload
    if response_payload is not None:
        resolved_response = response_payload
    metadata = _llama_metadata(request, resolved_request, resolved_response)
    if metadata_updates:
        metadata = metadata.model_copy(update=metadata_updates)
    return GenerationResponse.model_construct(
        text=text,
        provider_identity=identity if response_identity is None else response_identity,
        effective_parameters=(
            request.decoding_parameters if effective_parameters is None else effective_parameters
        ),
        raw_metadata=metadata,
    )


def _persist_rejected(
    tmp_path: Path,
    identity: ProviderIdentity,
    effective_json: str,
    response: GenerationResponse,
) -> None:
    request = make_request(identity)
    with SQLiteRunStore(
        tmp_path / f"{canonical_sha256(response)[:12]}.sqlite3", _manifest(identity, effective_json)
    ) as store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        with pytest.raises(StoreInvariantError, match="llama_cpp wire evidence"):
            store.persist_response(request.condition_id, response)


def test_store_rejects_llama_identity_with_generic_generation_method(tmp_path: Path) -> None:
    identity, effective_json = _identity("llama_cpp")
    request = make_request(identity)
    response = GenerationResponse.model_construct(
        text="bound response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=GenerationMetadata(request_sha256=canonical_sha256(request)),
    )
    _persist_rejected(tmp_path, identity, effective_json, response)


def test_store_rejects_non_llama_identity_with_llama_generation_method(tmp_path: Path) -> None:
    identity, effective_json = _identity("fake")
    request = make_request(identity)
    request_payload, response_payload = _wire_payloads(request, identity)
    response = GenerationResponse.model_construct(
        text="bound response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=_llama_metadata(request, request_payload, response_payload),
    )
    _persist_rejected(tmp_path, identity, effective_json, response)


@pytest.mark.parametrize(
    "tamper",
    (
        "request_prompt",
        "response_content",
        "response_settings",
        "response_model",
        "response_stop",
        "response_budget",
    ),
)
def test_store_rejects_cross_object_llama_wire_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    identity, effective_json = _identity("llama_cpp")
    request = make_request(identity)
    request_payload, response_payload = _wire_payloads(request, identity)
    if tamper == "request_prompt":
        request_payload["prompt"] = "foreign prompt"
    elif tamper == "response_content":
        response_payload["content"] = "foreign response"
    elif tamper == "response_settings":
        settings = dict(response_payload["generation_settings"])  # type: ignore[arg-type]
        settings["top_k"] = request.decoding_parameters.top_k + 1
        response_payload["generation_settings"] = settings
    elif tamper == "response_model":
        response_payload["model"] = "foreign-model"
    elif tamper == "response_stop":
        response_payload["stop"] = False
    else:
        settings = dict(response_payload["generation_settings"])  # type: ignore[arg-type]
        settings["max_tokens"] = request.decoding_parameters.max_tokens + 1
        response_payload["generation_settings"] = settings
    response = GenerationResponse(
        text="bound response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=_llama_metadata(request, request_payload, response_payload),
    )
    _persist_rejected(tmp_path, identity, effective_json, response)


def test_binding_rejects_untyped_request_and_response() -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    response = _raw_bound_response(request, identity)
    with pytest.raises(TypeError, match="request must be a GenerationRequest"):
        require_llama_cpp_generation_binding(object(), response)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="response must be a GenerationResponse"):
        require_llama_cpp_generation_binding(request, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("metadata_updates", "message"),
    (
        ({"provider_request_json": None}, "request evidence is missing"),
        ({"provider_request_json": "{"}, "request evidence is invalid JSON"),
        ({"provider_request_json": "[]"}, "request evidence must be a JSON object"),
        ({"provider_response_json": None}, "response evidence is missing"),
    ),
)
def test_binding_rejects_missing_or_malformed_protocol_json(
    metadata_updates: dict[str, object],
    message: str,
) -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    response = _raw_bound_response(request, identity, metadata_updates=metadata_updates)
    with pytest.raises(LlamaCppGenerationBindingError, match=message):
        require_llama_cpp_generation_binding(request, response)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("temperature", "not-a-number", "temperature must be numeric"),
        ("top_p", float("inf"), "top_p must be finite"),
        ("top_k", True, "top_k must be an integer"),
    ),
)
def test_binding_rejects_malformed_generation_settings(
    field_name: str,
    value: object,
    message: str,
) -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    request_payload, response_payload = _wire_payloads(request, identity)
    settings = dict(response_payload["generation_settings"])  # type: ignore[arg-type]
    settings[field_name] = value
    response_payload["generation_settings"] = settings
    metadata = _llama_metadata(request, request_payload, _wire_payloads(request, identity)[1])
    metadata = metadata.model_copy(
        update={"provider_response_json": json.dumps(response_payload, separators=(",", ":"))}
    )
    response = GenerationResponse.model_construct(
        text="bound response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=metadata,
    )
    with pytest.raises(LlamaCppGenerationBindingError, match=message):
        require_llama_cpp_generation_binding(request, response)


def test_binding_rejects_non_object_generation_settings() -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    request_payload, response_payload = _wire_payloads(request, identity)
    response_payload["generation_settings"] = []
    response = _raw_bound_response(
        request,
        identity,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    with pytest.raises(LlamaCppGenerationBindingError, match="must be a JSON object"):
        require_llama_cpp_generation_binding(request, response)


def test_binding_rejects_response_identity_and_effective_parameter_drift() -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    foreign_identity = identity.model_copy(update={"build_id": "another-build"})
    wrong_identity_response = _raw_bound_response(
        request,
        identity,
        response_identity=foreign_identity,
    )
    with pytest.raises(LlamaCppGenerationBindingError, match="targets another request"):
        require_llama_cpp_generation_binding(request, wrong_identity_response)

    drifted_parameters = request.decoding_parameters.model_copy(
        update={"top_k": request.decoding_parameters.top_k + 1}
    )
    drifted_response = _raw_bound_response(
        request,
        identity,
        effective_parameters=drifted_parameters,
    )
    with pytest.raises(LlamaCppGenerationBindingError, match="GenerationResponse"):
        require_llama_cpp_generation_binding(request, drifted_response)


@pytest.mark.parametrize(
    ("prompt", "content", "message"),
    (
        ("", "bound response", "provider request prompt is invalid"),
        ("bound prompt", "   ", "provider response content is invalid"),
    ),
)
def test_reconstruction_rejects_invalid_prompt_or_content(
    prompt: str,
    content: str,
    message: str,
) -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    request_payload, response_payload = _wire_payloads(request, identity)
    request_payload["prompt"] = prompt
    response_payload["content"] = content
    metadata = _llama_metadata(request, request_payload, response_payload)
    with pytest.raises(LlamaCppGenerationBindingError, match=message):
        reconstruct_llama_cpp_generation_binding(
            condition=request.condition,
            decoding_parameters=request.decoding_parameters,
            provider_identity=identity,
            metadata=metadata,
        )

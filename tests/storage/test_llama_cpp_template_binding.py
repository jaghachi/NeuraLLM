"""Retained v2 template evidence remains bound to the original typed request."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from neurallm.domain.models import ProviderIdentity
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import GenerationMetadata, GenerationRequest, GenerationResponse
from neurallm.providers.llama_cpp_evidence import (
    LlamaCppGenerationBindingError,
    reconstruct_llama_cpp_generation_binding,
    require_llama_cpp_generation_binding,
)
from neurallm.providers.prompt_rendering import render_no_thinking_prompt
from neurallm.storage import SQLiteRunStore
from tests.storage.helpers import make_request
from tests.storage.test_llama_cpp_generation_binding import (
    _identity,
    _llama_metadata,
    _manifest,
    _persist_rejected,
    _wire_payloads,
)

_TEMPLATE = (
    Path(__file__)
    .parents[1]
    .joinpath("fixtures", "qwen35-chat-template.jinja")
    .read_text(encoding="utf-8")
    .removesuffix("\n")
)
_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"


def _template_identity() -> tuple[ProviderIdentity, str]:
    identity, effective_json = _identity("llama_cpp")
    return (
        identity.model_copy(
            update={
                "implementation_version": "llama-cpp-chat-template-http-v2",
                "chat_template_sha256": _TEMPLATE_SHA256,
            }
        ),
        effective_json,
    )


def _template_payloads(
    request: GenerationRequest, identity: ProviderIdentity
) -> tuple[dict[str, object], dict[str, object]]:
    completion, response = _wire_payloads(request, identity)
    rendered_prompt = render_no_thinking_prompt(request.prompt, _TEMPLATE_SHA256)
    completion["prompt"] = rendered_prompt
    response.update({"stop_type": "eos", "tokens_predicted": 3, "truncated": False})
    return (
        {
            "chat_template": _TEMPLATE,
            "template_request": {
                "model": identity.model_alias,
                "messages": [{"role": "user", "content": request.prompt}],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "template_response": {"prompt": rendered_prompt},
            "completion_request": completion,
        },
        response,
    )


def _template_metadata(
    request: GenerationRequest,
    request_payload: dict[str, object],
    response_payload: dict[str, object],
) -> GenerationMetadata:
    return GenerationMetadata(
        request_sha256=canonical_sha256(request),
        generation_method="llama_cpp_chat_template_http_v2",
        provider_request_json=canonical_json(request_payload),
        provider_request_sha256=canonical_sha256(request_payload),
        provider_response_json=canonical_json(response_payload),
        provider_response_sha256=canonical_sha256(response_payload),
    )


def _response(
    request: GenerationRequest,
    identity: ProviderIdentity,
    request_payload: dict[str, object],
    response_payload: dict[str, object],
) -> GenerationResponse:
    return GenerationResponse(
        text=str(response_payload["content"]),
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=_template_metadata(request, request_payload, response_payload),
    )


def test_captured_template_has_exact_pinned_hash() -> None:
    assert sha256(_TEMPLATE.encode("utf-8")).hexdigest() == _TEMPLATE_SHA256


@pytest.mark.parametrize("stop_type", ("eos", "word", "limit"))
def test_template_protocol_persists_and_reconstructs_original_prompt(
    tmp_path: Path, stop_type: str
) -> None:
    identity, effective_json = _template_identity()
    request = make_request(identity, prompt="  original prompt with boundary whitespace  ")
    request_payload, response_payload = _template_payloads(request, identity)
    response_payload["stop_type"] = stop_type
    response_payload["tokens_predicted"] = request.decoding_parameters.max_tokens
    response_payload["truncated"] = True
    # Retention never extracts an answer or strips model output, even with reasoning markers.
    response_payload["content"] = "  <think>unfinished reasoning and prompt terms  "
    response = _response(request, identity, request_payload, response_payload)
    with SQLiteRunStore(tmp_path / "v2.sqlite3", _manifest(identity, effective_json)) as store:
        store.prepare_turn(request)
        store.begin_dispatch(request.condition_id)
        store.persist_response(request.condition_id, response)
    reconstructed = reconstruct_llama_cpp_generation_binding(
        condition=request.condition,
        decoding_parameters=request.decoding_parameters,
        provider_identity=identity,
        metadata=response.raw_metadata,
    )
    assert reconstructed == (request, response)
    assert reconstructed[0].prompt == "  original prompt with boundary whitespace  "


@pytest.mark.parametrize(
    "tamper",
    (
        "template_text",
        "input_content",
        "input_role",
        "extra_role",
        "extra_message_key",
        "thinking_enabled",
        "thinking_integer",
        "extra_kwargs",
        "add_generation_prompt_integer",
        "extra_envelope_key",
        "missing_envelope_key",
        "extra_template_response_key",
        "rendered_prompt",
        "rendered_and_completion_prompt",
        "completion_prompt",
        "completion_stream_integer",
        "completion_seed_float",
    ),
)
def test_store_rejects_template_cross_binding_tampering(tmp_path: Path, tamper: str) -> None:
    identity, effective_json = _template_identity()
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    template_request = payload["template_request"]
    assert isinstance(template_request, dict)
    messages = template_request["messages"]
    assert isinstance(messages, list)
    kwargs = template_request["chat_template_kwargs"]
    assert isinstance(kwargs, dict)
    template_response = payload["template_response"]
    completion = payload["completion_request"]
    assert isinstance(template_response, dict)
    assert isinstance(completion, dict)
    if tamper == "template_text":
        payload["chat_template"] = _TEMPLATE + " "
    elif tamper == "input_content":
        messages[0]["content"] = "foreign original prompt"
    elif tamper == "input_role":
        messages[0]["role"] = "system"
    elif tamper == "extra_role":
        messages.append({"role": "assistant", "content": "prefill"})
    elif tamper == "extra_message_key":
        messages[0]["name"] = "unrecorded-user-name"
    elif tamper == "thinking_enabled":
        kwargs["enable_thinking"] = True
    elif tamper == "thinking_integer":
        kwargs["enable_thinking"] = 0
    elif tamper == "extra_kwargs":
        kwargs["hidden_setting"] = True
    elif tamper == "add_generation_prompt_integer":
        template_request["add_generation_prompt"] = 1
    elif tamper == "extra_envelope_key":
        payload["unbound_prompt"] = "other"
    elif tamper == "missing_envelope_key":
        del payload["chat_template"]
    elif tamper == "extra_template_response_key":
        template_response["other"] = "unbound"
    elif tamper == "rendered_prompt":
        template_response["prompt"] = "altered rendering"
    elif tamper == "rendered_and_completion_prompt":
        template_response["prompt"] = "altered rendering"
        completion["prompt"] = "altered rendering"
    elif tamper == "completion_prompt":
        completion["prompt"] = request.prompt
    elif tamper == "completion_stream_integer":
        completion["stream"] = 0
    else:
        completion["seed"] = float(request.decoding_parameters.seed)
    _persist_rejected(
        tmp_path, identity, effective_json, _response(request, identity, payload, response_payload)
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("stop_type", "unknown"),
        ("stop_type", None),
        ("tokens_predicted", -1),
        ("tokens_predicted", 129),
        ("tokens_predicted", True),
        ("tokens_predicted", 3.0),
        ("truncated", 0),
        ("truncated", None),
    ),
)
def test_store_rejects_invalid_template_termination(
    tmp_path: Path, key: str, value: object
) -> None:
    identity, effective_json = _template_identity()
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    response_payload[key] = value
    _persist_rejected(
        tmp_path, identity, effective_json, _response(request, identity, payload, response_payload)
    )


@pytest.mark.parametrize("key", ("stop_type", "tokens_predicted", "truncated"))
def test_store_rejects_missing_template_termination(tmp_path: Path, key: str) -> None:
    identity, effective_json = _template_identity()
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    del response_payload[key]
    _persist_rejected(
        tmp_path, identity, effective_json, _response(request, identity, payload, response_payload)
    )


def test_template_binding_rejects_unsupported_but_hash_bound_template() -> None:
    identity, _ = _template_identity()
    identity = identity.model_copy(update={"chat_template_sha256": sha256(b"other").hexdigest()})
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    payload["chat_template"] = "other"
    response = _response(request, identity, payload, response_payload)
    with pytest.raises(LlamaCppGenerationBindingError, match="chat template is unsupported"):
        require_llama_cpp_generation_binding(request, response)


@pytest.mark.parametrize("legacy_identity", (False, True))
def test_template_binding_rejects_wrong_protocol_identity(legacy_identity: bool) -> None:
    identity, _ = _template_identity()
    if legacy_identity:
        identity = identity.model_copy(
            update={"implementation_version": "llama-cpp-completion-http-v1"}
        )
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    metadata = _template_metadata(request, payload, response_payload)
    if not legacy_identity:
        metadata = metadata.model_copy(update={"generation_method": "llama_cpp_completion_http_v1"})
    response = GenerationResponse.model_construct(
        text="bound response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=metadata,
    )
    with pytest.raises(LlamaCppGenerationBindingError, match="template identity and protocol"):
        require_llama_cpp_generation_binding(request, response)


@pytest.mark.parametrize(
    "tamper", ("request_hash", "response_hash", "request_noncanonical", "logical_request_hash")
)
def test_template_binding_revalidates_hashes_for_unvalidated_models(tamper: str) -> None:
    identity, _ = _template_identity()
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    metadata = _template_metadata(request, payload, response_payload)
    if tamper == "request_hash":
        metadata = metadata.model_copy(update={"provider_request_sha256": "0" * 64})
    elif tamper == "response_hash":
        metadata = metadata.model_copy(update={"provider_response_sha256": "0" * 64})
    elif tamper == "request_noncanonical":
        metadata = metadata.model_copy(update={"provider_request_json": json.dumps(payload)})
    else:
        metadata = metadata.model_copy(update={"request_sha256": "0" * 64})
    response = GenerationResponse.model_construct(
        text="bound response",
        provider_identity=identity,
        effective_parameters=request.decoding_parameters,
        raw_metadata=metadata,
    )
    with pytest.raises(LlamaCppGenerationBindingError, match="canonical"):
        require_llama_cpp_generation_binding(request, response)


@pytest.mark.parametrize("messages", ([], [{"role": "user", "content": ""}], ["invalid"]))
def test_template_reconstruction_rejects_malformed_input(messages: list[object]) -> None:
    identity, _ = _template_identity()
    request = make_request(identity)
    payload, response_payload = _template_payloads(request, identity)
    template_request = payload["template_request"]
    assert isinstance(template_request, dict)
    template_request["messages"] = messages
    with pytest.raises(LlamaCppGenerationBindingError, match="template"):
        reconstruct_llama_cpp_generation_binding(
            condition=request.condition,
            decoding_parameters=request.decoding_parameters,
            provider_identity=identity,
            metadata=_template_metadata(request, payload, response_payload),
        )


def test_legacy_protocol_still_reconstructs_raw_prompt_without_termination_fields() -> None:
    identity, _ = _identity("llama_cpp")
    request = make_request(identity)
    payload, response_payload = _wire_payloads(request, identity)
    metadata = _llama_metadata(request, payload, response_payload)
    reconstructed_request, response = reconstruct_llama_cpp_generation_binding(
        condition=request.condition,
        decoding_parameters=request.decoding_parameters,
        provider_identity=identity,
        metadata=metadata,
    )
    assert reconstructed_request == request
    assert response.text == "bound response"
    assert response.raw_metadata == metadata

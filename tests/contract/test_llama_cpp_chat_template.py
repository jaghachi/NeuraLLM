"""Offline regressions for the versioned, template-aware llama.cpp protocol."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from neurallm.domain.models import DecodingParameters
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers import (
    LlamaCppProtocolError,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    preflight_llama_cpp,
    require_llama_cpp_generation_binding,
    require_llama_cpp_provider_binding,
)
from neurallm.providers.prompt_rendering import (
    QWEN35_NO_THINKING_TEMPLATE_SHA256,
    render_no_thinking_prompt,
)
from tests.contract.test_llama_cpp_provider import (
    _completion,
    _config,
    _parameters,
    _props,
    _RecordingHandler,
    _request,
)

# The server source has no final LF; the text fixture has one file terminator.
_QWEN_TEMPLATE = (
    (Path(__file__).parents[1] / "fixtures" / "qwen35-chat-template.jinja")
    .read_text(encoding="utf-8")
    .removesuffix("\n")
)


def _chat_config() -> LlamaCppProviderConfig:
    return _config(
        prompt_format="chat_template_no_thinking_v1",
        chat_template_sha256=QWEN35_NO_THINKING_TEMPLATE_SHA256,
    )


class _ChatHandler(_RecordingHandler):
    def __init__(
        self, *, stop_type: str = "eos", content: str = "A deterministic completion."
    ) -> None:
        super().__init__(
            props_factory=lambda _: _props(chat_template=_QWEN_TEMPLATE),
            completion_factory=lambda parameters: httpx.Response(
                200,
                json={
                    **_completion(parameters, content=content),
                    "stop_type": stop_type,
                    "truncated": False,
                    "tokens_predicted": parameters.max_tokens if stop_type == "limit" else 4,
                },
            ),
        )
        self.template_payloads: list[dict[str, object]] = []
        self.template_override: dict[str, object] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/completion":
            self.requests.append(request)
            payload = json.loads(request.content)
            self.completion_payloads.append(payload)
            parameters = DecodingParameters(
                temperature=payload["temperature"],
                top_p=payload["top_p"],
                top_k=payload["top_k"],
                presence_penalty=payload["presence_penalty"],
                max_tokens=payload["n_predict"],
                seed=payload["seed"],
            )
            assert self._completion_factory is not None
            return self._completion_factory(parameters)
        if request.url.path == "/apply-template":
            self.requests.append(request)
            payload = json.loads(request.content)
            self.template_payloads.append(payload)
            rendered = render_no_thinking_prompt(
                payload["messages"][0]["content"], QWEN35_NO_THINKING_TEMPLATE_SHA256
            )
            return httpx.Response(
                200,
                json=(
                    {"prompt": rendered}
                    if self.template_override is None
                    else self.template_override
                ),
            )
        return super().__call__(request)


def test_audited_template_fixture_and_exact_whitespace() -> None:
    assert sha256(_QWEN_TEMPLATE.encode()).hexdigest() == QWEN35_NO_THINKING_TEMPLATE_SHA256
    assert render_no_thinking_prompt("  Say ready.\n", QWEN35_NO_THINKING_TEMPLATE_SHA256) == (
        "<|im_start|>user\nSay ready.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize("stop_type", ["eos", "word", "limit"])
def test_chat_template_is_applied_once_and_wire_evidence_is_retained(stop_type: str) -> None:
    handler = _ChatHandler(stop_type=stop_type)
    with LlamaCppProvider(_chat_config(), transport=httpx.MockTransport(handler)) as provider:
        request = _request(provider)
        response = provider.generate(request)
        require_llama_cpp_generation_binding(request, response)
        require_llama_cpp_provider_binding(
            provider.provider_identity, provider.effective_configuration_json
        )
    assert response.raw_metadata.generation_method == "llama_cpp_chat_template_http_v2"
    assert response.provider_identity.implementation_version == "llama-cpp-chat-template-http-v2"
    assert len(handler.template_payloads) == 2  # Construction probe, then the logical request.
    assert len(handler.completion_payloads) == 1
    assert handler.template_payloads[1] == {
        "model": response.provider_identity.model_alias,
        "messages": [{"role": "user", "content": request.prompt}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    metadata = response.raw_metadata
    evidence = json.loads(metadata.provider_request_json or "{}")
    assert evidence["chat_template"] == _QWEN_TEMPLATE
    assert evidence["template_request"] == handler.template_payloads[1]
    assert evidence["completion_request"] == handler.completion_payloads[0]
    assert evidence["completion_request"]["prompt"] == evidence["template_response"]["prompt"]
    assert metadata.request_sha256 == canonical_sha256(request)
    assert json.loads(metadata.provider_response_json or "{}")["stop_type"] == stop_type
    assert response.text == "A deterministic completion."
    assert response.effective_parameters == _parameters()


@pytest.mark.parametrize(
    "bad_response",
    [
        {},
        {"prompt": ""},
        {"prompt": 7},
        {"prompt": "unformatted prompt"},
        {"prompt": "<|im_start|>assistant\n<think>\n"},
        {"prompt": "unexpected", "extra": True},
    ],
)
def test_invalid_template_result_never_dispatches_completion(
    bad_response: dict[str, object],
) -> None:
    handler = _ChatHandler()
    with LlamaCppProvider(_chat_config(), transport=httpx.MockTransport(handler)) as provider:
        handler.template_override = bad_response
        with pytest.raises(LlamaCppProtocolError, match="/apply-template"):
            provider.generate(_request(provider))
    assert len(handler.template_payloads) == 2
    assert handler.completion_payloads == []


def test_legacy_config_bytes_and_identity_are_preserved_and_v2_is_distinct() -> None:
    legacy_config = _config(chat_template_sha256=QWEN35_NO_THINKING_TEMPLATE_SHA256)
    assert "prompt_format" not in legacy_config.model_dump()
    assert canonical_json(legacy_config) == canonical_json(
        _config(
            chat_template_sha256=QWEN35_NO_THINKING_TEMPLATE_SHA256,
            prompt_format="raw_completion_v1",
        )
    )
    with LlamaCppProvider(legacy_config, transport=httpx.MockTransport(_ChatHandler())) as legacy:
        legacy_identity = legacy.provider_identity
        legacy_json = legacy.effective_configuration_json
    with LlamaCppProvider(_chat_config(), transport=httpx.MockTransport(_ChatHandler())) as current:
        assert current.provider_identity.identity_id != legacy_identity.identity_id
        assert (
            current.provider_identity.provider_config_hash != legacy_identity.provider_config_hash
        )
        assert legacy_identity.implementation_version == "llama-cpp-completion-http-v1"
    restored = require_llama_cpp_provider_binding(legacy_identity, legacy_json)
    assert canonical_json(restored) == legacy_json


@pytest.mark.parametrize("prompt", [" ", "<tool_response>x</tool_response>"])
def test_unsupported_prompt_rejected_without_template_or_completion_request(prompt: str) -> None:
    handler = _ChatHandler()
    with LlamaCppProvider(_chat_config(), transport=httpx.MockTransport(handler)) as provider:
        request = _request(provider).model_copy(update={"prompt": prompt})
        with pytest.raises(LlamaCppProtocolError):
            provider.generate(request)
    assert len(handler.template_payloads) == 1  # Only the construction probe.
    assert handler.completion_payloads == []


def test_unsupported_template_is_not_silently_treated_as_qwen() -> None:
    with pytest.raises(ValueError, match="audited Qwen3.5"):
        render_no_thinking_prompt("hello", "0" * 64)


def test_v2_preflight_proves_rendering_without_completion() -> None:
    handler = _ChatHandler()
    result = preflight_llama_cpp(_chat_config(), transport=httpx.MockTransport(handler))
    assert result.completion_requested is False
    assert len(handler.template_payloads) == 1
    assert handler.completion_payloads == []


def test_v2_preflight_rejects_wrong_renderer_before_any_generation() -> None:
    handler = _ChatHandler()
    handler.template_override = {"prompt": "raw prompt with reasoning enabled"}
    with pytest.raises(LlamaCppProtocolError, match="audited no-thinking"):
        preflight_llama_cpp(_chat_config(), transport=httpx.MockTransport(handler))
    assert len(handler.template_payloads) == 1
    assert handler.completion_payloads == []

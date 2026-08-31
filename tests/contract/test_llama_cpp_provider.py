"""Zero-network contract tests for the strict llama.cpp HTTP adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from neurallm.domain.models import DecodingParameters, ExperimentCondition
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers import (
    GenerationProvider,
    GenerationRequest,
    LlamaCppEffectiveConfiguration,
    LlamaCppIdentityDriftError,
    LlamaCppPreflightResult,
    LlamaCppProtocolError,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    LlamaCppTransportError,
    ProviderIdentityMismatchError,
    llama_cpp_provider_identity,
    preflight_llama_cpp,
)

_CHAT_TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}"
_MODEL_PATH = "C:/models/neurallm-test.gguf"
_BUILD_ID = "b5000-deadbeef"
_MODEL_ALIAS = "neurallm-test"


def _template_sha256(template: str = _CHAT_TEMPLATE) -> str:
    return sha256(template.encode("utf-8")).hexdigest()


def _config(**changes: object) -> LlamaCppProviderConfig:
    values: dict[str, object] = {
        "base_url": "http://127.0.0.1:8080",
        "model_alias": _MODEL_ALIAS,
        "model_path": _MODEL_PATH,
        "build_id": _BUILD_ID,
        "chat_template_sha256": _template_sha256(),
        "connect_timeout_seconds": 1.25,
        "read_timeout_seconds": 31.5,
        "write_timeout_seconds": 2.5,
        "pool_timeout_seconds": 3.75,
    }
    values.update(changes)
    return LlamaCppProviderConfig.model_validate(values)


def _defaults(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "presence_penalty": 0.0,
        "n_predict": -1,
        "max_tokens": -1,
        "seed": 4294967295,
        "stream": True,
        "cache_prompt": True,
    }
    values.update(changes)
    return values


def _props(
    *,
    model_path: str = _MODEL_PATH,
    build_id: str = _BUILD_ID,
    chat_template: str = _CHAT_TEMPLATE,
    defaults: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "default_generation_settings": {
            "id": 0,
            "is_processing": False,
            "params": dict(defaults or _defaults()),
        },
        "total_slots": 1,
        "model_path": model_path,
        "chat_template": chat_template,
        "chat_template_caps": {},
        "modalities": {"vision": False},
        "build_info": build_id,
        "is_sleeping": False,
    }


def _completion(
    parameters: DecodingParameters,
    *,
    content: object = "A deterministic completion.",
    model_alias: object = _MODEL_ALIAS,
    stop: object = True,
    settings_changes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    settings: dict[str, object] = {
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
        "top_k": parameters.top_k,
        "presence_penalty": parameters.presence_penalty,
        "n_predict": parameters.max_tokens,
        "max_tokens": parameters.max_tokens,
        "seed": parameters.seed,
    }
    settings.update(settings_changes or {})
    return {
        "content": content,
        "stop": stop,
        "model": model_alias,
        "generation_settings": settings,
        "tokens_predicted": 4,
    }


def _parameters() -> DecodingParameters:
    return DecodingParameters(
        temperature=0.7,
        top_p=0.9,
        top_k=31,
        presence_penalty=0.15,
        max_tokens=96,
        seed=17,
    )


def _request(provider: LlamaCppProvider, *, identity_id: str | None = None) -> GenerationRequest:
    provider_identity_id = identity_id or provider.provider_identity.identity_id
    return GenerationRequest(
        prompt="Explain the causal ordering of one experiment turn.",
        decoding_parameters=_parameters(),
        condition=ExperimentCondition(
            experiment_id="phase-2-provider-contract",
            dataset_version="development-v1",
            prompt_sequence_id="sequence-001",
            turn_index=0,
            policy_id="static-test",
            model_seed=17,
            controller_seed=23,
            provider_identity_id=provider_identity_id,
            base_decoding_profile_id="base-v1",
        ),
    )


class _RecordingHandler:
    def __init__(
        self,
        completion_factory: Callable[[DecodingParameters], httpx.Response] | None = None,
        *,
        props_factory: Callable[[int], dict[str, object]] | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.completion_payloads: list[dict[str, object]] = []
        self._completion_factory = completion_factory
        self._props_factory = props_factory
        self._props_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            self._props_calls += 1
            payload = self._props_factory(self._props_calls) if self._props_factory else _props()
            return httpx.Response(200, json=payload)
        if request.url.path == "/completion":
            payload_value: Any = json.loads(request.content)
            assert isinstance(payload_value, dict)
            self.completion_payloads.append(payload_value)
            if self._completion_factory is not None:
                return self._completion_factory(_parameters())
            return httpx.Response(200, json=_completion(_parameters()))
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    @property
    def completion_dispatch_count(self) -> int:
        return sum(request.url.path == "/completion" for request in self.requests)


def _provider(handler: Callable[[httpx.Request], httpx.Response]) -> LlamaCppProvider:
    return LlamaCppProvider(_config(), transport=httpx.MockTransport(handler))


def test_preflight_inspects_identity_only_and_never_dispatches_completion() -> None:
    handler = _RecordingHandler()

    result = preflight_llama_cpp(
        _config(),
        transport=httpx.MockTransport(handler),
    )

    assert [(request.method, request.url.path) for request in handler.requests] == [
        ("GET", "/health"),
        ("GET", "/props"),
    ]
    assert handler.completion_dispatch_count == 0
    assert result.provider_kind == "llama_cpp"
    assert result.completion_requested is False
    assert result.provider_identity_id == result.expected_identity.identity_id
    effective = LlamaCppEffectiveConfiguration.model_validate_json(
        result.expected_effective_configuration_json
    )
    assert result.expected_identity == llama_cpp_provider_identity(effective)
    assert canonical_json(result) == canonical_json(result.model_copy(deep=True))


def test_preflight_result_rejects_noncanonical_or_internally_inconsistent_evidence() -> None:
    result = preflight_llama_cpp(
        _config(),
        transport=httpx.MockTransport(_RecordingHandler()),
    )
    payload = result.model_dump(mode="python")

    effective = json.loads(result.expected_effective_configuration_json)
    with pytest.raises(ValidationError, match="must be canonical JSON"):
        LlamaCppPreflightResult.model_validate(
            {
                **payload,
                "expected_effective_configuration_json": json.dumps(effective, indent=2),
            }
        )

    with pytest.raises(ValidationError, match="disagrees with effective configuration"):
        LlamaCppPreflightResult.model_validate(
            {
                **payload,
                "expected_identity": result.expected_identity.model_copy(
                    update={"build_id": "different-build"}
                ),
            }
        )

    with pytest.raises(ValidationError, match="identity ID disagrees"):
        LlamaCppPreflightResult.model_validate({**payload, "provider_identity_id": "f" * 64})


def test_preflight_rejects_an_untyped_provider_configuration_before_io() -> None:
    with pytest.raises(TypeError, match="LlamaCppProviderConfig"):
        preflight_llama_cpp(object())  # type: ignore[arg-type]


def test_configuration_is_explicit_strict_and_validated() -> None:
    config = _config(base_url="http://127.0.0.1:8080/")

    assert config.base_url == "http://127.0.0.1:8080"
    with pytest.raises(ValidationError, match="connect_timeout_seconds"):
        _config(connect_timeout_seconds=0.0)
    with pytest.raises(ValidationError, match="base_url"):
        _config(base_url="localhost:8080")
    with pytest.raises(ValidationError, match="chat_template_sha256"):
        _config(chat_template_sha256="not-a-hash")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LlamaCppProviderConfig.model_validate(
            {**config.model_dump(), "environment_fallback": "forbidden"}
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_url", " http://127.0.0.1:8080", "surrounding whitespace"),
        ("base_url", "http://user:secret@127.0.0.1:8080", "credentials"),
        ("base_url", "http://127.0.0.1:8080?model=other", "query or fragment"),
        ("model_alias", " ", "must not be blank"),
    ],
)
def test_configuration_rejects_ambiguous_url_and_identity_inputs(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**{field: value})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_alias": "other"}, "must match the explicit client config"),
        ({"chat_template": "tampered"}, "template hash does not match"),
        ({"default_generation_settings_json": "{}"}, "finite canonical JSON"),
        (
            {"default_generation_settings_json": '{ "temperature": 0.7 }'},
            "canonical JSON",
        ),
    ],
)
def test_effective_configuration_rejects_unproducible_preflight_evidence(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "client_config": _config(),
        "model_alias": _MODEL_ALIAS,
        "model_path": _MODEL_PATH,
        "build_id": _BUILD_ID,
        "chat_template": _CHAT_TEMPLATE,
        "chat_template_sha256": _template_sha256(),
        "default_generation_settings_json": canonical_json(_defaults()),
        "total_slots": 1,
    }
    values.update(changes)

    with pytest.raises((ValidationError, ValueError), match=message):
        LlamaCppEffectiveConfiguration.model_validate(values)


def test_preflight_binds_exact_props_identity_and_explicit_timeouts() -> None:
    handler = _RecordingHandler()
    provider = _provider(handler)

    assert isinstance(provider, GenerationProvider)
    assert provider.provider_identity.provider_type == "llama_cpp"
    assert provider.provider_identity.model_alias == _MODEL_ALIAS
    assert provider.provider_identity.model_path == _MODEL_PATH
    assert provider.provider_identity.build_id == _BUILD_ID
    assert provider.provider_identity.chat_template_sha256 == _template_sha256()
    assert provider.provider_identity.provider_config_hash == canonical_sha256(
        provider.effective_configuration
    )
    assert provider.effective_configuration.chat_template == _CHAT_TEMPLATE
    assert json.loads(provider.effective_configuration.default_generation_settings_json) == (
        _defaults()
    )
    assert [request.url.path for request in handler.requests] == ["/health", "/props"]
    for request in handler.requests:
        assert request.extensions["timeout"] == {
            "connect": 1.25,
            "read": 31.5,
            "write": 2.5,
            "pool": 3.75,
        }


def test_provider_context_manager_closes_after_a_bound_preflight() -> None:
    handler = _RecordingHandler()

    with _provider(handler) as provider:
        assert provider.provider_identity.model_alias == _MODEL_ALIAS

    provider.close()


def test_generate_transmits_and_validates_all_controlled_settings_once() -> None:
    handler = _RecordingHandler()
    provider = _provider(handler)
    request = _request(provider)

    response = provider.generate(request)

    assert [request.url.path for request in handler.requests] == [
        "/health",
        "/props",
        "/health",
        "/props",
        "/completion",
    ]
    assert handler.completion_dispatch_count == 1
    assert handler.completion_payloads == [
        {
            "prompt": request.prompt,
            "model": _MODEL_ALIAS,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 31,
            "presence_penalty": 0.15,
            "n_predict": 96,
            "seed": 17,
            "stream": False,
            "cache_prompt": False,
        }
    ]
    assert response.text == "A deterministic completion."
    assert response.effective_parameters == request.decoding_parameters
    assert response.provider_identity == provider.provider_identity
    assert response.raw_metadata.request_sha256 == canonical_sha256(request)
    assert response.raw_metadata.generation_method == "llama_cpp_completion_http_v1"
    assert response.raw_metadata.provider_request_json == canonical_json(
        handler.completion_payloads[0]
    )
    assert response.raw_metadata.provider_request_sha256 == canonical_sha256(
        handler.completion_payloads[0]
    )
    assert response.raw_metadata.provider_response_json == canonical_json(
        _completion(_parameters())
    )
    assert response.raw_metadata.provider_response_sha256 == canonical_sha256(
        _completion(_parameters())
    )
    assert provider.last_raw_response_sha256 == canonical_sha256(_completion(_parameters()))


def test_float32_echo_rounding_is_recorded_within_a_narrow_tolerance() -> None:
    rounded = {
        "temperature": 0.699999988079071,
        "top_p": 0.8999999761581421,
        "presence_penalty": 0.15000000596046448,
    }
    handler = _RecordingHandler(
        lambda parameters: httpx.Response(
            200,
            json=_completion(parameters, settings_changes=rounded),
        )
    )
    provider = _provider(handler)

    response = provider.generate(_request(provider))

    assert response.effective_parameters.temperature == rounded["temperature"]
    assert response.effective_parameters.top_p == rounded["top_p"]
    assert response.effective_parameters.presence_penalty == rounded["presence_penalty"]


def test_tolerated_echo_still_must_satisfy_the_domain_parameter_contract() -> None:
    boundary_parameters = _parameters().model_copy(update={"top_p": 1.0})
    handler = _RecordingHandler(
        lambda _: httpx.Response(
            200,
            json=_completion(
                boundary_parameters,
                settings_changes={"top_p": 1.0000001},
            ),
        )
    )
    provider = _provider(handler)
    request = _request(provider).model_copy(update={"decoding_parameters": boundary_parameters})

    with pytest.raises(LlamaCppProtocolError, match="decoding-parameter contract"):
        provider.generate(request)

    assert handler.completion_dispatch_count == 1


def test_request_identity_mismatch_fails_before_any_generation_io() -> None:
    handler = _RecordingHandler()
    provider = _provider(handler)
    calls_after_construction = len(handler.requests)

    with pytest.raises(ProviderIdentityMismatchError, match="does not match"):
        provider.generate(_request(provider, identity_id="f" * 64))

    assert len(handler.requests) == calls_after_construction
    assert handler.completion_dispatch_count == 0


def test_identity_drift_fails_before_completion_dispatch() -> None:
    def props_factory(call: int) -> dict[str, object]:
        return _props(defaults=_defaults(top_k=40 if call == 1 else 41))

    handler = _RecordingHandler(props_factory=props_factory)
    provider = _provider(handler)

    with pytest.raises(LlamaCppIdentityDriftError, match="drifted before dispatch"):
        provider.generate(_request(provider))

    assert handler.completion_dispatch_count == 0


@pytest.mark.parametrize(
    ("props_payload", "message"),
    [
        ({"status": "not-props"}, "model_path"),
        (_props(defaults=_defaults(top_p="0.9")), "top_p"),
        (_props(chat_template=""), "chat_template"),
    ],
)
def test_malformed_or_missing_props_fail_closed(
    props_payload: dict[str, object],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=props_payload)

    with pytest.raises(LlamaCppProtocolError, match=message):
        _provider(handler)


@pytest.mark.parametrize(
    ("props_payload", "message"),
    [
        (_props(model_path="C:/models/other.gguf"), "model_path"),
        (_props(build_id="b5001-other"), "build_info"),
        (_props(chat_template="{{ different }}"), "chat_template"),
        ({**_props(), "total_slots": 0}, "total_slots"),
        (
            {**_props(), "default_generation_settings": "not-an-object"},
            "default_generation_settings",
        ),
        (
            _props(defaults={key: value for key, value in _defaults().items() if key != "seed"}),
            "missing: seed",
        ),
    ],
)
def test_preflight_rejects_config_mismatch_and_incomplete_effective_settings(
    props_payload: dict[str, object],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=props_payload)

    with pytest.raises((LlamaCppProtocolError, LlamaCppIdentityDriftError), match=message):
        _provider(handler)


@pytest.mark.parametrize(
    ("completion_changes", "message"),
    [
        ({"content": ""}, "content"),
        ({"stop": False}, "completed non-streaming"),
        ({"model_alias": "other-model"}, "model alias"),
        ({"settings_changes": {"temperature": 0.5}}, "temperature"),
        ({"settings_changes": {"top_p": 0.5}}, "top_p"),
        ({"settings_changes": {"top_k": 12}}, "top_k"),
        ({"settings_changes": {"presence_penalty": 0.5}}, "presence_penalty"),
        ({"settings_changes": {"n_predict": 95}}, "generation budget"),
        ({"settings_changes": {"max_tokens": 95}}, "generation budget"),
        ({"settings_changes": {"seed": 18}}, "seed"),
        ({"settings_changes": {"seed": "17"}}, "seed"),
    ],
)
def test_malformed_or_mismatched_completion_fails_closed_after_one_dispatch(
    completion_changes: dict[str, object],
    message: str,
) -> None:
    handler = _RecordingHandler(
        lambda parameters: httpx.Response(
            200,
            json=_completion(parameters, **completion_changes),
        )
    )
    provider = _provider(handler)

    with pytest.raises((LlamaCppProtocolError, LlamaCppIdentityDriftError), match=message):
        provider.generate(_request(provider))

    assert handler.completion_dispatch_count == 1
    assert provider.last_raw_response_sha256 is not None


def test_http_failure_is_not_retried_or_routed_to_a_fallback() -> None:
    handler = _RecordingHandler(lambda _: httpx.Response(503, json={"error": "busy"}))
    provider = _provider(handler)

    with pytest.raises(LlamaCppTransportError, match="no retry was attempted"):
        provider.generate(_request(provider))

    assert handler.completion_dispatch_count == 1
    assert [request.url.path for request in handler.requests].count("/completion") == 1


def test_read_timeout_has_one_completion_attempt_and_no_retry_or_fallback() -> None:
    def read_timeout(_: DecodingParameters) -> httpx.Response:
        raise httpx.ReadTimeout("completion read timed out")

    handler = _RecordingHandler(read_timeout)
    provider = _provider(handler)

    with pytest.raises(LlamaCppTransportError, match="no retry was attempted"):
        provider.generate(_request(provider))

    assert [request.url.path for request in handler.requests] == [
        "/health",
        "/props",
        "/health",
        "/props",
        "/completion",
    ]
    assert handler.completion_dispatch_count == 1
    assert len(handler.completion_payloads) == 1


def test_connection_error_invalid_json_and_non_object_json_fail_closed() -> None:
    def connection_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LlamaCppTransportError, match="no retry was attempted"):
        _provider(connection_error)

    def invalid_json(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, content=b"{", headers={"content-type": "application/json"})

    with pytest.raises(LlamaCppProtocolError, match="not valid JSON"):
        _provider(invalid_json)

    def non_object_json(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=[{"status": "ok"}])

    with pytest.raises(LlamaCppProtocolError, match="JSON object"):
        _provider(non_object_json)


def test_malformed_health_and_nonfinite_payloads_fail_closed() -> None:
    def malformed_health(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "loading"})

    with pytest.raises(LlamaCppProtocolError, match="exactly"):
        _provider(malformed_health)

    def nonfinite_props(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        payload = _props(defaults=_defaults(temperature=float("nan")))
        return httpx.Response(
            200,
            content=json.dumps(payload, allow_nan=True).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    with pytest.raises(LlamaCppProtocolError, match="canonical finite JSON"):
        _provider(nonfinite_props)


@pytest.mark.parametrize("invalid_seed", [-1, 0xFFFFFFFF])
def test_nondeterministic_or_out_of_range_seed_fails_before_dispatch(invalid_seed: int) -> None:
    handler = _RecordingHandler()
    provider = _provider(handler)
    request = _request(provider).model_copy(
        update={
            "decoding_parameters": _parameters().model_copy(update={"seed": invalid_seed}),
        }
    )

    with pytest.raises(LlamaCppProtocolError, match="deterministic seed"):
        provider.generate(request)

    assert handler.completion_dispatch_count == 0

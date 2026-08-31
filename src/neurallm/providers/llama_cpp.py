"""Strict, identity-bound adapter for the llama.cpp HTTP completion server."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
from typing import Final, Self, cast
from urllib.parse import urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from neurallm.domain.models import DecodingParameters, ProviderIdentity, Sha256Hex
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.providers.base import (
    DECODING_FLOAT_ABSOLUTE_TOLERANCE,
    DECODING_FLOAT_RELATIVE_TOLERANCE,
    GenerationMetadata,
    GenerationRequest,
    GenerationResponse,
    ProviderIdentityMismatchError,
)

LLAMA_CPP_IMPLEMENTATION_VERSION: Final = "llama-cpp-completion-http-v1"
_REQUIRED_EFFECTIVE_SETTINGS: Final = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "n_predict",
    "seed",
)


class LlamaCppProviderError(RuntimeError):
    """Base class for fail-closed llama.cpp provider failures."""


class LlamaCppTransportError(LlamaCppProviderError):
    """Raised when a single HTTP operation cannot complete successfully."""


class LlamaCppProtocolError(LlamaCppProviderError):
    """Raised when llama.cpp returns a malformed or inconsistent payload."""


class LlamaCppIdentityDriftError(LlamaCppProviderError):
    """Raised when inspected server identity differs from the bound identity."""


class _StrictFrozenModel(BaseModel):
    """Immutable, strict base for provider configuration and observations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class LlamaCppProviderConfig(_StrictFrozenModel):
    """Complete explicit configuration for one llama.cpp server identity.

    No field is read from the process environment. The expected server identity
    must be known before construction and is checked against ``/props``.
    """

    base_url: str
    model_alias: str
    model_path: str
    model_sha256: Sha256Hex
    build_id: str
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    connect_timeout_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    read_timeout_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    write_timeout_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    pool_timeout_seconds: float = Field(gt=0.0, allow_inf_nan=False)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("base_url must be non-blank and contain no surrounding whitespace")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if "?" in value or "#" in value or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        return value.rstrip("/")

    @field_validator("model_alias", "build_id")
    @classmethod
    def _validate_non_blank_identity_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity fields must not be blank")
        return value

    @field_validator("model_path")
    @classmethod
    def _validate_absolute_model_path(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("model_path must be non-blank with no surrounding whitespace")
        if not Path(value).is_absolute():
            raise ValueError(
                "model_path must be an absolute client-local path with no user expansion"
            )
        return value

    @property
    def timeout(self) -> httpx.Timeout:
        """Return the four explicit HTTP timeout components."""

        return httpx.Timeout(
            connect=self.connect_timeout_seconds,
            read=self.read_timeout_seconds,
            write=self.write_timeout_seconds,
            pool=self.pool_timeout_seconds,
        )


class LlamaCppEffectiveConfiguration(_StrictFrozenModel):
    """Static server properties inspected and bound before generation."""

    client_config: LlamaCppProviderConfig
    model_alias: str
    model_path: str
    model_sha256: Sha256Hex
    build_id: str
    chat_template: str
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    default_generation_settings_json: str
    total_slots: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> Self:
        if (
            self.model_alias != self.client_config.model_alias
            or self.model_path != self.client_config.model_path
            or self.model_sha256 != self.client_config.model_sha256
            or self.build_id != self.client_config.build_id
            or self.chat_template_sha256 != self.client_config.chat_template_sha256
        ):
            raise ValueError("effective identity fields must match the explicit client config")
        if _raw_text_sha256(self.chat_template) != self.chat_template_sha256:
            raise ValueError("effective chat template hash does not match the template text")
        try:
            settings: object = json.loads(self.default_generation_settings_json)
            if not isinstance(settings, dict) or not all(isinstance(key, str) for key in settings):
                raise ValueError("effective default settings must be a JSON object")
            if canonical_json(settings) != self.default_generation_settings_json:
                raise ValueError("effective default settings must be canonical JSON")
            _validate_default_settings_values(cast(dict[str, object], settings))
        except (json.JSONDecodeError, TypeError, LlamaCppProtocolError) as exc:
            raise ValueError("effective default settings must be finite canonical JSON") from exc
        return self


def _require_mapping(
    container: Mapping[str, object],
    key: str,
    *,
    endpoint: str,
) -> Mapping[str, object]:
    value = container.get(key)
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value):
        raise LlamaCppProtocolError(f"{endpoint} field {key!r} must be a JSON object")
    return cast(dict[str, object], value)


def _require_non_blank_string(
    container: Mapping[str, object],
    key: str,
    *,
    endpoint: str,
) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LlamaCppProtocolError(f"{endpoint} field {key!r} must be a non-blank string")
    return value


def _require_int(
    container: Mapping[str, object],
    key: str,
    *,
    endpoint: str,
) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LlamaCppProtocolError(f"{endpoint} field {key!r} must be an integer")
    return value


def _require_finite_float(
    container: Mapping[str, object],
    key: str,
    *,
    endpoint: str,
) -> float:
    value = container.get(key)
    if not isinstance(value, float) or not isfinite(value):
        raise LlamaCppProtocolError(f"{endpoint} field {key!r} must be a finite float")
    return value


def _raw_text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_default_settings_values(settings: Mapping[str, object]) -> None:
    missing = [key for key in _REQUIRED_EFFECTIVE_SETTINGS if key not in settings]
    if missing:
        raise LlamaCppProtocolError(
            "default generation settings are missing: " + ", ".join(missing)
        )
    _require_finite_float(settings, "temperature", endpoint="effective defaults")
    _require_finite_float(settings, "top_p", endpoint="effective defaults")
    _require_int(settings, "top_k", endpoint="effective defaults")
    _require_finite_float(settings, "presence_penalty", endpoint="effective defaults")
    _require_int(settings, "n_predict", endpoint="effective defaults")
    _require_int(settings, "seed", endpoint="effective defaults")


def llama_cpp_provider_identity(
    effective_configuration: LlamaCppEffectiveConfiguration,
) -> ProviderIdentity:
    """Derive the exact adapter identity from validated preflight evidence."""

    if not isinstance(effective_configuration, LlamaCppEffectiveConfiguration):
        raise TypeError("effective_configuration must be a LlamaCppEffectiveConfiguration")
    return ProviderIdentity(
        provider_type="llama_cpp",
        implementation_version=LLAMA_CPP_IMPLEMENTATION_VERSION,
        model_alias=effective_configuration.model_alias,
        build_id=effective_configuration.build_id,
        provider_config_hash=canonical_sha256(effective_configuration),
        model_path=effective_configuration.model_path,
        model_sha256=effective_configuration.model_sha256,
        chat_template_sha256=effective_configuration.chat_template_sha256,
    )


def require_llama_cpp_provider_binding(
    provider_identity: ProviderIdentity,
    effective_configuration_json: str,
) -> LlamaCppEffectiveConfiguration:
    """Parse and prove exact identity/configuration agreement for claim evidence."""

    if not isinstance(provider_identity, ProviderIdentity):
        raise TypeError("provider_identity must be a ProviderIdentity")
    if not isinstance(effective_configuration_json, str):
        raise TypeError("effective_configuration_json must be a string")
    if provider_identity.provider_type != "llama_cpp":
        raise ValueError("provider identity must be llama_cpp")
    effective = LlamaCppEffectiveConfiguration.model_validate_json(effective_configuration_json)
    if canonical_json(effective) != effective_configuration_json:
        raise ValueError("llama_cpp effective configuration must be canonical JSON")
    if llama_cpp_provider_identity(effective) != provider_identity:
        raise ValueError("llama_cpp provider identity disagrees with effective configuration")
    return effective


_ModelArtifactFingerprint = tuple[int, int, int, int, int]


def _artifact_fingerprint(stat_result: os.stat_result) -> _ModelArtifactFingerprint:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _measure_model_artifact(model_path: str) -> tuple[str, _ModelArtifactFingerprint]:
    path = Path(model_path)
    if not path.is_absolute():
        raise LlamaCppIdentityDriftError(
            "model_path must be an absolute client-local path for artifact verification"
        )
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise LlamaCppIdentityDriftError(
                "model artifact is not a readable client-local regular file"
            )
        with path.open("rb") as artifact:
            before_stat = os.fstat(artifact.fileno())
            if not stat.S_ISREG(before_stat.st_mode):
                raise LlamaCppIdentityDriftError(
                    "model artifact is not a readable client-local regular file"
                )
            before = _artifact_fingerprint(before_stat)
            digest = sha256()
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
            after = _artifact_fingerprint(os.fstat(artifact.fileno()))
        current = _artifact_fingerprint(path.stat())
    except (OSError, ValueError) as error:
        raise LlamaCppIdentityDriftError(
            "model artifact is not a readable client-local regular file"
        ) from error
    if before[:4] != after[:4] or after[:4] != current[:4]:
        raise LlamaCppIdentityDriftError("model artifact changed while its digest was measured")
    return digest.hexdigest(), current


class LlamaCppProvider:
    """Invoke one explicitly configured llama.cpp server without fallback.

    Construction performs one ``/health`` and one ``/props`` inspection. Each
    generation re-inspects those endpoints before exactly one ``/completion``
    dispatch. HTTPX's environment integration and redirects are disabled, and
    no retry transport is installed.
    """

    __slots__ = (
        "_client",
        "_config",
        "_effective_configuration",
        "_last_raw_response_sha256",
        "_model_artifact_fingerprint",
        "_provider_identity",
    )

    def __init__(
        self,
        config: LlamaCppProviderConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.Client(
            timeout=config.timeout,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )
        self._last_raw_response_sha256: str | None = None
        try:
            self._model_artifact_fingerprint = self._verify_model_artifact()
            effective_configuration = self._inspect_effective_configuration()
            provider_identity = llama_cpp_provider_identity(effective_configuration)
        except Exception:
            self._client.close()
            raise
        self._effective_configuration = effective_configuration
        self._provider_identity = provider_identity

    @property
    def provider_identity(self) -> ProviderIdentity:
        """Return the server identity bound during construction."""

        return self._provider_identity

    @property
    def effective_configuration(self) -> LlamaCppEffectiveConfiguration:
        """Return the exact static configuration bound from ``/props``."""

        return self._effective_configuration

    @property
    def effective_configuration_json(self) -> str:
        """Return exact canonical client and inspected server configuration evidence."""

        return canonical_json(self._effective_configuration)

    @property
    def last_raw_response_sha256(self) -> str | None:
        """Return the canonical hash of the last parsed raw completion."""

        return self._last_raw_response_sha256

    def verify_model_artifact(self) -> None:
        """Rehash the local model artifact and require the configured digest."""

        self._model_artifact_fingerprint = self._verify_model_artifact()

    def close(self) -> None:
        """Close the owned synchronous HTTP client."""

        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Inspect identity and dispatch one non-streaming completion request."""

        if request.provider_identity_id != self.provider_identity.identity_id:
            raise ProviderIdentityMismatchError(
                "generation request provider identity does not match LlamaCppProvider"
            )
        if not 0 <= request.decoding_parameters.seed < 0xFFFFFFFF:
            raise LlamaCppProtocolError(
                "llama.cpp requires a deterministic seed in range 0..4294967294"
            )

        self._require_stable_model_artifact()
        self._require_stable_identity()
        parameters = request.decoding_parameters
        payload: dict[str, object] = {
            "prompt": request.prompt,
            "model": self._config.model_alias,
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "top_k": parameters.top_k,
            "presence_penalty": parameters.presence_penalty,
            "n_predict": parameters.max_tokens,
            "seed": parameters.seed,
            "stream": False,
            "cache_prompt": False,
        }
        raw_response = self._request_json("POST", "/completion", json=payload)
        self._last_raw_response_sha256 = canonical_sha256(raw_response)
        effective_parameters = self._validate_completion(raw_response, parameters)
        return GenerationResponse(
            text=cast(str, raw_response["content"]),
            provider_identity=self.provider_identity,
            effective_parameters=effective_parameters,
            raw_metadata=GenerationMetadata(
                request_sha256=canonical_sha256(request),
                generation_method="llama_cpp_completion_http_v1",
                provider_request_json=canonical_json(payload),
                provider_request_sha256=canonical_sha256(payload),
                provider_response_json=canonical_json(raw_response),
                provider_response_sha256=canonical_sha256(raw_response),
            ),
        )

    def _url(self, endpoint: str) -> str:
        return f"{self._config.base_url}{endpoint}"

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        json: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        try:
            response = self._client.request(method, self._url(endpoint), json=json)
        except httpx.HTTPError as error:
            raise LlamaCppTransportError(
                f"single {method} {endpoint} operation failed; no retry was attempted"
            ) from error
        if response.status_code != 200:
            raise LlamaCppTransportError(
                f"single {method} {endpoint} operation returned HTTP {response.status_code}; "
                "no retry was attempted"
            )
        try:
            value: object = response.json()
        except ValueError as error:
            raise LlamaCppProtocolError(f"{endpoint} response is not valid JSON") from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise LlamaCppProtocolError(f"{endpoint} response must be a JSON object")
        result = cast(dict[str, object], value)
        try:
            canonical_json(result)
        except (TypeError, ValueError) as error:
            raise LlamaCppProtocolError(
                f"{endpoint} response is not canonical finite JSON"
            ) from error
        return result

    def _inspect_effective_configuration(self) -> LlamaCppEffectiveConfiguration:
        health = self._request_json("GET", "/health")
        if health != {"status": "ok"}:
            raise LlamaCppProtocolError("/health response must be exactly {'status': 'ok'}")

        props = self._request_json("GET", "/props")
        model_path = _require_non_blank_string(props, "model_path", endpoint="/props")
        build_id = _require_non_blank_string(props, "build_info", endpoint="/props")
        chat_template = _require_non_blank_string(props, "chat_template", endpoint="/props")
        total_slots = _require_int(props, "total_slots", endpoint="/props")
        if total_slots <= 0:
            raise LlamaCppProtocolError("/props total_slots must be positive")

        default_wrapper = _require_mapping(
            props,
            "default_generation_settings",
            endpoint="/props",
        )
        default_settings = _require_mapping(
            default_wrapper,
            "params",
            endpoint="/props.default_generation_settings",
        )
        self._validate_default_settings(default_settings)
        settings_json = canonical_json(default_settings)
        template_sha256 = _raw_text_sha256(chat_template)

        if model_path != self._config.model_path:
            raise LlamaCppIdentityDriftError(
                "/props model_path does not match explicit provider configuration"
            )
        if build_id != self._config.build_id:
            raise LlamaCppIdentityDriftError(
                "/props build_info does not match explicit provider configuration"
            )
        if template_sha256 != self._config.chat_template_sha256:
            raise LlamaCppIdentityDriftError(
                "/props chat_template does not match explicit provider configuration"
            )

        return LlamaCppEffectiveConfiguration(
            client_config=self._config,
            model_alias=self._config.model_alias,
            model_path=model_path,
            model_sha256=self._config.model_sha256,
            build_id=build_id,
            chat_template=chat_template,
            chat_template_sha256=template_sha256,
            default_generation_settings_json=settings_json,
            total_slots=total_slots,
        )

    @staticmethod
    def _validate_default_settings(settings: Mapping[str, object]) -> None:
        _validate_default_settings_values(settings)

    def _require_stable_identity(self) -> None:
        observed = self._inspect_effective_configuration()
        observed_identity = llama_cpp_provider_identity(observed)
        if observed != self._effective_configuration or observed_identity != self.provider_identity:
            raise LlamaCppIdentityDriftError(
                "llama.cpp identity or effective configuration drifted before dispatch"
            )

    def _require_stable_model_artifact(self) -> None:
        try:
            current = _artifact_fingerprint(Path(self._config.model_path).stat())
        except OSError as error:
            raise LlamaCppIdentityDriftError(
                "model artifact is unavailable before completion dispatch"
            ) from error
        if current != self._model_artifact_fingerprint:
            self._model_artifact_fingerprint = self._verify_model_artifact()

    def _verify_model_artifact(self) -> _ModelArtifactFingerprint:
        observed_sha256, fingerprint = _measure_model_artifact(self._config.model_path)
        if observed_sha256 != self._config.model_sha256:
            raise LlamaCppIdentityDriftError(
                "model artifact SHA-256 does not match explicit provider configuration"
            )
        return fingerprint

    def _validate_completion(
        self,
        response: Mapping[str, object],
        requested: DecodingParameters,
    ) -> DecodingParameters:
        content = response.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LlamaCppProtocolError("/completion content must be a non-blank string")
        if response.get("stop") is not True:
            raise LlamaCppProtocolError("/completion must report a completed non-streaming result")
        model_alias = _require_non_blank_string(response, "model", endpoint="/completion")
        if model_alias != self._config.model_alias:
            raise LlamaCppIdentityDriftError(
                "/completion model alias does not match the bound provider identity"
            )

        settings = _require_mapping(response, "generation_settings", endpoint="/completion")
        temperature = _require_finite_float(settings, "temperature", endpoint="/completion")
        top_p = _require_finite_float(settings, "top_p", endpoint="/completion")
        top_k = _require_int(settings, "top_k", endpoint="/completion")
        presence_penalty = _require_finite_float(
            settings,
            "presence_penalty",
            endpoint="/completion",
        )
        n_predict = _require_int(settings, "n_predict", endpoint="/completion")
        max_tokens = _require_int(settings, "max_tokens", endpoint="/completion")
        seed = _require_int(settings, "seed", endpoint="/completion")

        float_pairs = (
            ("temperature", temperature, requested.temperature),
            ("top_p", top_p, requested.top_p),
            ("presence_penalty", presence_penalty, requested.presence_penalty),
        )
        for name, observed, expected in float_pairs:
            if not isclose(
                observed,
                expected,
                rel_tol=DECODING_FLOAT_RELATIVE_TOLERANCE,
                abs_tol=DECODING_FLOAT_ABSOLUTE_TOLERANCE,
            ):
                raise LlamaCppProtocolError(
                    f"/completion effective {name} differs from the requested value"
                )
        if top_k != requested.top_k:
            raise LlamaCppProtocolError(
                "/completion effective top_k differs from the requested value"
            )
        if n_predict != requested.max_tokens or max_tokens != requested.max_tokens:
            raise LlamaCppProtocolError(
                "/completion effective generation budget differs from fixed max_tokens"
            )
        if seed != requested.seed:
            raise LlamaCppProtocolError("/completion effective seed differs from requested seed")

        try:
            return DecodingParameters(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                max_tokens=n_predict,
                seed=seed,
            )
        except ValidationError as error:
            raise LlamaCppProtocolError(
                "/completion effective settings violate the decoding-parameter contract"
            ) from error


__all__ = [
    "LLAMA_CPP_IMPLEMENTATION_VERSION",
    "LlamaCppEffectiveConfiguration",
    "LlamaCppIdentityDriftError",
    "LlamaCppProtocolError",
    "LlamaCppProvider",
    "LlamaCppProviderConfig",
    "LlamaCppProviderError",
    "LlamaCppTransportError",
    "llama_cpp_provider_identity",
    "require_llama_cpp_provider_binding",
]

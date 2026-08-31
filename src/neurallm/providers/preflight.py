"""Identity-only llama.cpp preflight with no generation dispatch."""

from __future__ import annotations

from typing import Literal, Self

import httpx
from pydantic import Field, model_validator

from neurallm.domain.models import ProviderIdentity, Sha256Hex, StrictFrozenModel
from neurallm.domain.serialization import canonical_json
from neurallm.providers.llama_cpp import (
    LlamaCppEffectiveConfiguration,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)


class LlamaCppPreflightResult(StrictFrozenModel):
    """Canonical identity evidence produced without requesting a completion."""

    schema_version: Literal[1] = 1
    provider_kind: Literal["llama_cpp"] = "llama_cpp"
    expected_identity: ProviderIdentity
    provider_identity_id: Sha256Hex
    expected_effective_configuration_json: str = Field(min_length=2)
    completion_requested: Literal[False] = False

    @model_validator(mode="after")
    def _validate_identity_evidence(self) -> Self:
        effective = LlamaCppEffectiveConfiguration.model_validate_json(
            self.expected_effective_configuration_json
        )
        if canonical_json(effective) != self.expected_effective_configuration_json:
            raise ValueError("preflight effective configuration must be canonical JSON")
        if self.expected_identity != llama_cpp_provider_identity(effective):
            raise ValueError("preflight identity disagrees with effective configuration")
        if self.provider_identity_id != self.expected_identity.identity_id:
            raise ValueError("preflight identity ID disagrees with expected_identity")
        return self


def preflight_llama_cpp(
    config: LlamaCppProviderConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> LlamaCppPreflightResult:
    """Inspect ``/health`` and ``/props`` exactly once, never ``/completion``."""

    if not isinstance(config, LlamaCppProviderConfig):
        raise TypeError("config must be a LlamaCppProviderConfig")
    with LlamaCppProvider(config, transport=transport) as provider:
        identity = provider.provider_identity
        return LlamaCppPreflightResult(
            expected_identity=identity,
            provider_identity_id=identity.identity_id,
            expected_effective_configuration_json=provider.effective_configuration_json,
        )


__all__ = ["LlamaCppPreflightResult", "preflight_llama_cpp"]

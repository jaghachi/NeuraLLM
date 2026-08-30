"""Deterministic identifiers derived from canonical scientific inputs."""

from __future__ import annotations

from typing import Any

from neurallm.domain.models import ExperimentCondition, ProviderIdentity
from neurallm.domain.serialization import canonical_sha256


def deterministic_identifier(namespace: str, payload: Any) -> str:
    """Hash a payload under a non-empty namespace."""

    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    if not namespace.strip():
        raise ValueError("namespace must not be blank")
    return canonical_sha256({"namespace": namespace, "payload": payload})


def condition_id(condition: ExperimentCondition) -> str:
    """Return the canonical identifier for an experiment condition."""

    if not isinstance(condition, ExperimentCondition):
        raise TypeError("condition must be an ExperimentCondition")
    return canonical_sha256(condition)


def condition_identifier(condition: ExperimentCondition) -> str:
    """Spelled-out alias for :func:`condition_id`."""

    return condition_id(condition)


def provider_identity_id(identity: ProviderIdentity) -> str:
    """Return the canonical identifier for a provider identity."""

    if not isinstance(identity, ProviderIdentity):
        raise TypeError("identity must be a ProviderIdentity")
    return canonical_sha256(identity)

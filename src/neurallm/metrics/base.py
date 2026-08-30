"""Typed interfaces for deterministic response metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from neurallm.domain.models import MetricValue
from neurallm.metrics.validators import ValidatorSpec


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MetricContext(_StrictFrozenModel):
    """Complete deterministic input for response-derived metrics."""

    prompt_case_id: str = Field(min_length=1)
    prompt_family: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response_text: str
    validator: ValidatorSpec


MetricOutput = MetricValue[int] | MetricValue[float]


@runtime_checkable
class MetricPlugin(Protocol):
    """One versioned deterministic metric implementation."""

    metric_name: str
    metric_version: str

    def compute(self, context: MetricContext) -> MetricOutput:
        """Compute a provenance-bearing metric from immutable inputs."""
        ...


class MetricRegistry:
    """Immutable registry that rejects ambiguous metric names."""

    __slots__ = ("_plugins",)

    def __init__(self, plugins: Iterable[MetricPlugin]) -> None:
        indexed: dict[str, MetricPlugin] = {}
        for plugin in plugins:
            name = plugin.metric_name
            if not name.strip():
                raise ValueError("metric_name must not be blank")
            if name in indexed:
                raise ValueError(f"duplicate metric plugin: {name}")
            indexed[name] = plugin
        if not indexed:
            raise ValueError("at least one metric plugin is required")
        self._plugins = MappingProxyType(dict(sorted(indexed.items())))

    @property
    def plugins(self) -> Mapping[str, MetricPlugin]:
        return self._plugins

    def compute(self, context: MetricContext) -> Mapping[str, MetricOutput]:
        return MappingProxyType(
            {name: plugin.compute(context) for name, plugin in self._plugins.items()}
        )


__all__ = [
    "MetricContext",
    "MetricOutput",
    "MetricPlugin",
    "MetricRegistry",
]

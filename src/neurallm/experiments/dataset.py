"""Strict deterministic prompt-dataset loading and identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import PromptFeatures
from neurallm.domain.serialization import canonical_sha256
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.metrics.validators import ValidatorSpec


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PromptCase(_StrictFrozenModel):
    """One ordered prompt turn and its deterministic objective."""

    case_id: str = Field(min_length=1)
    prompt_family: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    prompt_features: PromptFeatures = Field(default_factory=lambda: PromptFeatures({}))
    validator: ValidatorSpec


class PromptSequence(_StrictFrozenModel):
    """One correlated ordered prompt sequence."""

    sequence_id: str = Field(min_length=1)
    cases: tuple[PromptCase, ...]

    @field_validator("cases", mode="before")
    @classmethod
    def _accept_yaml_cases(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_cases(self) -> Self:
        if not self.cases:
            raise ValueError("prompt sequence must contain at least one case")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case ids must be unique within a prompt sequence")
        return self


class PromptDataset(_StrictFrozenModel):
    """Canonical versioned input dataset."""

    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sequences: tuple[PromptSequence, ...]

    @field_validator("sequences", mode="before")
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_sequences(self) -> Self:
        if not self.sequences:
            raise ValueError("dataset must contain at least one sequence")
        sequence_ids = [sequence.sequence_id for sequence in self.sequences]
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("sequence ids must be unique")
        return self

    @property
    def dataset_hash(self) -> str:
        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "dataset_id": self.dataset_id,
                "version": self.version,
                "sequences": tuple(
                    sorted(self.sequences, key=lambda sequence: sequence.sequence_id)
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    dataset: PromptDataset
    source_path: Path


def load_dataset(path: Path, *, expected_version: str | None = None) -> LoadedDataset:
    """Load a dataset from one explicit path and enforce its expected version."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    source_path = path.expanduser().resolve(strict=True)
    dataset = PromptDataset.model_validate(load_yaml_mapping(source_path))
    if expected_version is not None and dataset.version != expected_version:
        raise ValueError(
            f"dataset version mismatch: expected {expected_version!r}, got {dataset.version!r}"
        )
    return LoadedDataset(dataset=dataset, source_path=source_path)


__all__ = [
    "LoadedDataset",
    "PromptCase",
    "PromptDataset",
    "PromptSequence",
    "load_dataset",
]

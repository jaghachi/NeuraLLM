"""Strict deterministic prompt-dataset loading and identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import PromptFeatures, Sha256Hex
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.metrics.validators import ValidatorSpec


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class DatasetSeal(_StrictFrozenModel):
    """Canonical external evidence freezing one evaluation dataset identity."""

    schema_version: Literal[1] = 1
    seal_protocol_version: Literal["dataset-seal-v1"] = "dataset-seal-v1"
    purpose: Literal[DatasetPurpose.EVALUATION] = DatasetPurpose.EVALUATION
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_sha256: Sha256Hex

    @field_validator("dataset_id", "dataset_version")
    @classmethod
    def _reject_blank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset seal identity must not be blank")
        return value

    @property
    def seal_sha256(self) -> str:
        """Return the canonical identity of this complete seal record."""

        return canonical_sha256(self)


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
    purpose: DatasetPurpose | None = None
    sequences: tuple[PromptSequence, ...]

    @field_validator("purpose", mode="before")
    @classmethod
    def _accept_yaml_purpose(cls, value: object) -> object:
        if isinstance(value, str):
            return DatasetPurpose(value)
        return value

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
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "sequences": tuple(sorted(self.sequences, key=lambda sequence: sequence.sequence_id)),
        }
        if self.purpose is not None:
            payload["purpose"] = self.purpose
        return canonical_sha256(payload)


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


def validate_dataset_identity(
    dataset: PromptDataset,
    *,
    expected_version: str,
    expected_purpose: DatasetPurpose | None,
    expected_sha256: str | None,
    seal: DatasetSeal | None,
) -> None:
    """Validate declared dataset purpose, content identity, and evaluation seal."""

    if dataset.version != expected_version:
        raise ValueError("dataset version does not match experiment configuration")
    if expected_purpose is None:
        if expected_sha256 is not None or seal is not None:
            raise ValueError("legacy dataset references cannot declare partial Phase 3 identity")
        return
    if dataset.purpose != expected_purpose:
        raise ValueError("dataset purpose does not match experiment configuration")
    if expected_sha256 is None or dataset.dataset_hash != expected_sha256:
        raise ValueError("dataset SHA-256 does not match experiment configuration")
    if expected_purpose is DatasetPurpose.EVALUATION:
        if seal is None:
            raise ValueError("evaluation dataset requires a canonical seal")
        if (
            seal.dataset_id != dataset.dataset_id
            or seal.dataset_version != dataset.version
            or seal.dataset_sha256 != dataset.dataset_hash
        ):
            raise ValueError("evaluation dataset does not match its canonical seal")
    elif seal is not None:
        raise ValueError("only evaluation datasets may carry a seal")


def require_development_selection_input(dataset: PromptDataset) -> PromptDataset:
    """Fail closed unless static selection receives a declared development dataset."""

    if dataset.purpose is not DatasetPurpose.DEVELOPMENT:
        raise ValueError("static selection requires a development-purpose dataset")
    return dataset


__all__ = [
    "DatasetPurpose",
    "DatasetSeal",
    "LoadedDataset",
    "PromptCase",
    "PromptDataset",
    "PromptSequence",
    "load_dataset",
    "require_development_selection_input",
    "validate_dataset_identity",
]

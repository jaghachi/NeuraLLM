"""Provider-free publication of frozen confirmatory scientific identities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from neurallm.domain.serialization import canonical_json_bytes
from neurallm.experiments.config import LoadedExperimentConfig, load_experiment_config
from neurallm.experiments.dataset import (
    load_dataset,
    validate_dataset_identity,
)
from neurallm.experiments.plan import build_plan
from neurallm.experiments.protocol import PreregistrationSeal, RunTier
from neurallm.experiments.static_selection import (
    validate_static_selection_evidence_against_dataset,
)
from neurallm.experiments.yaml_loader import load_yaml_mapping


@dataclass(frozen=True, slots=True)
class PreregistrationPublication:
    """One validated, provider-free preregistration publication result."""

    config_path: Path
    dataset_path: Path
    output_path: Path
    seal: PreregistrationSeal
    created: bool
    sealed_config_path: Path | None = None
    sealed_config_created: bool | None = None
    development_selection_dataset_path: Path | None = None

    @property
    def scientific_identity_sha256(self) -> str:
        """Return the frozen confirmatory scientific identity."""

        return self.seal.scientific_identity_sha256

    @property
    def preregistration_sha256(self) -> str:
        """Return the canonical identity of the complete seal artifact."""

        return self.seal.seal_sha256


def load_preregistration_seal(path: Path) -> PreregistrationSeal:
    """Load one explicit JSON-or-YAML preregistration seal path."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    source_path = path.expanduser().resolve(strict=True)
    return PreregistrationSeal.model_validate(load_yaml_mapping(source_path))


def _existing_bytes_match(path: Path, expected: bytes) -> bool:
    try:
        return path.read_bytes() == expected
    except OSError:
        return False


def _conflicting_artifact(path: Path, artifact_name: str) -> FileExistsError:
    return FileExistsError(f"refusing to overwrite differing {artifact_name}: {path}")


def _ensure_publishable(path: Path, payload: bytes, artifact_name: str) -> None:
    if os.path.lexists(path) and not _existing_bytes_match(path, payload):
        raise _conflicting_artifact(path, artifact_name)


def _publish_exact_bytes(path: Path, payload: bytes, artifact_name: str) -> bool:
    """Atomically create one exact artifact without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if _existing_bytes_match(path, payload):
            return False
        raise _conflicting_artifact(path, artifact_name)

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if _existing_bytes_match(path, payload):
                return False
            raise _conflicting_artifact(path, artifact_name) from None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_materialized_config(directory: Path, payload: bytes) -> LoadedExperimentConfig:
    """Resolve one prospective config through the same reference-aware loader."""

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".preregister-materialized.",
            suffix=".yaml",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return load_experiment_config(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_preregistration(
    config_path: Path,
    output_path: Path,
    sealed_config_output_path: Path | None = None,
) -> PreregistrationPublication:
    """Validate, seal, optionally materialize, and revalidate one confirmatory plan.

    This path consumes only declared files and never constructs or contacts a
    generation provider. The output path is incidental and is never included
    in the candidate plan or its scientific identity. When requested, the
    sealed configuration is written beside the source configuration so all
    explicit relative references retain their meaning.
    """

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    if sealed_config_output_path is not None and not isinstance(sealed_config_output_path, Path):
        raise TypeError("sealed_config_output_path must be a pathlib.Path or None")
    loaded_config = load_experiment_config(config_path)
    config = loaded_config.config
    if config.protocol is None or config.protocol.run_tier is not RunTier.CONFIRMATORY:
        raise ValueError("preregistration publishing requires a confirmatory protocol")
    if config.preregistration is not None:
        raise ValueError("preregistration publishing requires an unsealed configuration")

    loaded_dataset = load_dataset(
        loaded_config.dataset_path,
        expected_version=config.dataset.version,
    )
    validate_dataset_identity(
        loaded_dataset.dataset,
        expected_version=config.dataset.version,
        expected_purpose=config.dataset.purpose,
        expected_sha256=config.dataset.expected_dataset_sha256,
        seal=config.dataset.seal,
    )
    development_input = config.development_selection_input
    development_path = loaded_config.development_selection_dataset_path
    if development_input is None or development_path is None:
        raise RuntimeError("confirmatory preregistration lacks its development dataset reference")
    loaded_development_dataset = load_dataset(
        development_path,
        expected_version=development_input.dataset.version,
    )
    validate_dataset_identity(
        loaded_development_dataset.dataset,
        expected_version=development_input.dataset.version,
        expected_purpose=development_input.dataset.purpose,
        expected_sha256=development_input.dataset.expected_dataset_sha256,
        seal=development_input.dataset.seal,
    )
    selection_evidence = config.static_selection_evidence
    if selection_evidence is None:
        raise RuntimeError("confirmatory preregistration lacks static-selection evidence")
    validate_static_selection_evidence_against_dataset(
        selection_evidence,
        loaded_development_dataset.dataset,
    )
    candidate = build_plan(
        loaded_config,
        loaded_dataset,
        require_frozen_preregistration=False,
    )
    seal = PreregistrationSeal(
        experiment_id=config.experiment_id,
        scientific_identity_sha256=candidate.scientific_identity_sha256,
    )

    resolved_output = output_path.expanduser().resolve()
    seal_bytes = canonical_json_bytes(seal)
    resolved_sealed_config: Path | None = None
    source_payload = load_yaml_mapping(loaded_config.source_path)
    source_payload["preregistration"] = seal.model_dump(mode="json")
    sealed_config_bytes = yaml.safe_dump(
        source_payload,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")
    materialized_config = _validate_materialized_config(
        loaded_config.source_path.parent,
        sealed_config_bytes,
    )
    sealed_plan = build_plan(materialized_config, loaded_dataset)
    if sealed_plan.scientific_identity_sha256 != candidate.scientific_identity_sha256:
        raise RuntimeError("sealed plan identity differs from its published preregistration")
    if (
        materialized_config.dataset_path != loaded_config.dataset_path
        or materialized_config.provider_config_path != loaded_config.provider_config_path
        or materialized_config.artifact_root != loaded_config.artifact_root
        or materialized_config.development_selection_dataset_path
        != loaded_config.development_selection_dataset_path
        or materialized_config.static_selection_evidence_path
        != loaded_config.static_selection_evidence_path
    ):
        raise RuntimeError("materialized sealed configuration changed a resolved reference")

    if sealed_config_output_path is not None:
        resolved_sealed_config = sealed_config_output_path.expanduser().resolve()
        if resolved_sealed_config == resolved_output:
            raise ValueError("seal and sealed configuration output paths must be distinct")
        if resolved_sealed_config.parent != loaded_config.source_path.parent:
            raise ValueError(
                "sealed configuration output must be beside the source configuration "
                "so relative references remain unchanged"
            )
        _ensure_publishable(
            resolved_sealed_config,
            sealed_config_bytes,
            "sealed experiment configuration",
        )

    _ensure_publishable(resolved_output, seal_bytes, "preregistration artifact")
    created = _publish_exact_bytes(
        resolved_output,
        seal_bytes,
        "preregistration artifact",
    )

    reloaded_seal = load_preregistration_seal(resolved_output)
    if reloaded_seal != seal or reloaded_seal.seal_sha256 != seal.seal_sha256:
        raise RuntimeError("published preregistration seal identity changed during reload")

    sealed_config_created: bool | None = None
    if resolved_sealed_config is not None:
        sealed_config_created = _publish_exact_bytes(
            resolved_sealed_config,
            sealed_config_bytes,
            "sealed experiment configuration",
        )
        published_config = load_experiment_config(resolved_sealed_config)
        if published_config.config != materialized_config.config:
            raise RuntimeError("published sealed configuration changed during reload")
        published_plan = build_plan(published_config, loaded_dataset)
        if published_plan.scientific_identity_sha256 != candidate.scientific_identity_sha256:
            raise RuntimeError("published sealed configuration changed scientific identity")

    return PreregistrationPublication(
        config_path=loaded_config.source_path,
        dataset_path=loaded_dataset.source_path,
        output_path=resolved_output,
        seal=reloaded_seal,
        created=created,
        sealed_config_path=resolved_sealed_config,
        sealed_config_created=sealed_config_created,
        development_selection_dataset_path=loaded_development_dataset.source_path,
    )


__all__ = [
    "PreregistrationPublication",
    "load_preregistration_seal",
    "publish_preregistration",
]

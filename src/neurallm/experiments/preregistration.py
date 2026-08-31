"""Provider-free publication of frozen confirmatory scientific identities."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from neurallm.domain.serialization import canonical_json_bytes
from neurallm.experiments.config import ExperimentConfig, load_experiment_config
from neurallm.experiments.dataset import (
    load_dataset,
    validate_dataset_identity,
)
from neurallm.experiments.plan import build_plan
from neurallm.experiments.protocol import PreregistrationSeal, RunTier
from neurallm.experiments.yaml_loader import load_yaml_mapping


@dataclass(frozen=True, slots=True)
class PreregistrationPublication:
    """One validated, provider-free preregistration publication result."""

    config_path: Path
    dataset_path: Path
    output_path: Path
    seal: PreregistrationSeal
    created: bool

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


def _conflicting_artifact(path: Path) -> FileExistsError:
    return FileExistsError(f"refusing to overwrite differing preregistration artifact: {path}")


def _publish_canonical_json(path: Path, payload: bytes) -> bool:
    """Atomically create one exact artifact without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if _existing_bytes_match(path, payload):
            return False
        raise _conflicting_artifact(path)

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
            raise _conflicting_artifact(path) from None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_preregistration(
    config_path: Path,
    output_path: Path,
) -> PreregistrationPublication:
    """Validate, seal, publish, reload, and revalidate one confirmatory plan.

    This path consumes only declared files and never constructs or contacts a
    generation provider. The output path is incidental and is never included
    in the candidate plan or its scientific identity.
    """

    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
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
    created = _publish_canonical_json(resolved_output, canonical_json_bytes(seal))

    reloaded_seal = load_preregistration_seal(resolved_output)
    if reloaded_seal != seal or reloaded_seal.seal_sha256 != seal.seal_sha256:
        raise RuntimeError("published preregistration seal identity changed during reload")

    sealed_config = ExperimentConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "preregistration": reloaded_seal,
        }
    )
    sealed_plan = build_plan(
        replace(loaded_config, config=sealed_config),
        loaded_dataset,
    )
    if sealed_plan.scientific_identity_sha256 != candidate.scientific_identity_sha256:
        raise RuntimeError("sealed plan identity differs from its published preregistration")

    return PreregistrationPublication(
        config_path=loaded_config.source_path,
        dataset_path=loaded_dataset.source_path,
        output_path=resolved_output,
        seal=reloaded_seal,
        created=created,
    )


__all__ = [
    "PreregistrationPublication",
    "load_preregistration_seal",
    "publish_preregistration",
]

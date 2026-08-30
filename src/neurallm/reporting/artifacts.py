"""Export the compact, deterministic Phase 2 view of a canonical run store."""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from neurallm.domain.models import MetricValue, ResponseMetrics, RunManifest
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.storage import SQLiteRunStore, StoredTurn, TurnState

CLOSED_RUN_ARTIFACTS = frozenset(
    {
        "run.sqlite3",
        "manifest.json",
        "results.csv",
        "comparisons.csv",
        "decision.json",
        "report.md",
    }
)
SQLITE_RECOVERY_SIDECARS = frozenset({"run.sqlite3-journal", "run.sqlite3-shm", "run.sqlite3-wal"})

_RESULT_FIELDS = (
    "condition_id",
    "request_sha256",
    "history_commitment_sha256",
    "experiment_id",
    "dataset_version",
    "prompt_sequence_id",
    "turn_index",
    "policy_id",
    "model_seed",
    "controller_seed",
    "provider_identity_id",
    "base_decoding_profile_id",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "max_tokens",
    "response_text",
    "task_score",
    "instruction_adherence",
    "response_length_tokens",
    "repetition_ratio",
    "repeated_3_gram_ratio",
    "repeated_4_gram_ratio",
    "distinct_2",
    "distinct_3",
    "late_window_repetition_ratio",
    "format_validity",
    "semantic_similarity",
    "semantic_similarity_available",
)

_COMPARISON_FIELDS = (
    "comparison_id",
    "focal_policy_id",
    "comparator_policy_id",
    "estimate",
    "status",
)


@dataclass(frozen=True, slots=True)
class ArtifactExportSummary:
    """Stable identities and counts for one completed export."""

    output_directory: Path
    manifest_sha256: str
    scientific_result_sha256: str
    committed_turns: int
    artifact_names: tuple[str, ...]


def _metric_value(metric: MetricValue[int] | MetricValue[float]) -> int | float | str:
    return "" if metric.value is None else metric.value


def _result_row(turn: StoredTurn) -> dict[str, object]:
    if turn.response is None or turn.metrics is None or turn.history_commitment_sha256 is None:
        raise ValueError("committed turn is missing response, metric, or history evidence")
    condition = turn.condition
    parameters = turn.request.decoding_parameters
    metrics: ResponseMetrics = turn.metrics
    return {
        "condition_id": turn.condition_id,
        "request_sha256": turn.request_sha256,
        "history_commitment_sha256": turn.history_commitment_sha256,
        "experiment_id": condition.experiment_id,
        "dataset_version": condition.dataset_version,
        "prompt_sequence_id": condition.prompt_sequence_id,
        "turn_index": condition.turn_index,
        "policy_id": condition.policy_id,
        "model_seed": condition.model_seed,
        "controller_seed": condition.controller_seed,
        "provider_identity_id": condition.provider_identity_id,
        "base_decoding_profile_id": condition.base_decoding_profile_id,
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
        "top_k": parameters.top_k,
        "presence_penalty": parameters.presence_penalty,
        "max_tokens": parameters.max_tokens,
        "response_text": turn.response.text,
        "task_score": _metric_value(metrics.task_score),
        "instruction_adherence": _metric_value(metrics.instruction_adherence),
        "response_length_tokens": _metric_value(metrics.response_length_tokens),
        "repetition_ratio": _metric_value(metrics.repetition_ratio),
        "repeated_3_gram_ratio": _metric_value(metrics.repeated_3_gram_ratio),
        "repeated_4_gram_ratio": _metric_value(metrics.repeated_4_gram_ratio),
        "distinct_2": _metric_value(metrics.distinct_2),
        "distinct_3": _metric_value(metrics.distinct_3),
        "late_window_repetition_ratio": _metric_value(metrics.late_window_repetition_ratio),
        "format_validity": _metric_value(metrics.format_validity),
        "semantic_similarity": _metric_value(metrics.semantic_similarity),
        "semantic_similarity_available": metrics.semantic_similarity.availability,
    }


def _csv_text(fieldnames: tuple[str, ...], rows: tuple[dict[str, object], ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_atomic(path: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _reject_unexpected_files(
    output_directory: Path,
    *,
    allow_sqlite_recovery_sidecars: bool = False,
) -> None:
    allowed = set(CLOSED_RUN_ARTIFACTS)
    if allow_sqlite_recovery_sidecars:
        allowed.update(SQLITE_RECOVERY_SIDECARS)
    unexpected = sorted(
        item.name for item in output_directory.iterdir() if item.name not in allowed
    )
    if unexpected:
        raise ValueError(f"run directory contains unexpected artifacts: {unexpected!r}")


def _decision_payload(manifest: RunManifest, turns: tuple[StoredTurn, ...]) -> dict[str, object]:
    result_sha256 = scientific_result_sha256(turns)
    return {
        "schema_version": 1,
        "implementation_phase": 2,
        "claim_scope": "engineering_validation_only",
        "scientific_decision": None,
        "comparison_status": "not_available_until_phase_3",
        "decision_rule_version": manifest.decision_rule_version,
        "manifest_sha256": canonical_sha256(manifest),
        "scientific_result_sha256": result_sha256,
        "provider_type": manifest.provider_identity.provider_type,
        "committed_turns": len(turns),
        "database_integrity_verified": True,
        "rationale": (
            "Phase 2 validates the provider-to-artifact engineering path only; "
            "it does not estimate policy efficacy or select a scientific outcome."
        ),
    }


def _report_text(manifest: RunManifest, turns: tuple[StoredTurn, ...]) -> str:
    return (
        "# NeuraLLM Phase 2 Engineering Report\n\n"
        "This closed run validates the deterministic provider-to-artifact execution path. "
        "It does not establish policy efficacy, comparator advantage, neural activity, or a "
        "scientific decision.\n\n"
        f"- Manifest SHA-256: `{canonical_sha256(manifest)}`\n"
        f"- Scientific result SHA-256: `{scientific_result_sha256(turns)}`\n"
        f"- Provider type: `{manifest.provider_identity.provider_type}`\n"
        f"- Provider identity: `{manifest.provider_identity.identity_id}`\n"
        f"- Committed turns: `{len(turns)}`\n"
        f"- Database schema version: `{manifest.database_schema_version}`\n"
        f"- Decision rule: `{manifest.decision_rule_version}`\n\n"
        "`comparisons.csv` is intentionally empty because serious comparators and statistical "
        "evaluation begin in Phase 3. The canonical response and metric evidence remains in "
        "`run.sqlite3`; the other files are deterministic derived views.\n"
    )


def scientific_result_sha256(turns: tuple[StoredTurn, ...]) -> str:
    """Hash only canonical committed scientific results, excluding run location/source state."""

    if not turns:
        raise ValueError("scientific result requires at least one committed turn")
    evidence: list[dict[str, object]] = []
    for turn in turns:
        if (
            turn.state is not TurnState.COMMITTED
            or turn.response is None
            or turn.metrics is None
            or turn.policy_state_json is None
            or turn.policy_trace_json is None
            or turn.history_commitment_sha256 is None
        ):
            raise ValueError("scientific result contains incomplete turn evidence")
        evidence.append(
            {
                "condition_id": turn.condition_id,
                "request": turn.request,
                "history_binding": turn.history,
                "response": turn.response,
                "metrics": turn.metrics,
                "policy_state": json.loads(turn.policy_state_json),
                "policy_trace": json.loads(turn.policy_trace_json),
                "history_commitment_sha256": turn.history_commitment_sha256,
            }
        )
    return canonical_sha256({"schema_version": 1, "turns": evidence})


def export_closed_run(output_directory: Path) -> ArtifactExportSummary:
    """Verify and export exactly the compact Phase 2 artifact set."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be a pathlib.Path")
    output_directory = output_directory.expanduser().resolve(strict=True)
    if not output_directory.is_dir():
        raise ValueError("output_directory must be a directory")
    _reject_unexpected_files(output_directory, allow_sqlite_recovery_sidecars=True)
    database_path = output_directory / "run.sqlite3"
    if not database_path.is_file():
        raise FileNotFoundError("run directory does not contain run.sqlite3")

    with SQLiteRunStore(database_path) as store:
        store.verify_integrity()
        manifest = store.get_manifest()
        if manifest is None:
            raise ValueError("run store does not contain a manifest")
        finalization = store.get_finalization()
        if finalization is None:
            raise ValueError("run store is not finalized")
        turns = store.list_turns()
        if not turns:
            raise ValueError("closed run must contain at least one turn")
        incomplete = tuple(
            turn.condition_id for turn in turns if turn.state is not TurnState.COMMITTED
        )
        if incomplete:
            raise ValueError(f"run contains non-committed turns: {incomplete!r}")
        result_sha256 = scientific_result_sha256(turns)
        if result_sha256 != finalization.scientific_result_sha256:
            raise ValueError(
                "finalized scientific result hash does not match the recomputed output"
            )
        store.compact()

    result_rows = tuple(_result_row(turn) for turn in turns)
    decision = _decision_payload(manifest, turns)
    _write_atomic(output_directory / "manifest.json", canonical_json(manifest) + "\n")
    _write_atomic(output_directory / "results.csv", _csv_text(_RESULT_FIELDS, result_rows))
    _write_atomic(output_directory / "comparisons.csv", _csv_text(_COMPARISON_FIELDS, ()))
    _write_atomic(output_directory / "decision.json", canonical_json(decision) + "\n")
    _write_atomic(output_directory / "report.md", _report_text(manifest, turns))
    _reject_unexpected_files(output_directory)
    return ArtifactExportSummary(
        output_directory=output_directory,
        manifest_sha256=canonical_sha256(manifest),
        scientific_result_sha256=result_sha256,
        committed_turns=len(turns),
        artifact_names=tuple(sorted(CLOSED_RUN_ARTIFACTS)),
    )


__all__ = [
    "CLOSED_RUN_ARTIFACTS",
    "SQLITE_RECOVERY_SIDECARS",
    "ArtifactExportSummary",
    "export_closed_run",
    "scientific_result_sha256",
]

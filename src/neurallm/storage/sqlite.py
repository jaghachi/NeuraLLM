"""Transactional, hash-verified SQLite storage for experiment turns."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from neurallm.control.policy import PolicyState, PolicyTrace
from neurallm.domain.models import ActionBounds, ExperimentCondition, ResponseMetrics, RunManifest
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.aggregation import aggregate_matched_units, validate_exact_coverage
from neurallm.evaluation.attribution import PersistentStateAttributionResult
from neurallm.evaluation.confirmatory import ConfirmatoryEvaluationResult
from neurallm.evaluation.contract import phase3_analysis_contract_sha256
from neurallm.evaluation.models import (
    CoverageResult,
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    GuardrailResult,
    PairwiseComparisonResult,
    Phase3EvaluationResult,
    SequenceExpectation,
    TurnEvaluationRecord,
)
from neurallm.evaluation.scientific import (
    EfficacyComparisonResult,
    ScientificGuardrailResult,
)
from neurallm.metrics import MetricContext, compute_response_metrics
from neurallm.providers.base import (
    GenerationRequest,
    GenerationResponse,
    effective_parameters_match_request,
)
from neurallm.providers.llama_cpp import require_llama_cpp_provider_binding
from neurallm.providers.llama_cpp_evidence import require_llama_cpp_generation_binding
from neurallm.storage.errors import (
    DuplicateLogicalRequestError,
    HistoryMismatchError,
    ManifestMismatchError,
    SchemaVersionError,
    StateTransitionError,
    StoreCorruptionError,
    StoreInvariantError,
    UncertainDispatchError,
)
from neurallm.storage.migrations import (
    APPLICATION_ID,
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
)
from neurallm.storage.models import (
    AnalysisFinalization,
    AnalysisManifest,
    CommittedHistory,
    DurableExecutionAccounting,
    HistoryBinding,
    ResumeAction,
    RunFinalization,
    ScientificAnalysisFinalization,
    ScientificAnalysisManifest,
    StoredAnalysis,
    StoredScientificAnalysis,
    StoredTurn,
    TurnInputEvidence,
    TurnState,
)
from neurallm.storage.provenance import scientific_result_sha256

ModelT = TypeVar("ModelT", bound=BaseModel)
PolicyStateT = TypeVar("PolicyStateT", bound=PolicyState)

_PHASE3_ANALYSIS_IMPLEMENTATION_VERSION = "phase3-analysis-storage-v1"
_LEGACY_SCIENTIFIC_ANALYSIS_IMPLEMENTATION_VERSION = "confirmatory-scientific-analysis-storage-v1"
_SCIENTIFIC_ANALYSIS_IMPLEMENTATION_VERSION = "confirmatory-scientific-analysis-storage-v2"
_SUPPORTED_ANALYSIS_IMPLEMENTATION_VERSIONS = {
    _PHASE3_ANALYSIS_IMPLEMENTATION_VERSION,
    _SCIENTIFIC_ANALYSIS_IMPLEMENTATION_VERSION,
}
_SCIENTIFIC_UNIT_POLICY_IDS = frozenset(
    {"best_static", "heuristic_adaptive", "neural_persistent", "random_matched"}
)
_SCIENTIFIC_MODEL_BACKED_POLICY_IDS = _SCIENTIFIC_UNIT_POLICY_IDS | {
    "neural_matched_history_state_reset"
}


def _stored_scientific_records(
    turns: tuple[StoredTurn, ...],
    *,
    dataset_sha256: str,
    action_bounds: ActionBounds,
) -> tuple[TurnEvaluationRecord, ...]:
    """Reconstruct raw model-backed evaluator records from committed turn metrics."""

    records: list[TurnEvaluationRecord] = []
    for turn in turns:
        condition = turn.condition
        if condition.policy_id not in _SCIENTIFIC_MODEL_BACKED_POLICY_IDS:
            continue
        metrics = turn.metrics
        if turn.state is not TurnState.COMMITTED or metrics is None:
            raise ValueError("scientific unit evidence requires committed turn metrics")
        task_score = metrics.task_score.value
        instruction_adherence = metrics.instruction_adherence.value
        response_length_tokens = metrics.response_length_tokens.value
        repetition_ratio = metrics.repetition_ratio.value
        if (
            task_score is None
            or instruction_adherence is None
            or response_length_tokens is None
            or repetition_ratio is None
        ):
            raise ValueError("scientific unit evidence requires every primary metric")
        from neurallm.experiments.analysis import _trace_evidence

        try:
            (
                action_magnitude,
                action_within_bounds,
                action_saturated,
                observation_has_previous_response,
            ) = _trace_evidence(turn, action_bounds)
        except StoreInvariantError as exc:
            raise ValueError(f"scientific action evidence is invalid: {exc}") from exc
        if (turn.history is not None) != (condition.turn_index > 0):
            raise ValueError("scientific history evidence disagrees with the logical turn")
        previous_commitment = (
            None
            if not observation_has_previous_response or turn.history is None
            else turn.history.previous_history_commitment_sha256
        )
        records.append(
            TurnEvaluationRecord(
                dataset_sha256=dataset_sha256,
                prompt_sequence_id=condition.prompt_sequence_id,
                turn_index=condition.turn_index,
                policy_id=condition.policy_id,
                model_seed=condition.model_seed,
                controller_seed=condition.controller_seed,
                provider_identity_id=condition.provider_identity_id,
                has_previous_response=observation_has_previous_response,
                previous_history_commitment_sha256=previous_commitment,
                task_score=float(task_score),
                instruction_adherence=float(instruction_adherence),
                response_length_tokens=int(response_length_tokens),
                repetition_ratio=float(repetition_ratio),
                action_magnitude=action_magnitude,
                action_within_bounds=action_within_bounds,
                action_saturated=action_saturated,
            )
        )
    return tuple(records)


def _stored_scientific_unit_metrics(
    records: tuple[TurnEvaluationRecord, ...],
) -> dict[tuple[str, int, str], tuple[float, float, float, float]]:
    """Aggregate the four efficacy-arm raw metrics at the frozen matched unit."""

    outcomes = aggregate_matched_units(
        tuple(record for record in records if record.policy_id in _SCIENTIFIC_UNIT_POLICY_IDS)
    )
    return {
        (
            outcome.unit_key.prompt_sequence_id,
            outcome.unit_key.model_seed,
            outcome.policy_id,
        ): (
            outcome.task_score,
            outcome.instruction_adherence,
            outcome.repetition_ratio,
            outcome.response_length_tokens,
        )
        for outcome in outcomes
    }


def _source_scientific_guardrails(
    records: tuple[TurnEvaluationRecord, ...],
    *,
    run_manifest: RunManifest,
    dataset_seal_sha256: str,
    evaluation_spec: EvaluationSpec,
) -> tuple[CoverageResult, tuple[ScientificGuardrailResult, ...]]:
    """Recompute claim-bearing guardrails from committed source evidence."""

    if (
        evaluation_spec.focal_policy_id != "neural_persistent"
        or evaluation_spec.required_serious_comparator_ids != ("best_static", "heuristic_adaptive")
        or evaluation_spec.negative_control_policy_ids != ("random_matched",)
    ):
        raise ValueError("confirmatory evaluation spec has the wrong frozen policy roles")
    sequence_indexes: dict[str, set[int]] = {}
    for record in records:
        sequence_indexes.setdefault(record.prompt_sequence_id, set()).add(record.turn_index)
    if not sequence_indexes or any(
        indexes != set(range(len(indexes))) for indexes in sequence_indexes.values()
    ):
        raise ValueError("committed sequence turn indexes are not contiguous from zero")
    design = ExpectedEvaluationDesign(
        dataset_purpose=DatasetPurpose.EVALUATION,
        dataset_sha256=run_manifest.dataset_hash,
        dataset_seal_sha256=dataset_seal_sha256,
        provider_identity_id=run_manifest.provider_identity.identity_id,
        sequences=tuple(
            SequenceExpectation(prompt_sequence_id=sequence_id, turn_count=len(indexes))
            for sequence_id, indexes in sorted(sequence_indexes.items())
        ),
        model_seeds=run_manifest.seed_schedule.model_seeds,
        controller_seeds=run_manifest.seed_schedule.controller_seeds,
        policy_ids=tuple(run_manifest.policy_config_hashes),
    )
    coverage = validate_exact_coverage(records, design)
    if not coverage.exact:
        raise ValueError("committed turns do not cover the exact confirmatory design")

    from neurallm.experiments.scientific_analysis import _scientific_guardrails_from_records

    return coverage, _scientific_guardrails_from_records(records, design, evaluation_spec)


_EXPECTED_SCHEMA_OBJECTS = {
    ("table", "analysis_decision"),
    ("table", "analysis_finalization"),
    ("table", "analysis_manifest"),
    ("table", "comparison_results"),
    ("table", "guardrail_results"),
    ("index", "turns_schedule_index"),
    ("index", "turns_state_index"),
    ("table", "history_commitments"),
    ("table", "responses"),
    ("table", "run_finalization"),
    ("table", "run_manifest"),
    ("table", "schema_migrations"),
    ("table", "turn_metrics"),
    ("table", "turn_inputs"),
    ("table", "turns"),
    ("trigger", "comparisons_insert_after_analysis_finalization_guard"),
    ("trigger", "guardrails_insert_after_analysis_finalization_guard"),
    ("trigger", "turn_inputs_insert_after_run_finalization_guard"),
    ("trigger", "turns_forward_state_guard"),
    ("trigger", "turns_insert_after_finalization_guard"),
}


class SQLiteRunStore:
    """The single canonical mutable record for one experiment run.

    Every scientific payload is stored as canonical JSON alongside its digest.
    State transitions occur in ``BEGIN IMMEDIATE`` transactions.  The API has
    no transition that can send an uncertain or committed request back to the
    provider.
    """

    __slots__ = ("_closed", "_connection", "_path")

    def __init__(
        self,
        path: str | Path,
        manifest: RunManifest | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._path = Path(path)
        if self._path.exists() and self._path.is_dir():
            raise ValueError("SQLite run-store path must not be a directory")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        try:
            self._connection = sqlite3.connect(
                self._path,
                timeout=timeout_seconds,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
            self._initialize_schema()
            self._quick_check()
            if manifest is not None:
                self.bind_manifest(manifest)
        except (sqlite3.DatabaseError, OSError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise StoreCorruptionError(f"unable to open SQLite run store: {exc}") from exc

    @property
    def path(self) -> Path:
        """Return the canonical database path."""

        return self._path

    @property
    def schema_version(self) -> int:
        """Return the validated on-disk schema version."""

        self._ensure_open()
        row = self._connection.execute("PRAGMA user_version").fetchone()
        if row is None or not isinstance(row[0], int):
            raise StoreCorruptionError("database user_version is unreadable")
        return row[0]

    def __enter__(self) -> SQLiteRunStore:
        self._ensure_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Checkpoint recoverable WAL data and close the store."""

        if self._closed:
            return
        try:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            self._connection.close()
            self._closed = True

    def bind_manifest(self, manifest: RunManifest) -> str:
        """Bind one immutable run manifest and return its canonical digest."""

        if not isinstance(manifest, RunManifest):
            raise TypeError("manifest must be a RunManifest")
        if not 1 <= manifest.database_schema_version <= CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError(
                "manifest database_schema_version is not supported by this storage schema"
            )
        manifest_json = canonical_json(manifest)
        manifest_sha256 = canonical_sha256(manifest)
        with self._transaction():
            row = self._connection.execute(
                "SELECT manifest_json, manifest_sha256 FROM run_manifest WHERE singleton_id = 1"
            ).fetchone()
            if row is None:
                count_row = self._connection.execute("SELECT COUNT(*) FROM turns").fetchone()
                if count_row is None or count_row[0] != 0:
                    raise StoreCorruptionError("turns exist before a run manifest is bound")
                self._connection.execute(
                    """
                    INSERT INTO run_manifest(singleton_id, manifest_json, manifest_sha256)
                    VALUES (1, ?, ?)
                    """,
                    (manifest_json, manifest_sha256),
                )
            elif row[0] != manifest_json or row[1] != manifest_sha256:
                raise ManifestMismatchError("SQLite run store is bound to another manifest")
        return manifest_sha256

    def get_manifest(self) -> RunManifest | None:
        """Load and hash-validate the bound manifest, if present."""

        self._ensure_open()
        row = self._connection.execute(
            "SELECT manifest_json, manifest_sha256 FROM run_manifest WHERE singleton_id = 1"
        ).fetchone()
        if row is None:
            return None
        manifest_json = self._required_text(row, "manifest_json")
        manifest_sha256 = self._required_text(row, "manifest_sha256")
        manifest = self._decode_model(
            RunManifest,
            manifest_json,
            manifest_sha256,
            "run manifest",
        )
        if not 1 <= manifest.database_schema_version <= CURRENT_SCHEMA_VERSION:
            raise StoreCorruptionError("stored manifest names an unsupported database schema")
        return manifest

    def finalize_run(
        self,
        expected_condition_ids: tuple[str, ...],
        scientific_result_sha256: str,
        execution_accounting: DurableExecutionAccounting | None = None,
    ) -> RunFinalization:
        """Atomically close an exact, fully committed run schedule, idempotently."""

        if not isinstance(expected_condition_ids, tuple) or not all(
            isinstance(condition_id, str) for condition_id in expected_condition_ids
        ):
            raise TypeError("expected_condition_ids must be a tuple of strings")
        if not isinstance(scientific_result_sha256, str):
            raise TypeError("scientific_result_sha256 must be a string")
        if execution_accounting is not None and not isinstance(
            execution_accounting, DurableExecutionAccounting
        ):
            raise TypeError("execution_accounting must be DurableExecutionAccounting or None")
        canonical_condition_ids = tuple(sorted(expected_condition_ids))
        with self._transaction():
            manifest = self._require_manifest()
            finalization = RunFinalization(
                expected_condition_ids=canonical_condition_ids,
                expected_condition_count=len(canonical_condition_ids),
                manifest_sha256=canonical_sha256(manifest),
                scientific_result_sha256=scientific_result_sha256,
                execution_accounting=execution_accounting,
            )
            self._validate_finalization_against_store(finalization, stored=False)
            finalization_json = canonical_json(finalization)
            finalization_sha256 = canonical_sha256(finalization)
            row = self._connection.execute(
                """
                SELECT finalization_json, finalization_sha256
                FROM run_finalization
                WHERE singleton_id = 1
                """
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO run_finalization(
                        singleton_id,
                        finalization_json,
                        finalization_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (finalization_json, finalization_sha256),
                )
            else:
                stored_finalization = self._decode_model(
                    RunFinalization,
                    self._required_text(row, "finalization_json"),
                    self._required_text(row, "finalization_sha256"),
                    "run finalization",
                )
                if stored_finalization != finalization:
                    raise StoreInvariantError(
                        "run is already finalized with different closure evidence"
                    )
        return finalization

    def get_finalization(self) -> RunFinalization | None:
        """Load and fully validate the durable run-closure record, if present."""

        self._ensure_open()
        row = self._connection.execute(
            """
            SELECT finalization_json, finalization_sha256
            FROM run_finalization
            WHERE singleton_id = 1
            """
        ).fetchone()
        if row is None:
            return None
        finalization = self._decode_model(
            RunFinalization,
            self._required_text(row, "finalization_json"),
            self._required_text(row, "finalization_sha256"),
            "run finalization",
        )
        self._validate_finalization_against_store(finalization, stored=True)
        return finalization

    def persist_analysis(
        self,
        manifest: AnalysisManifest,
        result: Phase3EvaluationResult,
    ) -> AnalysisFinalization:
        """Atomically persist and close one complete Phase 3 analysis, idempotently."""

        if not isinstance(manifest, AnalysisManifest):
            raise TypeError("manifest must be an AnalysisManifest")
        if not isinstance(result, Phase3EvaluationResult):
            raise TypeError("result must be a Phase3EvaluationResult")
        comparisons = tuple(result.comparisons)
        guardrails = tuple(result.global_guardrails) + tuple(
            guardrail for comparison in comparisons for guardrail in comparison.guardrails
        )
        comparison_payloads = tuple(
            sorted(
                (
                    (
                        canonical_sha256(comparison),
                        canonical_json(comparison),
                        comparison,
                    )
                    for comparison in comparisons
                ),
                key=lambda payload: payload[0],
            )
        )
        guardrail_payloads = tuple(
            sorted(
                (
                    (canonical_sha256(guardrail), canonical_json(guardrail), guardrail)
                    for guardrail in guardrails
                ),
                key=lambda payload: payload[0],
            )
        )
        comparison_hashes = tuple(payload[0] for payload in comparison_payloads)
        guardrail_hashes = tuple(payload[0] for payload in guardrail_payloads)
        if len(comparison_hashes) != len(set(comparison_hashes)):
            raise StoreInvariantError("analysis contains duplicate comparison evidence")
        if not guardrail_hashes or len(guardrail_hashes) != len(set(guardrail_hashes)):
            raise StoreInvariantError("analysis guardrail evidence must be nonempty and unique")

        with self._transaction():
            self._validate_analysis_binding(manifest, result, stored=False)
            manifest_json = canonical_json(manifest)
            manifest_sha256 = canonical_sha256(manifest)
            manifest_row = self._connection.execute(
                """
                SELECT manifest_json, manifest_sha256
                FROM analysis_manifest
                WHERE singleton_id = 1
                """
            ).fetchone()
            if manifest_row is None:
                self._connection.execute(
                    """
                    INSERT INTO analysis_manifest(
                        singleton_id,
                        manifest_json,
                        manifest_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (manifest_json, manifest_sha256),
                )
            elif (
                self._required_text(manifest_row, "manifest_json") != manifest_json
                or self._required_text(manifest_row, "manifest_sha256") != manifest_sha256
            ):
                raise StoreInvariantError("run store is bound to another analysis manifest")

            for comparison_id, result_json, _ in comparison_payloads:
                self._persist_analysis_member(
                    table="comparison_results",
                    id_column="comparison_id",
                    member_id=comparison_id,
                    result_json=result_json,
                )
            for guardrail_id, result_json, _ in guardrail_payloads:
                self._persist_analysis_member(
                    table="guardrail_results",
                    id_column="guardrail_id",
                    member_id=guardrail_id,
                    result_json=result_json,
                )

            decision_json = canonical_json(result)
            decision_sha256 = canonical_sha256(result)
            decision_row = self._connection.execute(
                """
                SELECT decision_json, decision_sha256
                FROM analysis_decision
                WHERE singleton_id = 1
                """
            ).fetchone()
            if decision_row is None:
                self._connection.execute(
                    """
                    INSERT INTO analysis_decision(
                        singleton_id,
                        decision_json,
                        decision_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (decision_json, decision_sha256),
                )
            elif (
                self._required_text(decision_row, "decision_json") != decision_json
                or self._required_text(decision_row, "decision_sha256") != decision_sha256
            ):
                raise StoreInvariantError("analysis decision is already bound to another result")

            finalization = AnalysisFinalization(
                analysis_manifest_sha256=manifest_sha256,
                evaluation_result_sha256=result.result_sha256,
                decision_sha256=decision_sha256,
                comparison_result_sha256s=comparison_hashes,
                guardrail_result_sha256s=guardrail_hashes,
                comparison_count=len(comparison_hashes),
                guardrail_count=len(guardrail_hashes),
            )
            finalization_json = canonical_json(finalization)
            finalization_sha256 = canonical_sha256(finalization)
            finalization_row = self._connection.execute(
                """
                SELECT finalization_json, finalization_sha256
                FROM analysis_finalization
                WHERE singleton_id = 1
                """
            ).fetchone()
            if finalization_row is None:
                self._connection.execute(
                    """
                    INSERT INTO analysis_finalization(
                        singleton_id,
                        finalization_json,
                        finalization_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (finalization_json, finalization_sha256),
                )
            else:
                stored_finalization = self._decode_model(
                    AnalysisFinalization,
                    self._required_text(finalization_row, "finalization_json"),
                    self._required_text(finalization_row, "finalization_sha256"),
                    "analysis finalization",
                )
                if stored_finalization != finalization:
                    raise StoreInvariantError(
                        "analysis is already finalized with different closure evidence"
                    )
        return finalization

    def persist_scientific_analysis(
        self,
        manifest: ScientificAnalysisManifest,
        result: ConfirmatoryEvaluationResult,
        *,
        context: object,
    ) -> ScientificAnalysisFinalization:
        """Atomically persist one complete confirmatory analysis, idempotently."""

        if not isinstance(manifest, ScientificAnalysisManifest):
            raise TypeError("manifest must be a ScientificAnalysisManifest")
        if not isinstance(result, ConfirmatoryEvaluationResult):
            raise TypeError("result must be a ConfirmatoryEvaluationResult")
        from neurallm.experiments.scientific_analysis import ConfirmatoryAnalysisContext

        if not isinstance(context, ConfirmatoryAnalysisContext):
            raise TypeError("context must be a ConfirmatoryAnalysisContext")

        comparison_members: tuple[BaseModel, ...] = (
            *result.efficacy_comparisons,
            result.attribution,
        )
        if len(result.efficacy_comparisons) != 3 or len(comparison_members) != 4:
            raise StoreInvariantError(
                "scientific analysis requires exactly three efficacy comparisons and attribution"
            )
        comparison_payloads = tuple(
            sorted(
                (
                    (canonical_sha256(member), canonical_json(member), member)
                    for member in comparison_members
                ),
                key=lambda payload: payload[0],
            )
        )
        comparison_hashes = tuple(payload[0] for payload in comparison_payloads)
        if len(comparison_hashes) != len(set(comparison_hashes)):
            raise StoreInvariantError("scientific analysis contains duplicate comparison evidence")

        guardrails = self._scientific_guardrails(result, stored=False)
        guardrail_payloads = tuple(
            sorted(
                (
                    (canonical_sha256(guardrail), canonical_json(guardrail), guardrail)
                    for guardrail in guardrails
                ),
                key=lambda payload: payload[0],
            )
        )
        guardrail_hashes = tuple(payload[0] for payload in guardrail_payloads)

        with self._transaction():
            self._validate_scientific_analysis_binding(
                manifest,
                result,
                stored=False,
                context=context,
            )
            manifest_json = canonical_json(manifest)
            manifest_sha256 = canonical_sha256(manifest)
            manifest_row = self._connection.execute(
                """
                SELECT manifest_json, manifest_sha256
                FROM analysis_manifest
                WHERE singleton_id = 1
                """
            ).fetchone()
            if manifest_row is None:
                self._connection.execute(
                    """
                    INSERT INTO analysis_manifest(
                        singleton_id,
                        manifest_json,
                        manifest_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (manifest_json, manifest_sha256),
                )
            elif (
                self._required_text(manifest_row, "manifest_json") != manifest_json
                or self._required_text(manifest_row, "manifest_sha256") != manifest_sha256
            ):
                raise StoreInvariantError("run store is bound to another analysis manifest")

            for comparison_id, result_json, _ in comparison_payloads:
                self._persist_analysis_member(
                    table="comparison_results",
                    id_column="comparison_id",
                    member_id=comparison_id,
                    result_json=result_json,
                )
            for guardrail_id, result_json, _ in guardrail_payloads:
                self._persist_analysis_member(
                    table="guardrail_results",
                    id_column="guardrail_id",
                    member_id=guardrail_id,
                    result_json=result_json,
                )

            decision_json = canonical_json(result)
            decision_sha256 = canonical_sha256(result)
            decision_row = self._connection.execute(
                """
                SELECT decision_json, decision_sha256
                FROM analysis_decision
                WHERE singleton_id = 1
                """
            ).fetchone()
            if decision_row is None:
                self._connection.execute(
                    """
                    INSERT INTO analysis_decision(
                        singleton_id,
                        decision_json,
                        decision_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (decision_json, decision_sha256),
                )
            elif (
                self._required_text(decision_row, "decision_json") != decision_json
                or self._required_text(decision_row, "decision_sha256") != decision_sha256
            ):
                raise StoreInvariantError(
                    "scientific analysis decision is already bound to another result"
                )

            finalization = ScientificAnalysisFinalization(
                analysis_manifest_sha256=manifest_sha256,
                evaluation_result_sha256=result.result_sha256,
                decision_sha256=decision_sha256,
                comparison_result_sha256s=comparison_hashes,
                guardrail_result_sha256s=guardrail_hashes,
                guardrail_count=len(guardrail_hashes),
            )
            finalization_json = canonical_json(finalization)
            finalization_sha256 = canonical_sha256(finalization)
            finalization_row = self._connection.execute(
                """
                SELECT finalization_json, finalization_sha256
                FROM analysis_finalization
                WHERE singleton_id = 1
                """
            ).fetchone()
            if finalization_row is None:
                self._connection.execute(
                    """
                    INSERT INTO analysis_finalization(
                        singleton_id,
                        finalization_json,
                        finalization_sha256
                    ) VALUES (1, ?, ?)
                    """,
                    (finalization_json, finalization_sha256),
                )
            else:
                stored_finalization = self._decode_model(
                    ScientificAnalysisFinalization,
                    self._required_text(finalization_row, "finalization_json"),
                    self._required_text(finalization_row, "finalization_sha256"),
                    "scientific analysis finalization",
                )
                if stored_finalization != finalization:
                    raise StoreInvariantError(
                        "scientific analysis is already finalized with different evidence"
                    )
        return finalization

    def get_analysis(self) -> StoredAnalysis | None:
        """Load and cross-validate the finalized Phase 3 analysis, if present."""

        self._ensure_open()
        manifest_row = self._connection.execute(
            """
            SELECT manifest_json, manifest_sha256
            FROM analysis_manifest
            WHERE singleton_id = 1
            """
        ).fetchone()
        if manifest_row is None:
            counts = tuple(
                self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "comparison_results",
                    "guardrail_results",
                    "analysis_decision",
                    "analysis_finalization",
                )
            )
            if any(counts):
                raise StoreCorruptionError("analysis evidence exists without a manifest")
            return None

        implementation_version = self._analysis_manifest_implementation_version(manifest_row)
        if implementation_version == _SCIENTIFIC_ANALYSIS_IMPLEMENTATION_VERSION:
            return None

        manifest = self._decode_model(
            AnalysisManifest,
            self._required_text(manifest_row, "manifest_json"),
            self._required_text(manifest_row, "manifest_sha256"),
            "analysis manifest",
        )
        comparison_rows = self._connection.execute(
            """
            SELECT comparison_id, result_json, result_sha256
            FROM comparison_results
            ORDER BY comparison_id
            """
        ).fetchall()
        comparisons = tuple(
            self._decode_analysis_member(
                PairwiseComparisonResult,
                row,
                id_column="comparison_id",
                label="comparison result",
            )
            for row in comparison_rows
        )
        guardrail_rows = self._connection.execute(
            """
            SELECT guardrail_id, result_json, result_sha256
            FROM guardrail_results
            ORDER BY guardrail_id
            """
        ).fetchall()
        guardrails = tuple(
            self._decode_analysis_member(
                GuardrailResult,
                row,
                id_column="guardrail_id",
                label="guardrail result",
            )
            for row in guardrail_rows
        )
        decision_row = self._connection.execute(
            """
            SELECT decision_json, decision_sha256
            FROM analysis_decision
            WHERE singleton_id = 1
            """
        ).fetchone()
        finalization_row = self._connection.execute(
            """
            SELECT finalization_json, finalization_sha256
            FROM analysis_finalization
            WHERE singleton_id = 1
            """
        ).fetchone()
        if decision_row is None or finalization_row is None:
            raise StoreCorruptionError("analysis manifest is not atomically finalized")
        result = self._decode_model(
            Phase3EvaluationResult,
            self._required_text(decision_row, "decision_json"),
            self._required_text(decision_row, "decision_sha256"),
            "analysis decision",
        )
        finalization = self._decode_model(
            AnalysisFinalization,
            self._required_text(finalization_row, "finalization_json"),
            self._required_text(finalization_row, "finalization_sha256"),
            "analysis finalization",
        )
        self._validate_analysis_binding(manifest, result, stored=True)
        expected_comparison_hashes = tuple(
            sorted(canonical_sha256(comparison) for comparison in result.comparisons)
        )
        expected_guardrails = tuple(result.global_guardrails) + tuple(
            guardrail for comparison in result.comparisons for guardrail in comparison.guardrails
        )
        expected_guardrail_hashes = tuple(
            sorted(canonical_sha256(guardrail) for guardrail in expected_guardrails)
        )
        actual_comparison_hashes = tuple(
            sorted(canonical_sha256(comparison) for comparison in comparisons)
        )
        actual_guardrail_hashes = tuple(
            sorted(canonical_sha256(guardrail) for guardrail in guardrails)
        )
        expected_finalization = AnalysisFinalization(
            analysis_manifest_sha256=canonical_sha256(manifest),
            evaluation_result_sha256=result.result_sha256,
            decision_sha256=canonical_sha256(result),
            comparison_result_sha256s=expected_comparison_hashes,
            guardrail_result_sha256s=expected_guardrail_hashes,
            comparison_count=len(expected_comparison_hashes),
            guardrail_count=len(expected_guardrail_hashes),
        )
        if (
            actual_comparison_hashes != expected_comparison_hashes
            or actual_guardrail_hashes != expected_guardrail_hashes
            or finalization != expected_finalization
        ):
            raise StoreCorruptionError(
                "persisted analysis members do not match the finalized decision"
            )
        return StoredAnalysis(
            manifest=manifest,
            result=result,
            comparisons=tuple(result.comparisons),
            guardrails=expected_guardrails,
            finalization=finalization,
        )

    def get_scientific_analysis(self) -> StoredScientificAnalysis | None:
        """Load and cross-validate the finalized confirmatory analysis, if present."""

        self._ensure_open()
        manifest_row = self._connection.execute(
            """
            SELECT manifest_json, manifest_sha256
            FROM analysis_manifest
            WHERE singleton_id = 1
            """
        ).fetchone()
        if manifest_row is None:
            counts = tuple(
                self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "comparison_results",
                    "guardrail_results",
                    "analysis_decision",
                    "analysis_finalization",
                )
            )
            if any(counts):
                raise StoreCorruptionError("analysis evidence exists without a manifest")
            return None

        implementation_version = self._analysis_manifest_implementation_version(manifest_row)
        if implementation_version == _PHASE3_ANALYSIS_IMPLEMENTATION_VERSION:
            return None

        manifest = self._decode_model(
            ScientificAnalysisManifest,
            self._required_text(manifest_row, "manifest_json"),
            self._required_text(manifest_row, "manifest_sha256"),
            "scientific analysis manifest",
        )
        comparison_rows = self._connection.execute(
            """
            SELECT comparison_id, result_json, result_sha256
            FROM comparison_results
            ORDER BY comparison_id
            """
        ).fetchall()
        comparison_members = tuple(
            self._decode_scientific_comparison(row) for row in comparison_rows
        )
        efficacy_members = tuple(
            member for member in comparison_members if isinstance(member, EfficacyComparisonResult)
        )
        attribution_members = tuple(
            member
            for member in comparison_members
            if isinstance(member, PersistentStateAttributionResult)
        )
        if (
            len(efficacy_members) != 3
            or len(attribution_members) != 1
            or len(comparison_members) != 4
        ):
            raise StoreCorruptionError(
                "scientific analysis must persist three efficacy comparisons and attribution"
            )

        guardrail_rows = self._connection.execute(
            """
            SELECT guardrail_id, result_json, result_sha256
            FROM guardrail_results
            ORDER BY guardrail_id
            """
        ).fetchall()
        guardrails = tuple(
            self._decode_analysis_member(
                ScientificGuardrailResult,
                row,
                id_column="guardrail_id",
                label="scientific guardrail result",
            )
            for row in guardrail_rows
        )
        decision_row = self._connection.execute(
            """
            SELECT decision_json, decision_sha256
            FROM analysis_decision
            WHERE singleton_id = 1
            """
        ).fetchone()
        finalization_row = self._connection.execute(
            """
            SELECT finalization_json, finalization_sha256
            FROM analysis_finalization
            WHERE singleton_id = 1
            """
        ).fetchone()
        if decision_row is None or finalization_row is None:
            raise StoreCorruptionError("scientific analysis manifest is not atomically finalized")
        result = self._decode_model(
            ConfirmatoryEvaluationResult,
            self._required_text(decision_row, "decision_json"),
            self._required_text(decision_row, "decision_sha256"),
            "scientific analysis decision",
        )
        finalization = self._decode_model(
            ScientificAnalysisFinalization,
            self._required_text(finalization_row, "finalization_json"),
            self._required_text(finalization_row, "finalization_sha256"),
            "scientific analysis finalization",
        )
        self._validate_scientific_analysis_binding(manifest, result, stored=True)

        expected_comparisons: tuple[BaseModel, ...] = (
            *result.efficacy_comparisons,
            result.attribution,
        )
        expected_comparison_hashes = tuple(
            sorted(canonical_sha256(member) for member in expected_comparisons)
        )
        expected_guardrails = self._scientific_guardrails(result, stored=True)
        expected_guardrail_hashes = tuple(
            sorted(canonical_sha256(guardrail) for guardrail in expected_guardrails)
        )
        actual_comparison_hashes = tuple(
            sorted(canonical_sha256(member) for member in comparison_members)
        )
        actual_guardrail_hashes = tuple(
            sorted(canonical_sha256(guardrail) for guardrail in guardrails)
        )
        expected_finalization = ScientificAnalysisFinalization(
            analysis_manifest_sha256=canonical_sha256(manifest),
            evaluation_result_sha256=result.result_sha256,
            decision_sha256=canonical_sha256(result),
            comparison_result_sha256s=expected_comparison_hashes,
            guardrail_result_sha256s=expected_guardrail_hashes,
            guardrail_count=len(expected_guardrail_hashes),
        )
        if (
            actual_comparison_hashes != expected_comparison_hashes
            or actual_guardrail_hashes != expected_guardrail_hashes
            or finalization != expected_finalization
        ):
            raise StoreCorruptionError(
                "persisted scientific analysis members do not match the finalized decision"
            )
        return StoredScientificAnalysis(
            manifest=manifest,
            result=result,
            efficacy_comparisons=result.efficacy_comparisons,
            attribution=result.attribution,
            guardrails=expected_guardrails,
            finalization=finalization,
        )

    def prepare_turn(
        self,
        request: GenerationRequest,
        history: HistoryBinding | None = None,
        input_evidence: TurnInputEvidence | None = None,
    ) -> StoredTurn:
        """Persist a logical request before dispatch, idempotently if identical."""

        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        if history is not None and not isinstance(history, HistoryBinding):
            raise TypeError("history must be a HistoryBinding or None")
        if input_evidence is not None and not isinstance(input_evidence, TurnInputEvidence):
            raise TypeError("input_evidence must be a TurnInputEvidence or None")
        condition = request.condition
        condition_json = canonical_json(condition)
        request_json = canonical_json(request)
        condition_id = condition.condition_id
        request_sha256 = canonical_sha256(request)
        if input_evidence is not None and input_evidence.condition_id != condition_id:
            raise ValueError("turn input evidence targets another condition")
        try:
            with self._transaction():
                manifest = self._require_manifest()
                finalization = self.get_finalization()
                if condition.provider_identity_id != manifest.provider_identity.identity_id:
                    raise StoreInvariantError(
                        "request provider identity does not match the bound run manifest"
                    )
                self._validate_history_binding(condition, history)
                existing = self._read_turn_or_none(condition_id)
                if existing is not None:
                    if (
                        existing.request_sha256 != request_sha256
                        or existing.request != request
                        or existing.history != history
                    ):
                        raise DuplicateLogicalRequestError(
                            "condition is already bound to different request or history data"
                        )
                    if input_evidence is not None:
                        self._bind_turn_input(input_evidence)
                    return existing
                if finalization is not None:
                    raise StoreInvariantError("cannot prepare a new turn after run finalization")
                self._connection.execute(
                    """
                    INSERT INTO turns(
                        condition_id,
                        request_sha256,
                        condition_json,
                        request_json,
                        experiment_id,
                        dataset_version,
                        prompt_sequence_id,
                        turn_index,
                        policy_id,
                        model_seed,
                        controller_seed,
                        provider_identity_id,
                        base_decoding_profile_id,
                        previous_condition_id,
                        previous_history_commitment_sha256,
                        state,
                        uncertain_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        condition_id,
                        request_sha256,
                        condition_json,
                        request_json,
                        condition.experiment_id,
                        condition.dataset_version,
                        condition.prompt_sequence_id,
                        condition.turn_index,
                        condition.policy_id,
                        condition.model_seed,
                        condition.controller_seed,
                        condition.provider_identity_id,
                        condition.base_decoding_profile_id,
                        history.previous_condition_id if history else None,
                        history.previous_history_commitment_sha256 if history else None,
                        TurnState.PREPARED.value,
                    ),
                )
                if input_evidence is not None:
                    self._bind_turn_input(input_evidence)
                return self._read_turn(condition_id)
        except sqlite3.IntegrityError as exc:
            raise DuplicateLogicalRequestError(
                "unique logical request or condition constraint rejected the turn"
            ) from exc

    def get_turn_input(self, condition_id: str) -> TurnInputEvidence | None:
        """Load and hash-validate complete prompt-side evidence for one turn."""

        self._ensure_open()
        row = self._connection.execute(
            "SELECT input_json, input_sha256 FROM turn_inputs WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None:
            return None
        evidence = self._decode_model(
            TurnInputEvidence,
            self._required_text(row, "input_json"),
            self._required_text(row, "input_sha256"),
            "turn input evidence",
        )
        if evidence.condition_id != condition_id:
            raise StoreCorruptionError("turn input evidence targets another condition")
        return evidence

    def list_turn_inputs(self) -> tuple[TurnInputEvidence, ...]:
        """Return every validated prompt-side evidence record in condition order."""

        self._ensure_open()
        rows = self._connection.execute(
            "SELECT condition_id FROM turn_inputs ORDER BY condition_id"
        ).fetchall()
        evidence: list[TurnInputEvidence] = []
        for row in rows:
            condition_id = self._required_text(row, "condition_id")
            record = self.get_turn_input(condition_id)
            if record is None:
                raise StoreCorruptionError("turn input evidence disappeared during read")
            evidence.append(record)
        return tuple(evidence)

    def begin_dispatch(self, condition_id: str) -> StoredTurn:
        """Atomically mark a prepared request as entering its sole dispatch."""

        became_uncertain = False
        with self._transaction():
            turn = self._read_turn(condition_id)
            if turn.state is TurnState.DISPATCHING:
                self._set_uncertain(
                    condition_id,
                    "dispatch was requested again without a persisted response",
                )
                became_uncertain = True
            elif turn.state is TurnState.UNCERTAIN_DISPATCH:
                became_uncertain = True
            elif turn.state is not TurnState.PREPARED:
                raise StateTransitionError(f"cannot dispatch a turn in state {turn.state.value}")
            else:
                self._connection.execute(
                    "UPDATE turns SET state = ? WHERE condition_id = ?",
                    (TurnState.DISPATCHING.value, condition_id),
                )
        if became_uncertain:
            raise UncertainDispatchError(
                "request may already have reached the provider and cannot be retried"
            )
        return self.get_turn(condition_id)

    def mark_dispatch_uncertain(self, condition_id: str, reason: str) -> StoredTurn:
        """Record a provider-call failure after the durable dispatch boundary."""

        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be blank")
        with self._transaction():
            turn = self._read_turn(condition_id)
            if turn.state is not TurnState.DISPATCHING:
                raise StateTransitionError("only a DISPATCHING turn can become UNCERTAIN_DISPATCH")
            self._set_uncertain(condition_id, reason)
        return self.get_turn(condition_id)

    def persist_response(
        self,
        condition_id: str,
        response: GenerationResponse,
    ) -> StoredTurn:
        """Persist the raw validated response in the dispatch transaction stage."""

        if not isinstance(response, GenerationResponse):
            raise TypeError("response must be a GenerationResponse")
        response_json = canonical_json(response)
        response_sha256 = canonical_sha256(response)
        with self._transaction():
            turn = self._read_turn(condition_id)
            if turn.state is not TurnState.DISPATCHING:
                raise StateTransitionError(f"cannot persist a response in state {turn.state.value}")
            self._validate_response_against_request(response, turn.request)
            self._connection.execute(
                """
                INSERT INTO responses(condition_id, response_json, response_sha256)
                VALUES (?, ?, ?)
                """,
                (condition_id, response_json, response_sha256),
            )
            self._connection.execute(
                "UPDATE turns SET state = ? WHERE condition_id = ?",
                (TurnState.RESPONSE_PERSISTED.value, condition_id),
            )
        return self.get_turn(condition_id)

    def persist_metrics(
        self,
        condition_id: str,
        metrics: ResponseMetrics,
    ) -> StoredTurn:
        """Persist deterministic metrics after a raw response exists."""

        if not isinstance(metrics, ResponseMetrics):
            raise TypeError("metrics must be ResponseMetrics")
        metrics_json = canonical_json(metrics)
        metrics_sha256 = canonical_sha256(metrics)
        with self._transaction():
            turn = self._read_turn(condition_id)
            if turn.state is not TurnState.RESPONSE_PERSISTED:
                raise StateTransitionError(f"cannot persist metrics in state {turn.state.value}")
            self._connection.execute(
                """
                INSERT INTO turn_metrics(condition_id, metrics_json, metrics_sha256)
                VALUES (?, ?, ?)
                """,
                (condition_id, metrics_json, metrics_sha256),
            )
            self._connection.execute(
                "UPDATE turns SET state = ? WHERE condition_id = ?",
                (TurnState.METRICS_COMPUTED.value, condition_id),
            )
        return self.get_turn(condition_id)

    def commit_turn(
        self,
        condition_id: str,
        policy_state: PolicyState,
        policy_trace: PolicyTrace,
    ) -> CommittedHistory:
        """Commit policy state and the only history visible to a later turn."""

        if not isinstance(policy_state, PolicyState):
            raise TypeError("policy_state must be a PolicyState")
        if not isinstance(policy_trace, PolicyTrace):
            raise TypeError("policy_trace must be a PolicyTrace")
        policy_state_json = canonical_json(policy_state)
        policy_state_sha256 = canonical_sha256(policy_state)
        policy_trace_json = canonical_json(policy_trace)
        policy_trace_sha256 = canonical_sha256(policy_trace)
        with self._transaction():
            turn = self._read_turn(condition_id)
            if turn.state is not TurnState.METRICS_COMPUTED:
                raise StateTransitionError(f"cannot commit a turn in state {turn.state.value}")
            if policy_trace.policy_id != turn.condition.policy_id:
                raise StoreInvariantError("policy trace policy_id does not match the condition")
            if policy_trace.turn_index != turn.condition.turn_index:
                raise StoreInvariantError("policy trace turn_index does not match the condition")
            manifest = self._require_manifest()
            try:
                manifest.action_bounds.require(policy_trace.action)
            except ValueError as exc:
                raise StoreInvariantError("policy trace action exceeds manifest bounds") from exc
            row = self._connection.execute(
                """
                SELECT r.response_sha256, m.metrics_sha256
                FROM responses AS r
                JOIN turn_metrics AS m ON m.condition_id = r.condition_id
                WHERE r.condition_id = ?
                """,
                (condition_id,),
            ).fetchone()
            if row is None:
                raise StoreCorruptionError("metric checkpoint is missing response evidence")
            response_sha256 = self._required_text(row, "response_sha256")
            metrics_sha256 = self._required_text(row, "metrics_sha256")
            previous_commitment = (
                turn.history.previous_history_commitment_sha256 if turn.history else None
            )
            history_commitment_sha256 = canonical_sha256(
                {
                    "condition_id": condition_id,
                    "request_sha256": turn.request_sha256,
                    "response_sha256": response_sha256,
                    "metrics_sha256": metrics_sha256,
                    "policy_state_sha256": policy_state_sha256,
                    "policy_trace_sha256": policy_trace_sha256,
                    "previous_history_commitment_sha256": previous_commitment,
                }
            )
            self._connection.execute(
                """
                INSERT INTO history_commitments(
                    condition_id,
                    policy_state_json,
                    policy_state_sha256,
                    policy_trace_json,
                    policy_trace_sha256,
                    history_commitment_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    condition_id,
                    policy_state_json,
                    policy_state_sha256,
                    policy_trace_json,
                    policy_trace_sha256,
                    history_commitment_sha256,
                ),
            )
            self._connection.execute(
                "UPDATE turns SET state = ? WHERE condition_id = ?",
                (TurnState.COMMITTED.value, condition_id),
            )
        return self.get_committed_history(condition_id)

    def resume_action(self, condition_id: str) -> ResumeAction:
        """Return the only safe continuation, failing closed on dispatch ambiguity."""

        became_uncertain = False
        action: ResumeAction | None = None
        with self._transaction():
            turn = self._read_turn(condition_id)
            if turn.state is TurnState.PREPARED:
                action = ResumeAction.DISPATCH_PREPARED
            elif turn.state is TurnState.DISPATCHING:
                self._set_uncertain(
                    condition_id,
                    "store resumed after dispatch without a persisted response",
                )
                became_uncertain = True
            elif turn.state is TurnState.UNCERTAIN_DISPATCH:
                became_uncertain = True
            elif turn.state is TurnState.RESPONSE_PERSISTED:
                action = ResumeAction.COMPUTE_METRICS
            elif turn.state is TurnState.METRICS_COMPUTED:
                action = ResumeAction.COMMIT
            elif turn.state is TurnState.COMMITTED:
                action = ResumeAction.SKIP_COMMITTED
        if became_uncertain:
            raise UncertainDispatchError(
                "dispatched request lacks a safely committed response and cannot be retried"
            )
        if action is None:
            raise StoreCorruptionError("turn has no recognized resume action")
        return action

    def get_turn(self, condition_id: str) -> StoredTurn:
        """Load one turn and validate all persisted hashes and checkpoints."""

        self._ensure_open()
        return self._read_turn(condition_id)

    def list_turns(self, state: TurnState | None = None) -> tuple[StoredTurn, ...]:
        """Return validated turns in deterministic schedule order."""

        self._ensure_open()
        if state is not None and not isinstance(state, TurnState):
            raise TypeError("state must be a TurnState or None")
        sql = "SELECT condition_id FROM turns"
        parameters: tuple[str, ...] = ()
        if state is not None:
            sql += " WHERE state = ?"
            parameters = (state.value,)
        sql += (
            " ORDER BY experiment_id, dataset_version, prompt_sequence_id, model_seed, "
            "controller_seed, policy_id, turn_index"
        )
        rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(self._read_turn(self._required_text(row, "condition_id")) for row in rows)

    def get_committed_history(self, condition_id: str) -> CommittedHistory:
        """Load the hash-verified committed history for one turn."""

        turn = self.get_turn(condition_id)
        if turn.state is not TurnState.COMMITTED:
            raise HistoryMismatchError("requested turn has not committed causal history")
        if (
            turn.response is None
            or turn.metrics is None
            or turn.policy_state_json is None
            or turn.policy_trace_json is None
            or turn.history_commitment_sha256 is None
        ):
            raise StoreCorruptionError("committed turn is missing durable evidence")
        return CommittedHistory(
            condition_id=turn.condition_id,
            condition=turn.condition,
            request_sha256=turn.request_sha256,
            response_sha256=canonical_sha256(turn.response),
            metrics=turn.metrics,
            policy_state_json=turn.policy_state_json,
            policy_trace_json=turn.policy_trace_json,
            previous_history_commitment_sha256=(
                turn.history.previous_history_commitment_sha256 if turn.history else None
            ),
            history_commitment_sha256=turn.history_commitment_sha256,
        )

    def history_binding_for(self, condition_id: str) -> HistoryBinding:
        """Return the exact binding a later turn must persist."""

        committed = self.get_committed_history(condition_id)
        return HistoryBinding(
            previous_condition_id=condition_id,
            previous_history_commitment_sha256=committed.history_commitment_sha256,
        )

    def load_policy_state(
        self,
        condition_id: str,
        state_type: type[PolicyStateT],
    ) -> PolicyStateT:
        """Rehydrate a committed controller-state subclass chosen by the runner."""

        if not issubclass(state_type, PolicyState):
            raise TypeError("state_type must be a PolicyState subclass")
        history = self.get_committed_history(condition_id)
        try:
            return state_type.model_validate_json(history.policy_state_json)
        except ValidationError as exc:
            raise StoreCorruptionError("stored policy state fails its declared model") from exc

    def verify_integrity(self) -> None:
        """Validate SQLite, foreign keys, manifest, and every scientific digest."""

        self._ensure_open()
        rows = self._connection.execute("PRAGMA integrity_check").fetchall()
        messages = tuple(row[0] for row in rows)
        if messages != ("ok",):
            raise StoreCorruptionError(f"SQLite integrity_check failed: {messages!r}")
        foreign_key_rows = self._connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            raise StoreCorruptionError("SQLite foreign-key integrity check failed")
        application_row = self._connection.execute("PRAGMA application_id").fetchone()
        if application_row is None or application_row[0] != APPLICATION_ID:
            raise StoreCorruptionError("database application_id is invalid")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError("database schema version changed unexpectedly")
        self.get_manifest()
        self.list_turns()
        self.list_turn_inputs()
        self.get_finalization()
        self.get_analysis()
        self.get_scientific_analysis()

    def compact(self) -> None:
        """Checkpoint and compact the single SQLite artifact in place."""

        self.verify_integrity()
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._connection.execute("VACUUM")
        self._connection.execute("PRAGMA optimize")
        self.verify_integrity()

    def _initialize_schema(self) -> None:
        application_row = self._connection.execute("PRAGMA application_id").fetchone()
        version_row = self._connection.execute("PRAGMA user_version").fetchone()
        if application_row is None or version_row is None:
            raise SchemaVersionError("SQLite schema metadata is unreadable")
        application_id = application_row[0]
        version = version_row[0]
        if not isinstance(application_id, int) or not isinstance(version, int):
            raise SchemaVersionError("SQLite schema metadata has invalid types")
        user_tables = {
            row[0]
            for row in self._connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        }
        if application_id == 0:
            if user_tables:
                raise SchemaVersionError("refusing to adopt an unmanaged SQLite database")
        elif application_id != APPLICATION_ID:
            raise SchemaVersionError("SQLite database belongs to another application")
        if version > CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError("SQLite schema is newer than this NeuraLLM build")
        for migration in MIGRATIONS:
            if migration.version <= version:
                continue
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in migration.statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                    (migration.version, migration.name),
                )
                self._connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                self._connection.execute(f"PRAGMA user_version = {migration.version}")
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            version = migration.version
        migration_rows = self._connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        expected = tuple(
            (migration.version, migration.name)
            for migration in MIGRATIONS
            if migration.version <= version
        )
        actual = tuple((row[0], row[1]) for row in migration_rows)
        if actual != expected or version != CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError("SQLite migration history is incomplete or mismatched")
        self._validate_schema_objects()

    def _validate_schema_objects(self) -> None:
        rows = self._connection.execute(
            """
            SELECT type, name
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        actual = {(row[0], row[1]) for row in rows}
        if actual != _EXPECTED_SCHEMA_OBJECTS:
            raise SchemaVersionError("SQLite schema objects do not match the applied migrations")

    def _quick_check(self) -> None:
        row = self._connection.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            raise StoreCorruptionError("SQLite quick_check did not return ok")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._ensure_open()
        if self._connection.in_transaction:
            raise StoreInvariantError("nested storage transactions are not supported")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _require_manifest(self) -> RunManifest:
        manifest = self.get_manifest()
        if manifest is None:
            raise StoreInvariantError("a run manifest must be bound before preparing turns")
        return manifest

    def _analysis_manifest_implementation_version(self, row: sqlite3.Row) -> str:
        payload = self._decode_json_object(
            self._required_text(row, "manifest_json"),
            self._required_text(row, "manifest_sha256"),
            "analysis manifest",
        )
        implementation_version = payload.get("implementation_version")
        if not isinstance(implementation_version, str):
            raise StoreCorruptionError(
                "analysis manifest has no string implementation_version discriminant"
            )
        if implementation_version == _LEGACY_SCIENTIFIC_ANALYSIS_IMPLEMENTATION_VERSION:
            raise StoreCorruptionError(
                "legacy confirmatory scientific analysis v1 is incompatible with the "
                "contract-bound v2 evidence envelope; rerun the analysis"
            )
        if implementation_version not in _SUPPORTED_ANALYSIS_IMPLEMENTATION_VERSIONS:
            raise StoreCorruptionError("analysis manifest has an unknown implementation_version")
        return implementation_version

    def _decode_scientific_comparison(
        self,
        row: sqlite3.Row,
    ) -> EfficacyComparisonResult | PersistentStateAttributionResult:
        comparison_id = self._required_text(row, "comparison_id")
        result_sha256 = self._required_text(row, "result_sha256")
        if comparison_id != result_sha256:
            raise StoreCorruptionError(
                "scientific comparison result identifier does not match its digest"
            )
        payload = self._decode_json_object(
            self._required_text(row, "result_json"),
            result_sha256,
            "scientific comparison result",
        )
        comparison_kind = payload.get("comparison_kind")
        if comparison_kind == "efficacy":
            return self._decode_analysis_member(
                EfficacyComparisonResult,
                row,
                id_column="comparison_id",
                label="scientific comparison result",
            )
        if comparison_kind == "persistent_state_attribution":
            return self._decode_analysis_member(
                PersistentStateAttributionResult,
                row,
                id_column="comparison_id",
                label="scientific comparison result",
            )
        raise StoreCorruptionError(
            "scientific comparison has an unknown comparison_kind discriminant"
        )

    @staticmethod
    def _scientific_guardrails(
        result: ConfirmatoryEvaluationResult,
        *,
        stored: bool,
    ) -> tuple[ScientificGuardrailResult, ...]:
        candidates = (
            tuple(result.guardrails)
            + tuple(
                guardrail
                for comparison in result.efficacy_comparisons
                for guardrail in comparison.guardrails
            )
            + tuple(result.attribution.causal_guardrails)
        )
        by_key: dict[tuple[str, str], ScientificGuardrailResult] = {}
        for guardrail in candidates:
            existing = by_key.get(guardrail.evidence_key)
            if existing is not None and existing != guardrail:
                if stored:
                    raise StoreCorruptionError(
                        "scientific guardrail key is bound to conflicting evidence"
                    )
                raise StoreInvariantError(
                    "scientific guardrail key is bound to conflicting evidence"
                )
            by_key[guardrail.evidence_key] = guardrail
        return tuple(by_key[key] for key in sorted(by_key))

    def _persist_analysis_member(
        self,
        *,
        table: str,
        id_column: str,
        member_id: str,
        result_json: str,
    ) -> None:
        if not self._connection.in_transaction:
            raise StoreInvariantError("analysis members must be persisted transactionally")
        if (table, id_column) not in {
            ("comparison_results", "comparison_id"),
            ("guardrail_results", "guardrail_id"),
        }:
            raise StoreInvariantError("unsupported analysis member table")
        result_sha256 = canonical_sha256(json.loads(result_json))
        row = self._connection.execute(
            f"SELECT result_json, result_sha256 FROM {table} WHERE {id_column} = ?",
            (member_id,),
        ).fetchone()
        if row is None:
            self._connection.execute(
                f"""
                INSERT INTO {table}({id_column}, result_json, result_sha256)
                VALUES (?, ?, ?)
                """,
                (member_id, result_json, result_sha256),
            )
            return
        if (
            self._required_text(row, "result_json") != result_json
            or self._required_text(row, "result_sha256") != result_sha256
        ):
            raise StoreInvariantError("analysis member identifier is bound to other evidence")

    def _decode_analysis_member(
        self,
        model_type: type[ModelT],
        row: sqlite3.Row,
        *,
        id_column: str,
        label: str,
    ) -> ModelT:
        member_id = self._required_text(row, id_column)
        result_sha256 = self._required_text(row, "result_sha256")
        if member_id != result_sha256:
            raise StoreCorruptionError(f"{label} identifier does not match its digest")
        return self._decode_model(
            model_type,
            self._required_text(row, "result_json"),
            result_sha256,
            label,
        )

    def _validate_analysis_binding(
        self,
        manifest: AnalysisManifest,
        result: Phase3EvaluationResult,
        *,
        stored: bool,
    ) -> None:
        run_manifest = self._require_manifest()
        run_finalization = self.get_finalization()

        def fail(message: str) -> None:
            if stored:
                raise StoreCorruptionError(message)
            raise StoreInvariantError(message)

        if run_finalization is None:
            fail("Phase 3 analysis requires a finalized run")
            return
        if (
            run_manifest.database_schema_version != CURRENT_SCHEMA_VERSION
            or run_manifest.decision_rule_version != "phase3-baseline-evaluator-v1"
        ):
            fail("Phase 3 analysis requires a schema-v2 Phase 3 run manifest")
        if manifest.run_manifest_sha256 != canonical_sha256(run_manifest):
            fail("analysis manifest does not match the run manifest")
        if manifest.run_finalization_sha256 != canonical_sha256(run_finalization):
            fail("analysis manifest does not match the run finalization")
        if manifest.scientific_result_sha256 != run_finalization.scientific_result_sha256:
            fail("analysis manifest does not match the finalized scientific result")
        if manifest.dataset_sha256 != run_manifest.dataset_hash:
            fail("analysis manifest does not match the run dataset")
        if manifest.evaluation_input_sha256 != result.input_sha256:
            fail("analysis manifest does not match the evaluator input")
        try:
            expected_contract_sha256 = phase3_analysis_contract_sha256(
                experiment_plan_sha256=manifest.experiment_plan_sha256,
                evaluation_spec=manifest.evaluation_spec,
                evaluation_spec_sha256=manifest.evaluation_spec_sha256,
                static_selection_record=manifest.static_selection_record,
                static_selection_result_sha256=manifest.static_selection_result_sha256,
                evaluation_design=manifest.evaluation_design,
                dataset_sha256=manifest.dataset_sha256,
                dataset_purpose=manifest.dataset_purpose,
                dataset_seal_sha256=manifest.dataset_seal_sha256,
            )
        except ValueError as exc:
            fail(f"analysis contract evidence is internally inconsistent: {exc}")
            return
        if run_manifest.phase3_analysis_contract_sha256 != expected_contract_sha256:
            fail("analysis evidence does not match the pre-execution Phase 3 contract")

    def _validate_scientific_analysis_binding(
        self,
        manifest: ScientificAnalysisManifest,
        result: ConfirmatoryEvaluationResult,
        *,
        stored: bool,
        context: object | None = None,
    ) -> None:
        run_manifest = self._require_manifest()
        run_finalization = self.get_finalization()

        def fail(message: str) -> None:
            if stored:
                raise StoreCorruptionError(message)
            raise StoreInvariantError(message)

        if run_finalization is None:
            fail("scientific analysis requires a finalized confirmatory run")
            return
        if run_finalization.execution_accounting is None:
            fail("scientific analysis requires durable execution accounting")
        if (
            run_manifest.database_schema_version != CURRENT_SCHEMA_VERSION
            or run_manifest.decision_rule_version != "confirmatory-scientific-decision-v2"
            or run_manifest.run_tier != "confirmatory"
            or not run_manifest.working_tree_clean
            or run_manifest.confirmatory_analysis_contract_sha256 is None
            or run_manifest.static_selection_evidence_sha256 is None
        ):
            fail("scientific analysis requires a schema-v2 confirmatory run manifest")
        if (
            run_manifest.provider_identity.provider_type != "llama_cpp"
            or run_manifest.provider_identity.model_sha256 is None
        ):
            fail("scientific analysis requires a digest-bound llama_cpp provider")
        try:
            require_llama_cpp_provider_binding(
                run_manifest.provider_identity,
                run_manifest.provider_effective_configuration_json,
            )
        except (TypeError, ValueError) as exc:
            fail(
                "scientific analysis requires internally consistent digest-bound "
                f"llama_cpp provider evidence: {exc}"
            )
            return
        stored_turns = self.list_turns()
        stored_inputs = self.list_turn_inputs()
        turn_by_id = {turn.condition_id: turn for turn in stored_turns}
        input_by_id = {evidence.condition_id: evidence for evidence in stored_inputs}
        if input_by_id.keys() != turn_by_id.keys():
            fail("scientific analysis requires exact prompt-side input evidence coverage")
        if (
            run_manifest.turn_input_evidence_sha256 is None
            or canonical_sha256(tuple(sorted(stored_inputs, key=lambda item: item.condition_id)))
            != run_manifest.turn_input_evidence_sha256
        ):
            fail("scientific prompt-side inputs do not match the frozen run identity")
        prompt_family_by_sequence: dict[str, str] = {}
        for condition_id, turn in turn_by_id.items():
            evidence = input_by_id[condition_id]
            prompt_family = evidence.prompt_family
            sequence_id = turn.condition.prompt_sequence_id
            previous_family = prompt_family_by_sequence.setdefault(sequence_id, prompt_family)
            if previous_family != prompt_family:
                fail("scientific prompt family is inconsistent within a prompt sequence")
            response = turn.response
            metrics = turn.metrics
            if response is None or metrics is None:
                fail("scientific input reconstruction requires committed response metrics")
                return
            reconstructed_metrics = compute_response_metrics(
                MetricContext(
                    prompt_case_id=evidence.prompt_case_id,
                    prompt_family=evidence.prompt_family,
                    prompt=turn.request.prompt,
                    response_text=response.text,
                    validator=evidence.validator,
                )
            )
            if metrics != reconstructed_metrics:
                fail("scientific response metrics do not reconstruct from committed inputs")
        if dict(sorted(prompt_family_by_sequence.items())) != dict(
            manifest.prompt_family_by_sequence
        ):
            fail("scientific prompt-family mapping does not reconstruct from committed inputs")
        if result.coverage.expected_count != len(
            stored_turns
        ) or result.coverage.observed_count != len(stored_turns):
            fail("confirmatory coverage does not reconstruct from committed turn evidence")
        try:
            if (
                run_manifest.evaluation_spec_json is None
                or run_manifest.evaluation_spec_sha256 is None
            ):
                raise ValueError("run manifest lacks its frozen evaluation spec")
            evaluation_spec = EvaluationSpec.model_validate_json(run_manifest.evaluation_spec_json)
            stored_records = _stored_scientific_records(
                stored_turns,
                dataset_sha256=run_manifest.dataset_hash,
                action_bounds=run_manifest.action_bounds,
            )
            source_coverage, source_guardrails = _source_scientific_guardrails(
                stored_records,
                run_manifest=run_manifest,
                dataset_seal_sha256=manifest.dataset_seal_sha256,
                evaluation_spec=evaluation_spec,
            )
            expected_unit_metrics = _stored_scientific_unit_metrics(stored_records)
        except (TypeError, ValueError) as exc:
            fail(f"scientific unit evidence cannot be reconstructed: {exc}")
            return
        if result.coverage != source_coverage:
            fail("confirmatory coverage does not match committed source evidence")
        expected_guardrails_by_key = {
            guardrail.evidence_key: guardrail for guardrail in source_guardrails
        }
        actual_guardrails_by_key = {
            guardrail.evidence_key: guardrail for guardrail in result.guardrails
        }
        if actual_guardrails_by_key != expected_guardrails_by_key:
            fail("scientific guardrails do not reconstruct from committed source evidence")
        actual_unit_metrics = {
            (
                outcome.unit_key.prompt_sequence_id,
                outcome.unit_key.model_seed,
                outcome.policy_id,
            ): (
                outcome.guardrail_clean_task_score.raw_task_score,
                outcome.instruction_adherence,
                outcome.repetition_ratio,
                outcome.response_length_tokens,
            )
            for outcome in result.unit_outcomes
        }
        if actual_unit_metrics != expected_unit_metrics:
            fail("scientific unit outcomes do not reconstruct from committed turn evidence")
        from neurallm.experiments.scientific_analysis import (
            _optional_metric_availability_from_turns,
            _recovery_evidence,
        )

        expected_optional_availability = _optional_metric_availability_from_turns(
            stored_turns,
            result.confirmatory_analysis_spec,
        )
        if dict(result.optional_metric_availability) != expected_optional_availability:
            fail("optional metric availability does not reconstruct from committed turns")

        try:
            _, expected_recovery_units, _, _ = _recovery_evidence(
                stored_records,
                result.confirmatory_analysis_spec,
            )
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"recovery unit evidence cannot be reconstructed: {exc}")
            return
        if result.recovery_unit_outcomes != expected_recovery_units:
            fail("recovery unit outcomes do not reconstruct from committed turn evidence")
        attribution_records = tuple(
            record
            for record in stored_records
            if record.turn_index > 0
            and record.policy_id in {"neural_persistent", "neural_matched_history_state_reset"}
        )
        attribution_outcomes = aggregate_matched_units(attribution_records)
        focal_attribution_keys = {
            (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed)
            for outcome in attribution_outcomes
            if outcome.policy_id == "neural_persistent"
        }
        reset_attribution_keys = {
            (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed)
            for outcome in attribution_outcomes
            if outcome.policy_id == "neural_matched_history_state_reset"
        }
        if focal_attribution_keys != reset_attribution_keys:
            fail("attribution unit evidence lacks exact persistent/reset matched-unit keys")
        attribution_by_key = {
            (
                outcome.policy_id,
                outcome.unit_key.prompt_sequence_id,
                outcome.unit_key.model_seed,
            ): outcome
            for outcome in attribution_outcomes
        }
        try:
            expected_attribution_units = tuple(
                (
                    outcome.unit_key.prompt_sequence_id,
                    outcome.unit_key.model_seed,
                    outcome.task_score
                    - attribution_by_key[
                        (
                            "neural_matched_history_state_reset",
                            outcome.unit_key.prompt_sequence_id,
                            outcome.unit_key.model_seed,
                        )
                    ].task_score,
                )
                for outcome in attribution_outcomes
                if outcome.policy_id == "neural_persistent"
            )
        except KeyError as exc:
            fail(f"attribution unit evidence lacks an exact reset match: {exc}")
            return
        actual_attribution_units = tuple(
            (
                outcome.unit_key.prompt_sequence_id,
                outcome.unit_key.model_seed,
                outcome.persistent_minus_reset_task_score,
            )
            for outcome in result.attribution_unit_outcomes
        )
        if actual_attribution_units != expected_attribution_units:
            fail("attribution unit outcomes do not reconstruct from committed turn evidence")
        if manifest.run_manifest_sha256 != canonical_sha256(run_manifest):
            fail("scientific analysis manifest does not match the run manifest")
        if manifest.run_finalization_sha256 != canonical_sha256(run_finalization):
            fail("scientific analysis manifest does not match the run finalization")
        if manifest.scientific_result_sha256 != run_finalization.scientific_result_sha256:
            fail("scientific analysis does not match the finalized scientific result")
        try:
            recomputed_scientific_result_sha256 = scientific_result_sha256(self.list_turns())
        except ValueError as exc:
            fail(f"scientific analysis cannot reconstruct the committed result: {exc}")
            return
        if recomputed_scientific_result_sha256 != run_finalization.scientific_result_sha256:
            fail("scientific analysis finalization does not match the committed result")
        if (
            run_manifest.scientific_identity_sha256 is None
            or manifest.scientific_identity_sha256 != run_manifest.scientific_identity_sha256
        ):
            fail("scientific analysis does not match the frozen plan scientific identity")
        if (
            run_manifest.preregistration_sha256 is None
            or manifest.preregistration_sha256 != run_manifest.preregistration_sha256
        ):
            fail("scientific analysis does not match the preregistration identity")
        if (
            run_manifest.static_selection_evidence_sha256 is None
            or manifest.static_selection_evidence_sha256
            != run_manifest.static_selection_evidence_sha256
        ):
            fail("scientific analysis does not match the static-selection evidence")
        if manifest.dataset_sha256 != run_manifest.dataset_hash:
            fail("scientific analysis does not match the run dataset")
        if manifest.evaluation_input_sha256 != result.input_sha256:
            fail("scientific analysis does not match the confirmatory evaluator input")
        if (
            result.confirmatory_analysis_spec != manifest.confirmatory_analysis_spec
            or result.confirmatory_analysis_spec_sha256
            != manifest.confirmatory_analysis_spec_sha256
            or result.prompt_family_by_sequence != manifest.prompt_family_by_sequence
            or result.prompt_family_design_sha256 != manifest.prompt_family_design_sha256
        ):
            fail("scientific result does not match the preregistered analysis design")
        if (
            not result.claim_eligible
            or not result.causal_mechanism_validated
            or not manifest.claim_eligible
            or not manifest.causal_mechanism_validated
            or result.run_manifest_sha256 != manifest.run_manifest_sha256
            or result.run_finalization_sha256 != manifest.run_finalization_sha256
            or result.analysis_contract_sha256 != manifest.confirmatory_analysis_contract_sha256
        ):
            fail("scientific result is not bound to claim-eligible closed-run evidence")
        if context is not None:
            from neurallm.experiments.scientific_analysis import ConfirmatoryAnalysisContext

            if not isinstance(context, ConfirmatoryAnalysisContext):
                fail("scientific analysis context has the wrong type")
            elif (
                not context.claim_eligible
                or not context.causal_mechanism_validated
                or context.analysis_contract_sha256
                != manifest.confirmatory_analysis_contract_sha256
                or context.evaluation_input_sha256 != result.input_sha256
                or context.run_manifest_sha256 != manifest.run_manifest_sha256
                or context.run_finalization_sha256 != manifest.run_finalization_sha256
            ):
                fail("scientific analysis context is not the exact claim-eligible run binding")
        from neurallm.experiments.scientific_analysis import (
            confirmatory_analysis_contract_sha256,
        )

        assert run_manifest.evaluation_spec_sha256 is not None
        assert run_manifest.turn_input_evidence_sha256 is not None
        assert run_manifest.static_selection_evidence_sha256 is not None
        try:
            expected_contract_sha256 = confirmatory_analysis_contract_sha256(
                scientific_identity_sha256=manifest.scientific_identity_sha256,
                preregistration_sha256=manifest.preregistration_sha256,
                static_selection_evidence_sha256=(manifest.static_selection_evidence_sha256),
                confirmatory_analysis_spec=manifest.confirmatory_analysis_spec,
                confirmatory_analysis_spec_sha256=(manifest.confirmatory_analysis_spec_sha256),
                evaluation_spec=evaluation_spec,
                evaluation_spec_sha256=run_manifest.evaluation_spec_sha256,
                turn_input_evidence_sha256=run_manifest.turn_input_evidence_sha256,
                prompt_family_by_sequence=manifest.prompt_family_by_sequence,
                prompt_family_design_sha256=manifest.prompt_family_design_sha256,
                dataset_sha256=manifest.dataset_sha256,
                dataset_purpose=manifest.dataset_purpose,
                dataset_seal_sha256=manifest.dataset_seal_sha256,
            )
        except ValueError as exc:
            fail(f"scientific analysis contract evidence is internally inconsistent: {exc}")
            return
        if (
            manifest.confirmatory_analysis_contract_sha256 != expected_contract_sha256
            or run_manifest.confirmatory_analysis_contract_sha256 != expected_contract_sha256
        ):
            fail(
                "scientific analysis evidence does not match the pre-execution "
                "confirmatory analysis contract"
            )

    def _bind_turn_input(self, evidence: TurnInputEvidence) -> None:
        """Insert one immutable input record inside the caller's transaction."""

        if not self._connection.in_transaction:
            raise StoreInvariantError("turn input evidence must be bound transactionally")
        evidence_json = canonical_json(evidence)
        evidence_sha256 = canonical_sha256(evidence)
        row = self._connection.execute(
            "SELECT input_json, input_sha256 FROM turn_inputs WHERE condition_id = ?",
            (evidence.condition_id,),
        ).fetchone()
        if row is None:
            finalized = self._connection.execute(
                "SELECT 1 FROM run_finalization WHERE singleton_id = 1"
            ).fetchone()
            if finalized is not None:
                raise StoreInvariantError("cannot bind turn input after run finalization")
            self._connection.execute(
                """
                INSERT INTO turn_inputs(condition_id, input_json, input_sha256)
                VALUES (?, ?, ?)
                """,
                (evidence.condition_id, evidence_json, evidence_sha256),
            )
            return
        if (
            self._required_text(row, "input_json") != evidence_json
            or self._required_text(row, "input_sha256") != evidence_sha256
        ):
            raise StoreInvariantError("turn is already bound to different input evidence")

    def _validate_finalization_against_store(
        self,
        finalization: RunFinalization,
        *,
        stored: bool,
    ) -> None:
        manifest = self._require_manifest()
        rows = self._connection.execute(
            "SELECT condition_id, state FROM turns ORDER BY condition_id"
        ).fetchall()
        actual_condition_ids = tuple(self._required_text(row, "condition_id") for row in rows)

        def fail(message: str) -> None:
            if stored:
                raise StoreCorruptionError(message)
            raise StoreInvariantError(message)

        if finalization.manifest_sha256 != canonical_sha256(manifest):
            fail("run finalization manifest hash does not match the bound manifest")
        if finalization.expected_condition_count != len(rows):
            fail("run finalization condition count does not match stored turns")
        if finalization.expected_condition_ids != actual_condition_ids:
            fail("run finalization condition IDs do not exactly match stored turns")
        if any(self._required_text(row, "state") != TurnState.COMMITTED.value for row in rows):
            fail("run finalization requires every expected turn to be committed")
        if finalization.execution_accounting is not None:
            response_count = int(
                self._connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            )
            accounting = finalization.execution_accounting
            if response_count != accounting.successful_responses:
                fail("run finalization accounting disagrees with persisted responses")
            if accounting.dispatched_logical_generations != len(rows):
                fail("run finalization accounting disagrees with dispatched turns")

    def _validate_history_binding(
        self,
        condition: ExperimentCondition,
        history: HistoryBinding | None,
    ) -> None:
        if condition.turn_index == 0:
            if history is not None:
                raise HistoryMismatchError("turn zero cannot bind previous-response history")
            return
        if history is None:
            raise HistoryMismatchError("nonzero turn requires exact committed predecessor history")
        previous = self._read_turn_or_none(history.previous_condition_id)
        if previous is None or previous.state is not TurnState.COMMITTED:
            raise HistoryMismatchError("history predecessor is absent or not committed")
        if previous.history_commitment_sha256 != history.previous_history_commitment_sha256:
            raise HistoryMismatchError("history commitment hash does not match the predecessor")
        manifest = self._require_manifest()
        expected_source_policy_id = manifest.matched_history_policy_sources.get(
            condition.policy_id,
            condition.policy_id,
        )
        if previous.condition.policy_id != expected_source_policy_id:
            raise HistoryMismatchError(
                "history predecessor belongs to different matched conditions or an "
                "undeclared source policy"
            )
        current_axes = (
            condition.experiment_id,
            condition.dataset_version,
            condition.prompt_sequence_id,
            condition.model_seed,
            condition.controller_seed,
            condition.provider_identity_id,
            condition.base_decoding_profile_id,
        )
        previous_axes = (
            previous.condition.experiment_id,
            previous.condition.dataset_version,
            previous.condition.prompt_sequence_id,
            previous.condition.model_seed,
            previous.condition.controller_seed,
            previous.condition.provider_identity_id,
            previous.condition.base_decoding_profile_id,
        )
        if current_axes != previous_axes:
            raise HistoryMismatchError(
                "history predecessor belongs to different matched conditions"
            )
        if previous.condition.turn_index != condition.turn_index - 1:
            raise HistoryMismatchError("history predecessor is not the immediately previous turn")

    def _read_turn(self, condition_id: str) -> StoredTurn:
        turn = self._read_turn_or_none(condition_id)
        if turn is None:
            raise StoreInvariantError(f"unknown condition_id: {condition_id}")
        return turn

    def _read_turn_or_none(self, condition_id: str) -> StoredTurn | None:
        if not isinstance(condition_id, str):
            raise TypeError("condition_id must be a string")
        row = self._connection.execute(
            """
            SELECT
                t.*,
                r.response_json,
                r.response_sha256,
                m.metrics_json,
                m.metrics_sha256,
                h.policy_state_json,
                h.policy_state_sha256,
                h.policy_trace_json,
                h.policy_trace_sha256,
                h.history_commitment_sha256
            FROM turns AS t
            LEFT JOIN responses AS r ON r.condition_id = t.condition_id
            LEFT JOIN turn_metrics AS m ON m.condition_id = t.condition_id
            LEFT JOIN history_commitments AS h ON h.condition_id = t.condition_id
            WHERE t.condition_id = ?
            """,
            (condition_id,),
        ).fetchone()
        if row is None:
            return None
        stored_condition_id = self._required_text(row, "condition_id")
        request_sha256 = self._required_text(row, "request_sha256")
        condition = self._decode_model(
            ExperimentCondition,
            self._required_text(row, "condition_json"),
            stored_condition_id,
            "experiment condition",
        )
        request = self._decode_model(
            GenerationRequest,
            self._required_text(row, "request_json"),
            request_sha256,
            "generation request",
        )
        if request.condition != condition or request.condition_id != stored_condition_id:
            raise StoreCorruptionError("stored request and condition identities disagree")
        self._validate_condition_columns(row, condition)
        try:
            state = TurnState(self._required_text(row, "state"))
        except ValueError as exc:
            raise StoreCorruptionError("turn contains an unknown checkpoint state") from exc
        previous_condition_id = self._nullable_text(row, "previous_condition_id")
        previous_commitment = self._nullable_text(row, "previous_history_commitment_sha256")
        if (previous_condition_id is None) != (previous_commitment is None):
            raise StoreCorruptionError("stored history binding is only partially populated")
        history = (
            HistoryBinding(
                previous_condition_id=previous_condition_id,
                previous_history_commitment_sha256=previous_commitment,
            )
            if previous_condition_id is not None and previous_commitment is not None
            else None
        )
        self._validate_stored_history_binding(condition, history)
        uncertain_reason = self._nullable_text(row, "uncertain_reason")
        if (state is TurnState.UNCERTAIN_DISPATCH) != (uncertain_reason is not None):
            raise StoreCorruptionError("uncertain dispatch reason does not match turn state")

        response_json = self._nullable_text(row, "response_json")
        response_sha256 = self._nullable_text(row, "response_sha256")
        metrics_json = self._nullable_text(row, "metrics_json")
        metrics_sha256 = self._nullable_text(row, "metrics_sha256")
        policy_state_json = self._nullable_text(row, "policy_state_json")
        policy_state_sha256 = self._nullable_text(row, "policy_state_sha256")
        policy_trace_json = self._nullable_text(row, "policy_trace_json")
        policy_trace_sha256 = self._nullable_text(row, "policy_trace_sha256")
        history_commitment_sha256 = self._nullable_text(row, "history_commitment_sha256")
        self._validate_checkpoint_presence(
            state,
            response_json,
            response_sha256,
            metrics_json,
            metrics_sha256,
            policy_state_json,
            policy_state_sha256,
            policy_trace_json,
            policy_trace_sha256,
            history_commitment_sha256,
        )
        response = (
            self._decode_model(
                GenerationResponse,
                response_json,
                response_sha256,
                "generation response",
            )
            if response_json is not None and response_sha256 is not None
            else None
        )
        if response is not None:
            self._validate_response_against_request(response, request)
        metrics = (
            self._decode_model(
                ResponseMetrics,
                metrics_json,
                metrics_sha256,
                "response metrics",
            )
            if metrics_json is not None and metrics_sha256 is not None
            else None
        )
        if policy_state_json is not None and policy_state_sha256 is not None:
            self._decode_json_object(
                policy_state_json,
                policy_state_sha256,
                "policy state",
            )
        if policy_trace_json is not None and policy_trace_sha256 is not None:
            trace_data = self._decode_json_object(
                policy_trace_json,
                policy_trace_sha256,
                "policy trace",
            )
            if trace_data.get("policy_id") != condition.policy_id:
                raise StoreCorruptionError("stored policy trace has the wrong policy_id")
            if trace_data.get("turn_index") != condition.turn_index:
                raise StoreCorruptionError("stored policy trace has the wrong turn_index")
        if history_commitment_sha256 is not None:
            expected_commitment = canonical_sha256(
                {
                    "condition_id": stored_condition_id,
                    "request_sha256": request_sha256,
                    "response_sha256": response_sha256,
                    "metrics_sha256": metrics_sha256,
                    "policy_state_sha256": policy_state_sha256,
                    "policy_trace_sha256": policy_trace_sha256,
                    "previous_history_commitment_sha256": previous_commitment,
                }
            )
            if history_commitment_sha256 != expected_commitment:
                raise StoreCorruptionError("stored history commitment hash does not verify")
        return StoredTurn(
            condition_id=stored_condition_id,
            request_sha256=request_sha256,
            state=state,
            condition=condition,
            request=request,
            history=history,
            response=response,
            metrics=metrics,
            policy_state_json=policy_state_json,
            policy_trace_json=policy_trace_json,
            history_commitment_sha256=history_commitment_sha256,
            uncertain_reason=uncertain_reason,
        )

    @staticmethod
    def _validate_response_against_request(
        response: GenerationResponse,
        request: GenerationRequest,
    ) -> None:
        if response.provider_identity.identity_id != request.provider_identity_id:
            raise StoreInvariantError("response provider identity does not match the request")
        if not effective_parameters_match_request(
            response.effective_parameters,
            request.decoding_parameters,
        ):
            raise StoreInvariantError("response effective parameters do not match the request")
        if response.raw_metadata.request_sha256 != canonical_sha256(request):
            raise StoreInvariantError("response metadata does not bind the canonical request")
        if (
            response.provider_identity.provider_type == "llama_cpp"
            or response.raw_metadata.generation_method == "llama_cpp_completion_http_v1"
        ):
            try:
                require_llama_cpp_generation_binding(request, response)
            except (TypeError, ValueError) as exc:
                raise StoreInvariantError(
                    "llama_cpp wire evidence does not bind the stored request and response"
                ) from exc

    def _validate_stored_history_binding(
        self,
        condition: ExperimentCondition,
        history: HistoryBinding | None,
    ) -> None:
        if condition.turn_index == 0:
            if history is not None:
                raise StoreCorruptionError("stored turn-zero history must be null")
            return
        if history is None:
            raise StoreCorruptionError("stored nonzero turn lacks a history binding")
        index_row = self._connection.execute(
            "SELECT turn_index FROM turns WHERE condition_id = ?",
            (history.previous_condition_id,),
        ).fetchone()
        if index_row is None or index_row[0] != condition.turn_index - 1:
            raise StoreCorruptionError("stored history is not the immediately previous turn")
        previous = self._read_turn(history.previous_condition_id)
        if previous.state is not TurnState.COMMITTED:
            raise StoreCorruptionError("stored history predecessor is not committed")
        if previous.history_commitment_sha256 != history.previous_history_commitment_sha256:
            raise StoreCorruptionError("stored predecessor history commitment is mismatched")
        manifest = self._require_manifest()
        expected_source_policy_id = manifest.matched_history_policy_sources.get(
            condition.policy_id,
            condition.policy_id,
        )
        if previous.condition.policy_id != expected_source_policy_id:
            raise StoreCorruptionError(
                "stored history crosses unmatched condition axes or an undeclared source policy"
            )
        current_axes = (
            condition.experiment_id,
            condition.dataset_version,
            condition.prompt_sequence_id,
            condition.model_seed,
            condition.controller_seed,
            condition.provider_identity_id,
            condition.base_decoding_profile_id,
        )
        previous_axes = (
            previous.condition.experiment_id,
            previous.condition.dataset_version,
            previous.condition.prompt_sequence_id,
            previous.condition.model_seed,
            previous.condition.controller_seed,
            previous.condition.provider_identity_id,
            previous.condition.base_decoding_profile_id,
        )
        if current_axes != previous_axes:
            raise StoreCorruptionError("stored history crosses unmatched condition axes")

    @staticmethod
    def _validate_condition_columns(
        row: sqlite3.Row,
        condition: ExperimentCondition,
    ) -> None:
        expected: dict[str, str | int] = {
            "experiment_id": condition.experiment_id,
            "dataset_version": condition.dataset_version,
            "prompt_sequence_id": condition.prompt_sequence_id,
            "turn_index": condition.turn_index,
            "policy_id": condition.policy_id,
            "model_seed": condition.model_seed,
            "controller_seed": condition.controller_seed,
            "provider_identity_id": condition.provider_identity_id,
            "base_decoding_profile_id": condition.base_decoding_profile_id,
        }
        for name, value in expected.items():
            if row[name] != value:
                raise StoreCorruptionError(f"stored condition column {name} disagrees with JSON")

    @staticmethod
    def _validate_checkpoint_presence(
        state: TurnState,
        response_json: str | None,
        response_sha256: str | None,
        metrics_json: str | None,
        metrics_sha256: str | None,
        policy_state_json: str | None,
        policy_state_sha256: str | None,
        policy_trace_json: str | None,
        policy_trace_sha256: str | None,
        history_commitment_sha256: str | None,
    ) -> None:
        response_complete = response_json is not None and response_sha256 is not None
        metrics_complete = metrics_json is not None and metrics_sha256 is not None
        commitment_values = (
            policy_state_json,
            policy_state_sha256,
            policy_trace_json,
            policy_trace_sha256,
            history_commitment_sha256,
        )
        commitment_complete = all(value is not None for value in commitment_values)
        if (response_json is None) != (response_sha256 is None):
            raise StoreCorruptionError("response checkpoint is partially populated")
        if (metrics_json is None) != (metrics_sha256 is None):
            raise StoreCorruptionError("metrics checkpoint is partially populated")
        if any(value is not None for value in commitment_values) and not commitment_complete:
            raise StoreCorruptionError("history commitment is partially populated")
        expected = {
            TurnState.PREPARED: (False, False, False),
            TurnState.DISPATCHING: (False, False, False),
            TurnState.UNCERTAIN_DISPATCH: (False, False, False),
            TurnState.RESPONSE_PERSISTED: (True, False, False),
            TurnState.METRICS_COMPUTED: (True, True, False),
            TurnState.COMMITTED: (True, True, True),
        }[state]
        if (response_complete, metrics_complete, commitment_complete) != expected:
            raise StoreCorruptionError("durable evidence does not match the checkpoint state")

    def _set_uncertain(self, condition_id: str, reason: str) -> None:
        self._connection.execute(
            """
            UPDATE turns
            SET state = ?, uncertain_reason = ?
            WHERE condition_id = ?
            """,
            (TurnState.UNCERTAIN_DISPATCH.value, reason, condition_id),
        )

    @staticmethod
    def _decode_model(
        model_type: type[ModelT],
        value_json: str,
        expected_sha256: str,
        label: str,
    ) -> ModelT:
        SQLiteRunStore._decode_canonical_json(value_json, expected_sha256, label)
        try:
            return model_type.model_validate_json(value_json)
        except ValidationError as exc:
            raise StoreCorruptionError(f"stored {label} fails typed validation") from exc

    @staticmethod
    def _decode_json_object(
        value_json: str,
        expected_sha256: str,
        label: str,
    ) -> dict[str, Any]:
        value = SQLiteRunStore._decode_canonical_json(
            value_json,
            expected_sha256,
            label,
        )
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise StoreCorruptionError(f"stored {label} is not a JSON object")
        return cast(dict[str, Any], value)

    @staticmethod
    def _decode_canonical_json(
        value_json: str,
        expected_sha256: str,
        label: str,
    ) -> Any:
        try:
            value: Any = json.loads(value_json)
            if canonical_json(value) != value_json:
                raise StoreCorruptionError(f"stored {label} is not canonical JSON")
            if canonical_sha256(value) != expected_sha256:
                raise StoreCorruptionError(f"stored {label} digest does not verify")
            return value
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if isinstance(exc, StoreCorruptionError):
                raise
            raise StoreCorruptionError(f"stored {label} is invalid JSON") from exc

    @staticmethod
    def _required_text(row: sqlite3.Row, name: str) -> str:
        value = row[name]
        if not isinstance(value, str):
            raise StoreCorruptionError(f"database field {name} is not text")
        return value

    @staticmethod
    def _nullable_text(row: sqlite3.Row, name: str) -> str | None:
        value = row[name]
        if value is not None and not isinstance(value, str):
            raise StoreCorruptionError(f"database field {name} is not nullable text")
        return value

    def _ensure_open(self) -> None:
        if self._closed:
            raise StoreInvariantError("SQLite run store is closed")


__all__ = ["SQLiteRunStore"]

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
from neurallm.domain.models import ExperimentCondition, ResponseMetrics, RunManifest
from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.contract import phase3_analysis_contract_sha256
from neurallm.evaluation.models import (
    GuardrailResult,
    PairwiseComparisonResult,
    Phase3EvaluationResult,
)
from neurallm.providers.base import (
    GenerationRequest,
    GenerationResponse,
    effective_parameters_match_request,
)
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
    HistoryBinding,
    ResumeAction,
    RunFinalization,
    StoredAnalysis,
    StoredTurn,
    TurnInputEvidence,
    TurnState,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
PolicyStateT = TypeVar("PolicyStateT", bound=PolicyState)

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
    ) -> RunFinalization:
        """Atomically close an exact, fully committed run schedule, idempotently."""

        if not isinstance(expected_condition_ids, tuple) or not all(
            isinstance(condition_id, str) for condition_id in expected_condition_ids
        ):
            raise TypeError("expected_condition_ids must be a tuple of strings")
        if not isinstance(scientific_result_sha256, str):
            raise TypeError("scientific_result_sha256 must be a string")
        canonical_condition_ids = tuple(sorted(expected_condition_ids))
        with self._transaction():
            manifest = self._require_manifest()
            finalization = RunFinalization(
                expected_condition_ids=canonical_condition_ids,
                expected_condition_count=len(canonical_condition_ids),
                manifest_sha256=canonical_sha256(manifest),
                scientific_result_sha256=scientific_result_sha256,
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

"""Canonical transactional storage and crash-safe resumption."""

from neurallm.storage.errors import (
    DuplicateLogicalRequestError,
    HistoryMismatchError,
    ManifestMismatchError,
    SchemaVersionError,
    StateTransitionError,
    StorageError,
    StoreCorruptionError,
    StoreInvariantError,
    UncertainDispatchError,
)
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION
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
from neurallm.storage.sqlite import SQLiteRunStore

__all__ = [
    "AnalysisFinalization",
    "AnalysisManifest",
    "CURRENT_SCHEMA_VERSION",
    "CommittedHistory",
    "DurableExecutionAccounting",
    "DuplicateLogicalRequestError",
    "HistoryBinding",
    "HistoryMismatchError",
    "ManifestMismatchError",
    "ResumeAction",
    "RunFinalization",
    "ScientificAnalysisFinalization",
    "ScientificAnalysisManifest",
    "SQLiteRunStore",
    "SchemaVersionError",
    "StateTransitionError",
    "StorageError",
    "StoreCorruptionError",
    "StoreInvariantError",
    "StoredTurn",
    "StoredAnalysis",
    "StoredScientificAnalysis",
    "TurnInputEvidence",
    "TurnState",
    "UncertainDispatchError",
    "scientific_result_sha256",
]

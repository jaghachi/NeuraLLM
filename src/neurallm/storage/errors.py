"""Fail-closed errors raised by the canonical SQLite run store."""


class StorageError(RuntimeError):
    """Base class for storage and resumption failures."""


class SchemaVersionError(StorageError):
    """Raised when a database schema cannot be safely opened or migrated."""


class StoreCorruptionError(StorageError):
    """Raised when persisted data fails its structural or hash checks."""


class StoreInvariantError(StorageError):
    """Raised when an operation would violate the run-store contract."""


class ManifestMismatchError(StoreInvariantError):
    """Raised when a store is rebound to a different run manifest."""


class DuplicateLogicalRequestError(StoreInvariantError):
    """Raised when one condition is associated with conflicting request data."""


class StateTransitionError(StoreInvariantError):
    """Raised when a turn operation is invalid for its current state."""


class HistoryMismatchError(StoreInvariantError):
    """Raised when prerequisite committed history is absent or mismatched."""


class UncertainDispatchError(StoreInvariantError):
    """Raised when a dispatched request has no safely reusable response."""


__all__ = [
    "DuplicateLogicalRequestError",
    "HistoryMismatchError",
    "ManifestMismatchError",
    "SchemaVersionError",
    "StateTransitionError",
    "StorageError",
    "StoreCorruptionError",
    "StoreInvariantError",
    "UncertainDispatchError",
]

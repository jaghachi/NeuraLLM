"""Ordered, append-only SQLite schema migrations for canonical run stores."""

from dataclasses import dataclass

APPLICATION_ID = 0x4E4C4C4D  # ASCII "NLLM"
CURRENT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class Migration:
    """One atomic forward schema migration."""

    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS = (
    Migration(
        version=1,
        name="canonical_transactional_turn_store",
        statements=(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            ) STRICT
            """,
            """
            CREATE TABLE run_manifest (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(manifest_sha256) = 64
                        AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                    )
            ) STRICT
            """,
            """
            CREATE TABLE run_finalization (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                finalization_json TEXT NOT NULL,
                finalization_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(finalization_sha256) = 64
                        AND finalization_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (singleton_id) REFERENCES run_manifest(singleton_id)
            ) STRICT
            """,
            """
            CREATE TABLE turns (
                condition_id TEXT PRIMARY KEY
                    CHECK (
                        length(condition_id) = 64
                        AND condition_id NOT GLOB '*[^0-9a-f]*'
                    ),
                request_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(request_sha256) = 64
                        AND request_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                condition_json TEXT NOT NULL,
                request_json TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                prompt_sequence_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
                policy_id TEXT NOT NULL,
                model_seed INTEGER NOT NULL,
                controller_seed INTEGER NOT NULL,
                provider_identity_id TEXT NOT NULL
                    CHECK (
                        length(provider_identity_id) = 64
                        AND provider_identity_id NOT GLOB '*[^0-9a-f]*'
                    ),
                base_decoding_profile_id TEXT NOT NULL,
                previous_condition_id TEXT,
                previous_history_commitment_sha256 TEXT,
                state TEXT NOT NULL CHECK (
                    state IN (
                        'PREPARED',
                        'DISPATCHING',
                        'RESPONSE_PERSISTED',
                        'METRICS_COMPUTED',
                        'COMMITTED',
                        'UNCERTAIN_DISPATCH'
                    )
                ),
                uncertain_reason TEXT,
                UNIQUE (
                    experiment_id,
                    dataset_version,
                    prompt_sequence_id,
                    turn_index,
                    policy_id,
                    model_seed,
                    controller_seed,
                    provider_identity_id,
                    base_decoding_profile_id
                ),
                FOREIGN KEY (previous_condition_id)
                    REFERENCES history_commitments(condition_id),
                CHECK (
                    (turn_index = 0
                        AND previous_condition_id IS NULL
                        AND previous_history_commitment_sha256 IS NULL)
                    OR
                    (turn_index > 0
                        AND previous_condition_id IS NOT NULL
                        AND previous_history_commitment_sha256 IS NOT NULL)
                ),
                CHECK (
                    (state = 'UNCERTAIN_DISPATCH' AND uncertain_reason IS NOT NULL)
                    OR
                    (state <> 'UNCERTAIN_DISPATCH' AND uncertain_reason IS NULL)
                )
            ) STRICT
            """,
            """
            CREATE TABLE responses (
                condition_id TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                response_sha256 TEXT NOT NULL
                    CHECK (
                        length(response_sha256) = 64
                        AND response_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (condition_id) REFERENCES turns(condition_id)
            ) STRICT
            """,
            """
            CREATE TABLE turn_metrics (
                condition_id TEXT PRIMARY KEY,
                metrics_json TEXT NOT NULL,
                metrics_sha256 TEXT NOT NULL
                    CHECK (
                        length(metrics_sha256) = 64
                        AND metrics_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (condition_id) REFERENCES turns(condition_id)
            ) STRICT
            """,
            """
            CREATE TABLE history_commitments (
                condition_id TEXT PRIMARY KEY,
                policy_state_json TEXT NOT NULL,
                policy_state_sha256 TEXT NOT NULL
                    CHECK (
                        length(policy_state_sha256) = 64
                        AND policy_state_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                policy_trace_json TEXT NOT NULL,
                policy_trace_sha256 TEXT NOT NULL
                    CHECK (
                        length(policy_trace_sha256) = 64
                        AND policy_trace_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                history_commitment_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(history_commitment_sha256) = 64
                        AND history_commitment_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (condition_id) REFERENCES turns(condition_id)
            ) STRICT
            """,
            """
            CREATE INDEX turns_state_index ON turns(state)
            """,
            """
            CREATE INDEX turns_schedule_index ON turns(
                prompt_sequence_id,
                model_seed,
                policy_id,
                turn_index
            )
            """,
            """
            CREATE TRIGGER turns_forward_state_guard
            BEFORE UPDATE OF state ON turns
            WHEN NOT (
                (OLD.state = 'PREPARED' AND NEW.state = 'DISPATCHING')
                OR
                (OLD.state = 'DISPATCHING' AND NEW.state = 'RESPONSE_PERSISTED')
                OR
                (OLD.state = 'DISPATCHING' AND NEW.state = 'UNCERTAIN_DISPATCH')
                OR
                (OLD.state = 'RESPONSE_PERSISTED' AND NEW.state = 'METRICS_COMPUTED')
                OR
                (OLD.state = 'METRICS_COMPUTED' AND NEW.state = 'COMMITTED')
            )
            BEGIN
                SELECT RAISE(ABORT, 'invalid turn state transition');
            END
            """,
            """
            CREATE TRIGGER turns_insert_after_finalization_guard
            BEFORE INSERT ON turns
            WHEN EXISTS (
                SELECT 1 FROM run_finalization WHERE singleton_id = 1
            )
            BEGIN
                SELECT RAISE(ABORT, 'cannot insert turn after run finalization');
            END
            """,
        ),
    ),
    Migration(
        version=2,
        name="phase3_analysis_evidence",
        statements=(
            """
            CREATE TABLE turn_inputs (
                condition_id TEXT PRIMARY KEY
                    CHECK (
                        length(condition_id) = 64
                        AND condition_id NOT GLOB '*[^0-9a-f]*'
                    ),
                input_json TEXT NOT NULL,
                input_sha256 TEXT NOT NULL
                    CHECK (
                        length(input_sha256) = 64
                        AND input_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (condition_id) REFERENCES turns(condition_id)
            ) STRICT
            """,
            """
            CREATE TABLE analysis_manifest (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(manifest_sha256) = 64
                        AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (singleton_id) REFERENCES run_manifest(singleton_id)
            ) STRICT
            """,
            """
            CREATE TABLE comparison_results (
                comparison_id TEXT PRIMARY KEY
                    CHECK (
                        length(comparison_id) = 64
                        AND comparison_id NOT GLOB '*[^0-9a-f]*'
                    ),
                analysis_singleton_id INTEGER NOT NULL DEFAULT 1
                    CHECK (analysis_singleton_id = 1),
                result_json TEXT NOT NULL,
                result_sha256 TEXT NOT NULL
                    CHECK (
                        length(result_sha256) = 64
                        AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (analysis_singleton_id)
                    REFERENCES analysis_manifest(singleton_id)
            ) STRICT
            """,
            """
            CREATE TABLE guardrail_results (
                guardrail_id TEXT PRIMARY KEY
                    CHECK (
                        length(guardrail_id) = 64
                        AND guardrail_id NOT GLOB '*[^0-9a-f]*'
                    ),
                analysis_singleton_id INTEGER NOT NULL DEFAULT 1
                    CHECK (analysis_singleton_id = 1),
                result_json TEXT NOT NULL,
                result_sha256 TEXT NOT NULL
                    CHECK (
                        length(result_sha256) = 64
                        AND result_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (analysis_singleton_id)
                    REFERENCES analysis_manifest(singleton_id)
            ) STRICT
            """,
            """
            CREATE TABLE analysis_decision (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                decision_json TEXT NOT NULL,
                decision_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(decision_sha256) = 64
                        AND decision_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (singleton_id) REFERENCES analysis_manifest(singleton_id)
            ) STRICT
            """,
            """
            CREATE TABLE analysis_finalization (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                finalization_json TEXT NOT NULL,
                finalization_sha256 TEXT NOT NULL UNIQUE
                    CHECK (
                        length(finalization_sha256) = 64
                        AND finalization_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                FOREIGN KEY (singleton_id) REFERENCES analysis_decision(singleton_id)
            ) STRICT
            """,
            """
            CREATE TRIGGER turn_inputs_insert_after_run_finalization_guard
            BEFORE INSERT ON turn_inputs
            WHEN EXISTS (
                SELECT 1 FROM run_finalization WHERE singleton_id = 1
            )
            BEGIN
                SELECT RAISE(ABORT, 'cannot insert turn input after run finalization');
            END
            """,
            """
            CREATE TRIGGER comparisons_insert_after_analysis_finalization_guard
            BEFORE INSERT ON comparison_results
            WHEN EXISTS (
                SELECT 1 FROM analysis_finalization WHERE singleton_id = 1
            )
            BEGIN
                SELECT RAISE(ABORT, 'cannot insert comparison after analysis finalization');
            END
            """,
            """
            CREATE TRIGGER guardrails_insert_after_analysis_finalization_guard
            BEFORE INSERT ON guardrail_results
            WHEN EXISTS (
                SELECT 1 FROM analysis_finalization WHERE singleton_id = 1
            )
            BEGIN
                SELECT RAISE(ABORT, 'cannot insert guardrail after analysis finalization');
            END
            """,
        ),
    ),
)


if tuple(migration.version for migration in MIGRATIONS) != tuple(
    range(1, CURRENT_SCHEMA_VERSION + 1)
):
    raise RuntimeError("storage migrations must be contiguous and append-only")


__all__ = ["APPLICATION_ID", "CURRENT_SCHEMA_VERSION", "MIGRATIONS", "Migration"]

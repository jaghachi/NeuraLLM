"""Durable exact-schedule closure tests for canonical run stores."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurallm.domain.serialization import canonical_sha256
from neurallm.providers.fake import FakeProvider
from neurallm.reporting import scientific_result_sha256
from neurallm.storage import (
    DurableExecutionAccounting,
    RunFinalization,
    SQLiteRunStore,
    StoreInvariantError,
    TurnState,
)
from tests.storage.helpers import complete_request, make_manifest, make_request


@pytest.mark.parametrize(
    ("condition_ids", "condition_count", "message"),
    (
        ((), 0, "must not be empty"),
        (("b" * 64, "a" * 64), 2, "sorted and unique"),
        (("a" * 64,), 2, "must equal"),
    ),
)
def test_finalization_model_rejects_invalid_schedule_identity(
    condition_ids: tuple[str, ...],
    condition_count: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        RunFinalization(
            expected_condition_ids=condition_ids,
            expected_condition_count=condition_count,
            manifest_sha256="c" * 64,
            scientific_result_sha256="d" * 64,
        )


def test_durable_execution_accounting_rejects_incoherent_counts() -> None:
    with pytest.raises(ValidationError, match="successful plus uncertain"):
        DurableExecutionAccounting(
            planned_logical_generations=20,
            dispatched_logical_generations=20,
            successful_responses=19,
            uncertain_dispatches=0,
            committed_logical_generations=19,
        )

    with pytest.raises(ValidationError, match="cannot exceed successful"):
        DurableExecutionAccounting(
            planned_logical_generations=20,
            dispatched_logical_generations=20,
            successful_responses=19,
            uncertain_dispatches=1,
            committed_logical_generations=20,
        )


def test_finalization_requires_exact_committed_schedule_and_allows_identical_replay(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    manifest = make_manifest(provider.provider_identity)
    first = make_request(provider.provider_identity, prompt_sequence_id="sequence-a")
    second = make_request(provider.provider_identity, prompt_sequence_id="sequence-b")

    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        with pytest.raises(TypeError, match="tuple of strings"):
            store.finalize_run([], "f" * 64)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a string"):
            store.finalize_run((), 1)  # type: ignore[arg-type]

        complete_request(store, provider, first)
        store.prepare_turn(second)

        with pytest.raises(StoreInvariantError, match="every expected turn"):
            store.finalize_run(
                (first.condition_id, second.condition_id),
                "f" * 64,
            )

        complete_request(store, provider, second)
        with pytest.raises(StoreInvariantError, match="condition count"):
            store.finalize_run((first.condition_id,), "f" * 64)

        result_sha256 = scientific_result_sha256(store.list_turns())
        expected_ids = tuple(sorted((first.condition_id, second.condition_id)))
        finalization = store.finalize_run(expected_ids, result_sha256)

        assert finalization == store.finalize_run(expected_ids, result_sha256)
        assert finalization == store.get_finalization()
        assert finalization.expected_condition_ids == expected_ids
        assert finalization.expected_condition_count == 2
        assert finalization.manifest_sha256 == canonical_sha256(manifest)
        assert finalization.scientific_result_sha256 == result_sha256
        assert store.prepare_turn(first).state is TurnState.COMMITTED
        with pytest.raises(StoreInvariantError, match="different closure evidence"):
            store.finalize_run(expected_ids, "f" * 64)

        new_request = make_request(
            provider.provider_identity,
            prompt_sequence_id="sequence-c",
        )
        with pytest.raises(StoreInvariantError, match="after run finalization"):
            store.prepare_turn(new_request)

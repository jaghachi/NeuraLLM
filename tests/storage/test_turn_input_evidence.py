"""Canonical prompt-side evidence required for offline metric reconstruction."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neurallm.domain.models import PromptFeatures
from neurallm.metrics import ValidatorSpec
from neurallm.providers.fake import FakeProvider
from neurallm.storage import (
    SQLiteRunStore,
    StoreCorruptionError,
    StoreInvariantError,
    TurnInputEvidence,
)
from tests.storage.helpers import make_manifest, make_request


def make_input(condition_id: str, *, prompt_case_id: str = "case-a") -> TurnInputEvidence:
    return TurnInputEvidence(
        condition_id=condition_id,
        prompt_case_id=prompt_case_id,
        prompt_family="constrained",
        prompt_features=PromptFeatures({"constraint_count": 1.0}),
        validator=ValidatorSpec(kind="non_empty"),
    )


def test_turn_input_is_immutable_hash_validated_and_idempotent(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    evidence = make_input(request.condition_id)

    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        store.prepare_turn(request, input_evidence=evidence)
        store.prepare_turn(request, input_evidence=evidence)

        assert store.get_turn_input(request.condition_id) == evidence
        assert store.list_turn_inputs() == (evidence,)
        with pytest.raises(StoreInvariantError, match="different input evidence"):
            store.prepare_turn(
                request,
                input_evidence=make_input(request.condition_id, prompt_case_id="other-case"),
            )


def test_turn_input_must_target_the_request_condition(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)

    with SQLiteRunStore(
        tmp_path / "run.sqlite3",
        make_manifest(provider.provider_identity),
    ) as store:
        with pytest.raises(ValueError, match="another condition"):
            store.prepare_turn(request, input_evidence=make_input("f" * 64))


def test_turn_input_corruption_fails_closed(tmp_path: Path) -> None:
    provider = FakeProvider()
    request = make_request(provider.provider_identity)
    database = tmp_path / "run.sqlite3"
    with SQLiteRunStore(database, make_manifest(provider.provider_identity)) as store:
        store.prepare_turn(request, input_evidence=make_input(request.condition_id))

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE turn_inputs SET input_json = ? WHERE condition_id = ?",
            ('{"corrupted":true}', request.condition_id),
        )

    with SQLiteRunStore(database) as store:
        with pytest.raises(StoreCorruptionError, match="turn input evidence"):
            store.verify_integrity()

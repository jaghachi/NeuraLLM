"""Manifest-authorized matched focal history storage contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurallm.domain.models import RunManifest
from neurallm.domain.serialization import canonical_sha256
from neurallm.providers.fake import FakeProvider
from neurallm.storage import HistoryMismatchError, SQLiteRunStore
from tests.storage.helpers import complete_request, make_manifest, make_request

_PERSISTENT = "neural_persistent"
_RESET = "neural_matched_history_state_reset"


def _phase4_manifest() -> RunManifest:
    provider = FakeProvider()
    payload = make_manifest(provider.provider_identity).model_dump(mode="python")
    payload["policy_config_hashes"] = {
        _PERSISTENT: canonical_sha256(_PERSISTENT),
        _RESET: canonical_sha256(_RESET),
    }
    payload["matched_history_policy_sources"] = {_RESET: _PERSISTENT}
    payload["decision_rule_version"] = "phase4-neural-mechanism-only-v1"
    return RunManifest.model_validate(payload)


def test_store_accepts_only_the_manifest_declared_focal_cross_policy_edge(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    manifest = _phase4_manifest()
    persistent_zero = make_request(
        provider.provider_identity,
        turn_index=0,
        policy_id=_PERSISTENT,
    )
    reset_zero = make_request(
        provider.provider_identity,
        turn_index=0,
        policy_id=_RESET,
    )
    reset_one = make_request(
        provider.provider_identity,
        turn_index=1,
        policy_id=_RESET,
    )

    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        complete_request(store, provider, persistent_zero)
        complete_request(store, provider, reset_zero)

        with pytest.raises(HistoryMismatchError, match="undeclared source policy"):
            store.prepare_turn(
                reset_one,
                store.history_binding_for(reset_zero.condition_id),
            )

        focal_binding = store.history_binding_for(persistent_zero.condition_id)
        prepared = store.prepare_turn(reset_one, focal_binding)
        assert prepared.history == focal_binding


@pytest.mark.parametrize(
    "current_kwargs",
    (
        {"prompt_sequence_id": "another-sequence"},
        {"model_seed": 99},
        {"controller_seed": 101},
    ),
)
def test_declared_cross_policy_history_still_requires_every_matched_axis(
    current_kwargs: dict[str, object],
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    manifest = _phase4_manifest()
    persistent_zero = make_request(
        provider.provider_identity,
        turn_index=0,
        policy_id=_PERSISTENT,
    )
    reset_one = make_request(
        provider.provider_identity,
        turn_index=1,
        policy_id=_RESET,
        **current_kwargs,  # type: ignore[arg-type]
    )

    with SQLiteRunStore(tmp_path / "run.sqlite3", manifest) as store:
        complete_request(store, provider, persistent_zero)
        with pytest.raises(HistoryMismatchError, match="different matched conditions"):
            store.prepare_turn(
                reset_one,
                store.history_binding_for(persistent_zero.condition_id),
            )

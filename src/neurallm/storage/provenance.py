"""Canonical aggregate hashes over committed run evidence."""

from __future__ import annotations

import json

from neurallm.domain.serialization import canonical_sha256
from neurallm.storage.models import StoredTurn, TurnState


def scientific_result_sha256(turns: tuple[StoredTurn, ...]) -> str:
    """Hash canonical committed results while excluding run location and source state."""

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


__all__ = ["scientific_result_sha256"]

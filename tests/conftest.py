"""Default-test safety fixtures."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import NoReturn

import pytest


@pytest.fixture(autouse=True)
def _forbid_network(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fail network access except for an explicitly selected live-marked test."""

    if request.node.get_closest_marker("live") is not None:
        yield
        return

    def blocked(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("network access is forbidden in the default test suite")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    yield

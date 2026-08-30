"""Default-test safety fixtures."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import NoReturn

import pytest


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any default test that attempts to open a network connection."""

    def blocked(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("network access is forbidden in the default test suite")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    yield

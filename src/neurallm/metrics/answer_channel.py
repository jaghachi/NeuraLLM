"""Strict, versioned separation of reasoning markup from the final answer."""

from __future__ import annotations

import re

_REASONING_MARKER = re.compile(r"<\s*/?\s*think\b", flags=re.IGNORECASE)


def final_answer_text(response_text: str) -> str | None:
    """Return the v2 answer channel, or None for invalid reasoning framing.

    Plain responses are unchanged. The only accepted reasoning envelope is one
    exact, leading ``<think>...</think>`` block (optional leading whitespace).
    Whitespace immediately after its closing delimiter is framing, not answer
    content; trailing answer whitespace is preserved. Nested, additional,
    misplaced, incomplete, or noncanonical reasoning delimiters fail closed.
    This is a reserved-marker grammar, not a heuristic detector of reasoning.
    """

    if not isinstance(response_text, str):
        raise TypeError("response_text must be a string")
    markers = tuple(_REASONING_MARKER.finditer(response_text))
    if not markers:
        return response_text
    framed = response_text.lstrip()
    if not framed.startswith("<think>") or len(markers) != 2:
        return None
    close = framed.find("</think>", len("<think>"))
    if close < 0:
        return None
    # The two marker starts must belong to this exact opening/closing pair;
    # additional or malformed markers cannot be repaired into an envelope.
    return framed[close + len("</think>") :].lstrip()


__all__ = ["final_answer_text"]

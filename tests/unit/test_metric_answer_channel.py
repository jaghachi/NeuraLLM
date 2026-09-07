"""The final-answer grammar fails closed without repairing answer content."""

import pytest

from neurallm.metrics.answer_channel import final_answer_text


@pytest.mark.parametrize(
    ("raw", "answer"),
    (
        ("plain answer", "plain answer"),
        ("  plain answer\n", "  plain answer\n"),
        ("", ""),
        ("<think>private content</think>answer", "answer"),
        ("\n<think>private content</think>\n\nanswer\n", "answer\n"),
        ("<think></think>answer", "answer"),
        ("<think>private content</think>", ""),
    ),
)
def test_answer_channel_uses_only_exact_leading_reasoning_envelope(raw: str, answer: str) -> None:
    assert final_answer_text(raw) == answer


@pytest.mark.parametrize(
    "raw",
    (
        "<think>unfinished keywords",
        "<think",
        "</think>answer",
        "answer <think>reason</think>",
        "<think>outer <think>inner</think></think>answer",
        "<think>one</think><think>two</think>answer",
        "<think>reason</think>answer </think>",
        "<THINK>reason</THINK>answer",
        "<think >reason</think>answer",
        "< think>reason</think>answer",
        "<think>reason</think >answer",
        "<think>reason</ think>answer",
        "<think>reason</THINK>answer",
        "<think>reason</think>answer <think",
    ),
)
def test_malformed_or_misplaced_reasoning_has_no_answer(raw: str) -> None:
    assert final_answer_text(raw) is None


def test_answer_channel_requires_text() -> None:
    with pytest.raises(TypeError, match="string"):
        final_answer_text(None)  # type: ignore[arg-type]

"""Audited deterministic rendering for the single-user, no-thinking protocol.

This is intentionally not a general Jinja interpreter. The allowlisted Qwen3.5
template's text-only user branch and generation suffix are fully specified here.
Other templates require a separately reviewed protocol before live use.
"""

QWEN35_NO_THINKING_TEMPLATE_SHA256 = (
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
)


def render_no_thinking_prompt(prompt: str, template_sha256: str) -> str:
    """Return the exact supported rendering, without network access or inference."""

    if template_sha256 != QWEN35_NO_THINKING_TEMPLATE_SHA256:
        raise ValueError("chat_template_no_thinking_v1 requires the audited Qwen3.5 template")
    content = prompt.strip()
    if not content:
        raise ValueError("chat_template_no_thinking_v1 requires a non-blank user prompt")
    if content.startswith("<tool_response>") and content.endswith("</tool_response>"):
        raise ValueError("chat_template_no_thinking_v1 requires a user query, not a tool response")
    return (
        "<|im_start|>user\n"
        + content
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

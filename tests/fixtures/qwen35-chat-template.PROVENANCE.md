# Qwen3.5 template fixture

`qwen35-chat-template.jinja` is the template exposed by llama.cpp build b10217
(`ddd4ec142`) for `lmstudio-community/Qwen3.5-4B-GGUF:Q4_K_M`, captured in the
original NeuraLLM engineering-smoke manifest. Model artifact SHA-256:
`25082a7dd3776cc3c741c6347d3bd04523f05796607b3fbc32fa3a25dfa1418c`.

The exact server string has no trailing newline. The text fixture adds one
file-terminating LF only; tests remove that single LF before hashing. The
remaining source SHA-256 is
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
No model weights or generated reasoning are included in this fixture.

Original model/template: Qwen, [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
GGUF distribution: [LM Studio Community model card](https://huggingface.co/lmstudio-community/Qwen3.5-4B-GGUF/blob/main/README.md),
which identifies the Apache-2.0 license. This third-party fixture remains under
Apache-2.0, reproduced in `LICENSE.qwen35.txt`; the project author's proprietary
license does not relicense the third-party material.

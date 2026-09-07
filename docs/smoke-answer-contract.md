# Phase 5 smoke answer-contract correction

## Observed evidence

The original `runs/model-backed-live-smoke` run at source commit
`e592b5384a2ffbb79ead48c9dc2e070e6372dda7` retained 20 planned, dispatched,
successful, and committed logical generations with zero uncertain dispatches.
Its manifest identity is
`fa896fc8288fabd847d6e15a1c89b33c0081e57ad9780abb91b98d830260ec78`;
scientific-result identity is
`1855c92b371e7fe87961144e37676a66c657d16375594006aa4e1d7f63608d53`.
These are canonical identities, not hashes of mutable SQLite container bytes.

A read-only immutable-SQLite audit found:

| Observation | Count |
| --- | ---: |
| `stop_type=limit`, `tokens_predicted=128` | 20/20 |
| Server `truncated=false` | 20/20 |
| Unclosed `<think>` block | 15/20 |
| Keyword-validator score 1.0, with all keywords inside unfinished reasoning | 10/10 |
| Exact-match score 0.0 | 5/5 |
| JSON-object score 0.0 | 5/5 |
| CSV response/score matches canonical store | 20/20 |

The v1 provider sends raw prompts to `/completion`. Although it binds the
server template hash, it does not apply the template. The lexical validator-v1
algorithm behaves as documented; the defect is the measurement/input-channel
contract, not a mismatch between its code and definition. These data establish
engineering execution, not usable final answers, efficacy, or attribution.

## Versioned correction

`chat_template_no_thinking_v1` selects provider implementation
`llama-cpp-chat-template-http-v2` and metadata method
`llama_cpp_chat_template_http_v2`. It supports exactly the audited Qwen3.5
template with SHA-256
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
One user message and explicit Boolean `enable_thinking=false` are rendered
through `/apply-template`. Construction probes compatibility without inference;
each request then checks the exact expected rendering before one completion.
There is no automatic model/provider fallback or retry.

This constraint matters because `/apply-template` returns the rendered prompt
but not all chat-parser metadata, and server renderer options are not fully
exposed by `/props`. It is not generally equivalent to a chat-completions
endpoint. See the pinned [endpoint implementation](https://github.com/ggml-org/llama.cpp/blob/ddd4ec142/tools/server/server-context.cpp#L4863-L4873)
and [chat parsing](https://github.com/ggml-org/llama.cpp/blob/ddd4ec142/tools/server/server-common.cpp#L1035-L1124).

The retained request envelope binds the original prompt, exact template source,
template request/response, and rendered completion request. Reconstruction
independently checks their relationships. Raw completion content is never
stripped from `GenerationResponse.text`. V2 also retains and validates
`stop_type`, `tokens_predicted`, and the distinct context-truncation flag.
An observed `limit` outcome stays a successful accounted response, without
claiming EOS or a usable answer. [Server response-field definitions](https://github.com/ggml-org/llama.cpp/blob/ddd4ec142/tools/server/README.md#L595-L608).

The separately versioned [final-answer metrics](metrics.md) exclude marked
reasoning from task scores and controller feedback. All candidates must use
one identical declared version set. Stored/exported pilot scores must be
reconstructed from the retained raw response, not trusted as supplied numbers.

## Remaining gate

This correction is offline implementation, not new live evidence. Preserve all
original local configs and run artifacts. New example templates choose v2,
but remain intentionally non-executable until fresh preflight evidence is bound.
Do not launch the prepared v1 pilot grid or reuse it as v2 selection evidence.

The next live tier is a fresh 20-generation engineering smoke with new
experiment/run IDs, unchanged fixed smoke budget, and the v2 provider/metric
contracts. Inspect actual answer suitability before new development pilots.
Later selection and confirmatory seals must bind the new identities; no
scientific decision is established by this correction or by the original smoke.

Independent offline reviews identified and required two additional checks:
construction-time renderer compatibility, and score reconstruction when loading
external pilot selection evidence. Both belong to the correction, not to a
claim that the live answer-quality gate has passed.

## Offline verification on 2026-09-07

Commands ran in the `neurallm2` environment from the repository root:

```powershell
conda run -n neurallm2 ruff check .
conda run -n neurallm2 ruff format --check .
conda run -n neurallm2 mypy src
# A fresh GUID-named directory under .tmp supplied the two test-output paths.
conda run --no-capture-output -n neurallm2 python -m pytest -q -p no:cacheprovider --basetemp <fresh-test-directory> --junitxml <fresh-report-path>
```

Lint passed, formatting passed for 147 files, and mypy passed for 71 source
files. The final complete default suite passed **798 tests**, with **1 live
test deselected**, in 231.38 seconds. A preceding full coverage run measured
90.26% and exposed one stale dummy-version guard-test fixture; only that test
fixture was repaired before the successful full rerun. The runtime was not
weakened to accept dummy versions. Windows' old shared pytest temp/cache
directories had ACL errors, so fresh test-only paths and disabled caching were
used; the product and assertions were unchanged for that host workaround.

The five-arm mock-HTTP integration executes exactly 20 completion fixtures,
retains limit outcomes, rejects unfinished-reasoning scores, and replays without
HTTP requests. This is explicitly not another live smoke.

A separate current-source audit opened the original SQLite database directly
with `mode=ro&immutable=1`, never through a writable store. All 20 request/response
bindings and all 220 metric values reconstructed exactly under v1. Canonical
manifest/result identities and history commitments matched, SQLite integrity
and foreign-key checks passed, and before/after SHA-256/name/size inventories
were identical for all six original files, without sidecars.

# Phase 2 metric definitions

## Claim boundary

Phase 2 computes deterministic response-level metrics and objective validators
for the experiment kernel. These values exercise storage, reconstruction, and
reporting. They do not implement the Phase 3 paired evaluator, calculate the
confirmatory `guardrail_clean_task_score`, compare policies, or support an
efficacy claim.

Every metric is stored as a strict provenance-bearing record:

```text
value | null
availability
metric_version
input_hash
```

Availability is true exactly when a value is present. Optional missing values
are not imputed. `input_hash` is a lowercase SHA-256 over the metric name and
version plus the prompt case ID, prompt family, prompt, response text, and
validator specification.

## Tokenization

The Phase 2 repetition and diversity metrics use
`unicode-nfkc-casefold-whitespace-v1`:

1. Normalize the response with Unicode NFKC.
2. Apply Unicode `casefold()`.
3. Split on Unicode whitespace.

This is an intentionally small deterministic text tokenizer, not the evaluated
model's tokenizer. Empty or whitespace-only responses produce zero tokens.

## Implemented metric tuple

For a token sequence of length `N`, let `U` be the number of unique tokens. For
an n-gram order `n`, let `G_n = max(N - n + 1, 0)` and `U_n` be the number of
unique observed n-grams.

| Metric | Phase 2 definition | Empty or too-short input |
| --- | --- | --- |
| `task_score` | Normalized result from the prompt case's deterministic validator | Validator-defined |
| `instruction_adherence` | The same deterministic validator score in Phase 2 | Validator-defined |
| `response_length_tokens` | `N` after Phase 2 tokenization | `0` |
| `repetition_ratio` | `(N - U) / N` | `0.0` when `N = 0` |
| `repeated_3_gram_ratio` | `(G_3 - U_3) / G_3` | `0.0` when `G_3 = 0` |
| `repeated_4_gram_ratio` | `(G_4 - U_4) / G_4` | `0.0` when `G_4 = 0` |
| `distinct_2` | `U_2 / G_2` | `0.0` when `G_2 = 0` |
| `distinct_3` | `U_3 / G_3` | `0.0` when `G_3 = 0` |
| `late_window_repetition_ratio` | Repeated tokens in the final quarter divided by final-quarter length | `0.0` when `N < 2` |
| `format_validity` | Validator-specific structural validity | Validator-defined |
| `semantic_similarity` | Explicitly unavailable in Phase 2 | `value = null`, `availability = false` |

The late window has `max(1, floor(N / 4))` tokens. Its initial seen set contains
all earlier tokens. A final-window token counts as repeated when it has already
appeared either before the window or earlier within the window.

## Objective validators

Dataset cases select exactly one strict validator:

| Kind | Configuration | Score and validity |
| --- | --- | --- |
| `non_empty` | No objective fields | Stripped non-empty response gives `task_score = instruction_adherence = format_validity = 1.0`; otherwise all are `0.0`. |
| `contains_all` | Non-empty unique `required_terms`; optional `case_sensitive` | Score is the fraction of required substrings present. Matching casefolds both sides unless case-sensitive. Format validity is `1.0` for a stripped non-empty response. |
| `exact_match` | Non-blank `expected_text`; optional `case_sensitive` | Score is `1.0` only when the full strings match under the selected case rule. No whitespace repair occurs. Format validity records stripped non-emptiness. |
| `json_object` | Non-empty unique `required_json_keys` | The response must parse as a JSON object. Duplicate object keys and non-finite JSON numbers are rejected. Score is the fraction of required keys present; format validity is `1.0` for any valid JSON object. |

For all four validator kinds, Phase 2 assigns `instruction_adherence` the same
normalized value as `task_score`. This is a deterministic kernel contract, not
an assertion that the two constructs are scientifically interchangeable.

## Frozen versions

The experiment configuration must match the implementation's complete metric
version map exactly:

| Metric | Version |
| --- | --- |
| `task_score` | `validator-v1` |
| `instruction_adherence` | `validator-v1` |
| `response_length_tokens` | `unicode-nfkc-casefold-whitespace-v1` |
| `repetition_ratio` | `token-repetition-unicode-nfkc-casefold-whitespace-v1` |
| `repeated_3_gram_ratio` | `repeated-3gram-unicode-nfkc-casefold-whitespace-v1` |
| `repeated_4_gram_ratio` | `repeated-4gram-unicode-nfkc-casefold-whitespace-v1` |
| `distinct_2` | `distinct-2gram-unicode-nfkc-casefold-whitespace-v1` |
| `distinct_3` | `distinct-3gram-unicode-nfkc-casefold-whitespace-v1` |
| `late_window_repetition_ratio` | `late-quarter-unicode-nfkc-casefold-whitespace-v1` |
| `format_validity` | `validator-v1` |
| `semantic_similarity` | `semantic-unavailable-v1` |

A version mismatch fails planning instead of silently mixing definitions.

## Canonical and derived records

The complete metric records, including availability, versions, and input hashes,
remain in `run.sqlite3`. `results.csv` is a derived convenience view containing
metric values and a dedicated semantic-availability column. The CSV does not
replace the canonical database and must not be used to infer missing provenance.

Phase 2 `comparisons.csv` contains only its header, and `decision.json` records a
null scientific decision with `engineering_validation_only` claim scope. Policy
comparisons, guardrail aggregation, confidence intervals, and final decision
logic begin in later phases.

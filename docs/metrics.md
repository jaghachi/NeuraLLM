# Metric and evaluator definitions

## Claim boundary

The current implementation computes deterministic response-level metrics and
uses a strict subset of them in the Phase 3 paired baseline evaluator. Phase 3
adds exact coverage, nested aggregation, deterministic paired statistics,
guardrails, and a limited verdict skeleton. The checked-in Phase 3 evaluator
fixtures use the fake provider and contain no neural policy; the separate Phase
4 causal harness does not enter this evaluator.

Those Phase 3 definitions do not establish live-provider validity, neural
efficacy, model-backed efficacy, or persistent-state attribution. A Phase 3
verdict has claim scope `phase-3-statistical-behavior-only`;
`scientific_decision` remains `null`. The separate Phase 5 evaluator and final
decision contracts are implemented, but checked-in code and offline fixtures do
not constitute an observed live or confirmatory result.

## Model-backed metric and claim boundary

Phase 5 uses exactly five arms. The first four in the declared reporting order
(`best_static`, `random_matched`, `heuristic_adaptive`, and
`neural_persistent`) form the independent efficacy population.
`neural_matched_history_state_reset` is the fifth arm and contributes only to
persistent-state attribution; its dependent matched-history responses must not
enter end-to-end efficacy estimates.

Metric coverage must account for the full frozen schedule before analysis:

| Tier | Exact schedule | Logical generations | Permitted interpretation |
| --- | --- | ---: | --- |
| Engineering smoke | 2 sequences x 2 turns x 1 model seed x 1 controller seed x 5 arms | 20 | Engineering validation only; no scientific claim |
| Development pilot | 6 sequences x 4 turns x 2 model seeds x 1 controller seed x 5 arms | 240 | Development calibration only; no scientific claim |
| Confirmatory | 24 sequences x 4 turns x 5 model seeds x 1 controller seed x 5 arms | 2,400 | One final decision only after all identity, coverage, provenance, and guardrail checks pass |

No model-backed metric evidence exists merely because a dataset or report
schema exists. Live evidence requires explicit no-generation provider
preflight, followed by both execution gates, `--execute` and
`--allow-live-provider`; default tests and fake-provider runs remain
zero-network evidence.

## Confirmatory endpoints and frozen analysis

The confirmatory primary endpoint is `guardrail_clean_task_score`. It preserves
the raw `task_score` for audit and exposes the same unmodified value for pairing
only when every declared gate passes. A failed or invalid gate makes the gated
value unavailable; the evaluator never substitutes zero, drops the unit, or
blends guardrail values into a weighted score.

End-to-end efficacy uses `neural_persistent` minus each of `best_static`,
`heuristic_adaptive`, and `random_matched` at the prompt-sequence by model-seed
unit. The first two are the exact Holm family; `random_matched` is reported as a
negative control. The frozen confirmatory defaults are 10,000 paired-bootstrap
resamples, 10,000 sign-flip resamples, 95% confidence, a `0.02` practical-effect
threshold, maximum adherence regression `0.01`, maximum length reduction
`0.05`, maximum action-saturation rate `0.05`, and exact matched coverage `1.0`.

Recovery is evaluated separately with exactly these three measures:

- `post_stressor_task_score_change`;
- `post_stressor_repetition_change`; and
- `time_to_return_to_target_band`.

For every recovery event and model seed, the first two endpoints use focal minus
comparator and time-to-return uses comparator minus focal, so every margin is
positive when it favors `neural_persistent`. Each endpoint retains the minimum
oriented margin across the exact serious comparator set (`best_static`,
`heuristic_adaptive`). That worst-case reduction produces one matched-unit value;
the two comparator margins are never averaged.

Failure to return inside the preregistered window is represented as
right-censored at window length plus one and counted explicitly. Persistent-
state attribution pairs `neural_persistent` with
`neural_matched_history_state_reset`, excludes turn zero from the effect, and
never enters the efficacy or Holm populations. Optional-metric missingness and
oppositely resolved `prompt_family` subgroup effects are recorded as typed
limitations with their preregistered disposition before the final decision.

### Phase 5 negative-side multiplicity

The stochastic gates that can establish `VALIDATED_NEGATIVE` form exactly one
seven-member family:

- efficacy against `best_static`, `heuristic_adaptive`, and `random_matched`;
- the three required recovery endpoints; and
- persistent-state attribution against
  `neural_matched_history_state_reset`.

Negative-side bootstraps use a Bonferroni simultaneous two-sided confidence of
`1 - 0.05 / 7 = 0.9928571428571429`, which controls familywise error at `0.05`.
Substantive deterministic guardrail failures, behavioral-alias findings, and
focal right-censor failures are direct gates and do not enter this stochastic
multiplicity family. Positive gates retain nominal 95% bootstrap intervals; the
positive efficacy hypotheses against the two serious comparators retain Holm
correction. The negative-side adjustment does not widen or replace either
positive rule.

Every confirmatory statistical result is loaded against the complete embedded
`ConfirmatoryAnalysisSpec`; seeds, resample counts, confidence levels,
permutation settings, and practical thresholds cannot drift independently of
that frozen spec. Prompt-family sensitivity bootstraps are retained as typed
subgroup results with their exact family assignment and unit count. The
confirmatory `statistics_call_count` is therefore derived from persisted
efficacy, recovery, attribution, and subgroup statistics instead of accepted as
an unverified counter.

The subgroup design itself is pre-execution evidence. Every prompt sequence must
map to exactly one `prompt_family`; the canonical sequence-to-family mapping and
its SHA-256 are repeated across the analysis contract, scientific manifest,
result, and `decision.json`. Unit outcomes must cover those sequence keys and
carry the same labels, so post hoc relabeling cannot manufacture or suppress a
subgroup.

Version 2 persists the raw efficacy, recovery, and attribution unit outcomes and
the exact optional-metric availability counts. The evaluator recomputes the
aggregates from those records and derives the complete limitation tuple from
availability, censoring, and subgroup-conflict evidence. This closes both raw
outcome substitution and limitation injection as decision inputs.

The Phase 5 versions are an inseparable, incompatible envelope: decision rule
`confirmatory-scientific-decision-v2`, analysis spec
`confirmatory-analysis-v2`, result `confirmatory-evaluation-v2`, storage
manifest `confirmatory-scientific-analysis-storage-v2`, and Phase 5
`decision.json` schema 2. The physical SQLite schema stays at 2, but legacy v1
scientific-analysis rows are rejected. Because the closed-run manifest is also
bound to the v1 rule, there is no supported in-place migration or offline
reanalyze path; current evidence requires a new v2 confirmatory workflow in a
fresh run directory.

The run manifest freezes the canonical `EvaluationSpec` JSON and SHA-256 plus
the hash of all condition-keyed `TurnInputEvidence`. Persistence requires exact
input coverage, recomputes deterministic metrics from each committed input and
response, and then recomputes guardrail status and thresholds from committed
action, history, and metric evidence under that spec before accepting the
result.

## Response-level provenance

Every response metric is stored as a strict provenance-bearing record:

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

The repetition and diversity metrics use
`unicode-nfkc-casefold-whitespace-v1`:

1. Normalize the response with Unicode NFKC.
2. Apply Unicode `casefold()`.
3. Split on Unicode whitespace.

This is an intentionally small deterministic text tokenizer, not the evaluated
model's tokenizer. Empty or whitespace-only responses produce zero tokens.

## Implemented response metric tuple

For a token sequence of length `N`, let `U` be the number of unique tokens.
For n-gram order `n`, let `G_n = max(N - n + 1, 0)` and `U_n` be the number
of unique observed n-grams.

| Metric | Definition | Empty or too-short input |
| --- | --- | --- |
| `task_score` | Normalized result from the prompt case's deterministic validator | Validator-defined |
| `instruction_adherence` | The same deterministic validator score in the current implementation | Validator-defined |
| `response_length_tokens` | `N` after deterministic tokenization | `0` |
| `repetition_ratio` | `(N - U) / N` | `0.0` when `N = 0` |
| `repeated_3_gram_ratio` | `(G_3 - U_3) / G_3` | `0.0` when `G_3 = 0` |
| `repeated_4_gram_ratio` | `(G_4 - U_4) / G_4` | `0.0` when `G_4 = 0` |
| `distinct_2` | `U_2 / G_2` | `0.0` when `G_2 = 0` |
| `distinct_3` | `U_3 / G_3` | `0.0` when `G_3 = 0` |
| `late_window_repetition_ratio` | Repeated tokens in the final quarter divided by final-quarter length | `0.0` when `N < 2` |
| `format_validity` | Validator-specific structural validity | Validator-defined |
| `semantic_similarity` | Explicitly unavailable | `value = null`, `availability = false` |

The late window has `max(1, floor(N / 4))` tokens. Its initial seen set contains
all earlier tokens. A final-window token counts as repeated when it has already
appeared either before the window or earlier within the window.

For all implemented validators, `instruction_adherence` currently equals
`task_score`. This is a deterministic kernel contract, not an assertion that
the two constructs are scientifically interchangeable.

## Objective validators

Dataset cases select exactly one strict validator:

| Kind | Configuration | Score and validity |
| --- | --- | --- |
| `non_empty` | No objective fields | Stripped non-empty response gives `task_score = instruction_adherence = format_validity = 1.0`; otherwise all are `0.0`. |
| `contains_all` | Non-empty unique `required_terms`; optional `case_sensitive` | Score is the fraction of required substrings present. Matching casefolds both sides unless case-sensitive. Format validity is `1.0` for a stripped non-empty response. |
| `exact_match` | Non-blank `expected_text`; optional `case_sensitive` | Score is `1.0` only when the full strings match under the selected case rule. No whitespace repair occurs. Format validity records stripped non-emptiness. |
| `json_object` | Non-empty unique `required_json_keys` | The response must parse as a JSON object. Duplicate object keys and non-finite JSON numbers are rejected. Score is the fraction of required keys present; format validity is `1.0` for any valid JSON object. |

## Frozen metric versions

An experiment configuration must match the complete implementation map exactly:

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

## Phase 3 evaluator inputs

One `TurnEvaluationRecord` binds the dataset hash, prompt sequence, turn,
policy, model seed, controller seed, provider identity, history-presence
semantics, required metrics, and action evidence. The evaluator requires these
metrics for every planned condition:

```text
task_score
instruction_adherence
response_length_tokens
repetition_ratio
```

It also records:

- normalized action magnitude: root-mean-square magnitude of the four
  step-clamped action components after each is divided by the maximum absolute
  value of its configured bound;
- action-bound compliance, checked against the raw declared action; and
- whether legal decoding application saturated any controlled dimension.

Missing required metrics are invalid evidence. They are not imputed.

## Exact coverage and statistical unit

The evaluator expands the frozen design into every expected key:

```text
prompt_sequence_id
turn_index
policy_id
model_seed
controller_seed
```

Observed keys must match this Cartesian grid exactly. Missing, unexpected, or
duplicate keys fail `matched_condition_coverage`. Dataset and provider identity,
turn-zero history semantics, action bounds, and required metric availability are
also validated before aggregation. If any integrity guardrail is invalid, the
result is `invalid`, `statistics_call_count` is zero, and no outcomes or
comparisons are produced.

The primary statistical unit is:

```text
prompt sequence x model seed
```

The frozen aggregation version is
`mean-controller-seed-then-turn-v1`:

1. For each policy, prompt sequence, model seed, and controller seed, average
   each required metric and action measure across turns.
2. Average those controller-seed means for the policy's prompt-sequence by
   model-seed unit.
3. Pair the focal and comparator unit values by the exact prompt-sequence and
   model-seed key.

Turns and controller seeds stay nested and never inflate `unit_count`.
Pairwise estimates are focal-minus-comparator mean differences in
`task_score`.

## Paired statistical methods

The evaluator records deterministic method versions, resample counts, and
seeds.

### Paired bootstrap

`paired-bootstrap-percentile-v1` resamples complete matched-unit differences
with replacement, computes each resampled mean, and returns the configured
percentile interval. The estimate is the arithmetic mean of the original paired
differences.

### Paired sign-flip test

`paired-sign-flip-exact-or-monte-carlo-v1` is a two-sided test on the absolute
mean paired difference:

- for at most 20 matched units, all `2^N` sign patterns are enumerated when
  that count does not exceed the configured resample budget;
- otherwise the method uses a deterministic seeded Monte Carlo sign stream and
  the add-one p-value correction `(extreme + 1) / (performed + 1)`.

### Multiplicity

`holm-v1` applies the Holm step-down adjustment to required serious
comparators. Negative controls are reported but excluded from that family.

The checked-in Phase 3 configurations freeze:

| Setting | Value |
| --- | --- |
| Bootstrap resamples | `10,000` |
| Confidence level | `0.95` |
| Sign-flip resamples | `10,000` |
| Practical-effect threshold | `0.02` |
| Equivalence margin | `0.005` |
| Maximum adherence regression | `0.01` |
| Maximum response-length reduction | `0.05` |
| Maximum focal action-saturation rate | `0.05` |
| Required matched coverage | `1.0` |
| Behavioral-alias tolerance | `0.0` |

The resampling seeds differ between the baseline-evaluation and synthetic
fixtures and are part of each `EvaluationSpec`.

## Guardrails

Phase 3 emits explicit machine-readable guardrails:

| Guardrail | Implemented rule |
| --- | --- |
| `matched_condition_coverage` | Observed keys exactly equal the frozen grid and every record carries the frozen dataset hash. Failure is invalid. |
| `provider_identity_stability` | Every record carries exactly the frozen provider identity. Failure is invalid. |
| `turn_zero_equivalence` | Every turn-zero record has `has_previous_response = false` and no previous history commitment. Failure is invalid. |
| `action_bound_compliance` | Every raw policy action is within the frozen action bounds. Failure is invalid. |
| `metric_availability` | Every required Phase 3 metric is present. Failure is invalid. |
| `action_saturation_rate` | The focal policy's turn-level saturation rate does not exceed the configured maximum. Failure is substantive. |
| `instruction_adherence_non_regression` | The matched focal-minus-comparator adherence mean is no worse than the configured negative margin. Failure is substantive. |
| `response_length_confound` | For matched units with improved repetition, the maximum within-unit response-shortening ratio is compared with the configured limit. This prevents lengthening elsewhere from canceling a shortening-based gain. |
| `behavioral_alias_detection` | All matched differences in task score, adherence, length, repetition, action magnitude, and action saturation are within the alias tolerance. Alias evidence supports `equivalent`, not superiority. |

The current turn-zero evaluator guardrail validates exact null/false history
semantics. Separate policy tests establish that all three implemented Phase 3
policies emit an exact zero action at turn zero.

## Phase 3 verdict rules

For one focal/comparator pair, substantive adherence, length-confound, or focal
saturation failure yields `inferior`. Otherwise:

- `equivalent` applies when behavior is aliased or the entire bootstrap
  interval lies within `[-equivalence_margin, +equivalence_margin]`;
- `superior` requires an estimate at least the practical threshold, a positive
  bootstrap lower bound, and the applicable raw or Holm-adjusted p-value no
  greater than `1 - confidence_level`;
- `inferior` applies symmetrically for a negative estimate and negative
  bootstrap upper bound; and
- all other valid comparisons are `inconclusive`.

The overall verdict uses serious comparators only: any serious `inferior`
yields `inferior`; all serious `superior` yields `superior`; all serious
`equivalent` yields `equivalent`; every other valid combination yields
`inconclusive`. An integrity failure yields `invalid` before this logic.

These are Phase 3 baseline-evaluator verdicts, not Phase 5 scientific decisions.

## Canonical and derived records

Complete response metrics and provenance remain in `run.sqlite3`.
`results.csv` is a convenience view containing metric values and a dedicated
semantic-availability column. It does not replace the canonical database.

Schema-v2 Phase 3 analysis stores the analysis manifest, comparisons,
guardrails, Phase 3 result, and analysis finalization in the same database with
canonical hashes and idempotent finalization. `comparisons.csv` exposes paired
estimates, bootstrap bounds, sign-flip results, Holm values where applicable,
alias and guardrail summaries, and pair verdicts. `decision.json` and
`report.md` expose the limited Phase 3 verdict and evidence identities.
`scientific_decision` remains null.

Historical Phase 2 behavior is unchanged: `comparisons.csv` is header-only and
`decision.json` uses `engineering_validation_only` scope with a null
scientific decision.

The final confirmatory report must keep `engineering validity`, `controller
activity`, `end-to-end efficacy`, `persistent-state attribution`, `guardrail
outcomes`, `limitations`, and `final decision` as distinct sections. The final
decision field accepts exactly `VALIDATED_POSITIVE`, `VALIDATED_NEGATIVE`,
`INCONCLUSIVE`, or `INVALID_RUN`. Smoke and pilot metrics can never populate
that field.

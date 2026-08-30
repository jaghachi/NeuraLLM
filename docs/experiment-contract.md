# NeuraLLM experiment contract

## Contract status

This document defines the scientific invariants that implementation and
preregistration must preserve. The current Phase 2 implementation realizes the
deterministic experiment kernel: strict development inputs, planning, a fixed
kernel policy, bounded action application, deterministic response metrics,
fake and llama.cpp provider boundaries, transactional SQLite execution/resume,
and compact derived artifacts. It does not claim live llama.cpp validity,
serious comparator evaluation, neural behavior, confirmatory statistics, or a
model-backed scientific result.

The experiment is valid even if the neural controller has no benefit. A run may
support only one final decision state, and engineering completion does not
depend on a positive result.

## Falsifiable claim

The confirmatory efficacy question is whether `neural_persistent` improves
output-based task performance over every required serious comparator under the
same model, prompts, seeds, base decoding profile, action limits, and fixed
generation budget.

The persistent-state attribution question is separate: given identical focal
history and matched current-turn conditions, does preserving neural state change
the current trajectory relative to resetting that state?

Evidence for one question must not be substituted for evidence for the other.
Controller movement establishes neither output efficacy nor persistent-state
attribution by itself.

## Analysis populations

### End-to-end efficacy

The efficacy population contains:

```text
best_static
random_matched
heuristic_adaptive
neural_persistent
```

Each policy operates independently:

- it produces its own response;
- it observes metrics only from its own previous response;
- it carries only its own explicitly declared state; and
- its next action cannot depend on another policy's output or state.

The required serious primary comparators are `best_static` and
`heuristic_adaptive`. `random_matched` is a negative-control and structured-action
sanity comparison. Static-profile selection uses development data only and is
frozen before the sealed evaluation dataset is opened.

### Persistent-state attribution

The attribution population pairs:

```text
neural_persistent
neural_matched_history_state_reset
```

For each attribution turn, the reset arm:

- receives the exact committed previous-turn `ResponseMetrics` tuple from the
  paired focal `neural_persistent` condition;
- preserves the real prompt-sequence turn index;
- preserves the focal history-presence semantics;
- resets neural substrate/controller state at the declared intervention
  boundary;
- uses the same current prompt, model seed, action decoder, provider, base
  decoding parameters, and action bounds; and
- never feeds its own response metrics into focal history.

`neural_matched_history_state_reset` is attribution-only. It must not enter the
independent efficacy population, be summarized as an operating baseline, or
replace any required efficacy comparator.

At turn zero the paired attribution arms must be byte-equivalent before the
state-reset intervention can have meaning. At later turns, focal-history hashes
must match and tests must establish that only declared persistent state differs.

## Experimental unit and condition identity

The primary statistical unit is:

```text
prompt sequence x model seed
```

Turns within one sequence are correlated observations and must not be treated as
independent samples.

Every `ExperimentCondition` uniquely binds:

```text
experiment_id
dataset_version
prompt_sequence_id
turn_index
policy_id
model_seed
controller_seed
provider_identity_id
base_decoding_profile_id
```

The logical request identity derives from this complete condition and the
validated request inputs. Execution order, timestamp, host path, process ID, and
retry count are not scientific identity fields. Missing, duplicate, or mismatched
conditions invalidate the affected confirmatory run rather than being silently
repaired after execution.

## Causal timing and history

The controller sees one strict pre-generation observation:

```text
turn_index
prompt_family
current_prompt_features
previous_response_metrics | null
has_previous_response
```

The observation excludes current- or future-response metrics, comparator
outcomes, final labels, response hashes as optimization targets, other-policy
state, judge output, and aggregate confirmatory-set results.

Turn zero always means no observed response history:

```text
previous_response_metrics = null
has_previous_response = false
```

Neutral metric values must not be injected and labeled history. A controller
may calculate documented internal defaults after receiving null, but those
values remain calculation defaults rather than measurements and must be marked
as such in its trace.

For turn `t > 0`, the observation may contain only metrics from a valid,
committed response allowed by the selected analysis design:

- the same policy's turn `t - 1` response for independent efficacy; or
- the paired focal persistent arm's exact committed turn `t - 1` metric tuple
  for matched-history attribution.

The current response can affect only a future action. Stale, uncommitted,
wrong-policy, wrong-turn, wrong-sequence, or hash-mismatched history fails closed.

## Generation budget and action surface

`DecodingParameters` contains:

```text
temperature
top_p
top_k
presence_penalty
max_tokens
seed
```

The prompt case or experiment plan fixes `max_tokens`. It is never part of
`ControllerAction`, and no policy may directly control `max_tokens` or
`num_predict`. This prevents shorter output from becoming a trivial route to an
apparent repetition improvement.

Every policy uses one action type containing only bounded deltas:

```text
temperature_delta
top_p_delta
top_k_delta
presence_penalty_delta
```

Initial pilot defaults for maximum movement per turn are:

```text
temperature       +/- 0.10
top_p              +/- 0.05
top_k              +/- 10
presence_penalty   +/- 0.20
```

These are development defaults, not timeless scientific constants. Any change
must occur during approved development or pilot work, be justified, and be
frozen before confirmatory execution. The system records the raw action,
step-clamped action, final legal parameters, and saturation indicators
separately.

## Deterministic planning and fake execution

The plan is a deterministic expansion of validated experiment configuration,
dataset version and hash, prompt sequences, policies, seeds, provider identity,
and base profile into a complete ordered set of conditions. The same canonical
inputs must produce the same conditions and identifiers regardless of host or
iteration order.

The deterministic `FakeProvider` returns identical validated responses for
identical canonical requests with zero network activity. Phase 2 implements the
complete planner, ordered schedule, dry-run and artifact identities, runner,
deterministic metric path, and compact artifact publication. `validate`, `plan`,
and `run --dry-run` construct the full schedule and identities without
constructing any provider or making HTTP calls. Fake output can establish the
offline provider-to-artifact engineering path; it cannot prove live llama.cpp
validity or a scientific model result.

## Provider identity and execution

The implemented Phase 2 llama.cpp path uses one strict completion provider with
explicit URL, identity fields, prompt-template hash, and connect/read/write/pool
timeouts. Construction inspects `/health` and `/props`; each generation repeats
the inspection before one `/completion` dispatch. There is no Ollama
compatibility, hidden environment-variable fallback, automatic model download,
redirect following, provider fallback, or automatic retry after dispatch.

The run manifest binds the exact provider identity, including llama.cpp model
alias, model path, build, prompt template, and effective configuration obtained
through validated identity inspection. The provider is invoked once for a
logical request. Provider identity drift, malformed responses, missing effective
settings, alias mismatch, or an uncertain dispatched request fails closed.

The deterministic fake provider is an explicit test selection, not a recovery
provider. Mock transport tests establish failure behavior only; live validity
requires an explicitly enabled live test or smoke run against the configured
runtime. Default CI performs zero live model calls.

## Manifest and canonical identity

Before execution, `RunManifest` binds at least:

```text
source_commit
working_tree_clean
experiment_config_hash
dataset_hash
provider_config_hash
provider_identity
policy_config_hashes
metric_versions
seed schedule
action bounds
decoding bounds
decision-rule version
database schema version
```

Scientific records use canonical UTF-8 JSON, sorted keys, compact separators,
finite values only, and lowercase SHA-256. `scientific_identity_sha256` covers
canonical scientific inputs while excluding incidental timestamps and
machine-local output paths.

Before confirmatory execution, freeze and publish the identity of:

```text
dataset versions
prompt schedule
policies and policy configurations
action bounds
base decoding profile
model seeds
metric definitions
guardrails
practical-effect thresholds
statistical methods
decision rules
provider requirements
```

No confirmatory identity field may change after execution begins. A changed
field defines a different run; it is not an in-place correction.

## Metrics and endpoint contract

Each prompt case has a deterministic objective validator that returns normalized
`task_score` in `[0, 1]`. Every metric entry records:

```text
value
availability
metric_version
input_hash
```

Unavailable optional metrics remain explicitly unavailable; they are never
silently imputed.

The exact Phase 2 tokenization, formulas, validator behavior, versions, and
availability rules are recorded in [the metric definitions](metrics.md).

The primary endpoint is:

```text
guardrail_clean_task_score
```

It is output-based and aggregated at the prompt-sequence by model-seed unit.
Guardrails gate the result rather than being blended into an opaque weighted
score. Controller-state recovery, neural stability, action movement, and
decoding trajectories are explanatory diagnostics.

Mechanism-level output recovery includes:

```text
post_stressor_task_score_change
post_stressor_repetition_change
time_to_return_to_target_band
```

Required secondary measures include repetition, repeated 3-gram and 4-gram
ratios, distinct 2-grams and 3-grams, late-window repetition, response length,
and explicitly available or unavailable semantic similarity. Repetition gains
receive no credit when explained primarily by shorter output.

Required guardrails are:

```text
instruction_adherence_non_regression
response_length_confound
matched_condition_coverage
provider_identity_stability
turn_zero_equivalence
action_bound_compliance
action_saturation_rate
behavioral_alias_detection
metric_availability
```

An optional LLM judge may be only a secondary evaluator. It uses a separate
explicit identity, blinded policy labels, randomized response order, a fixed
rubric, cached judgments, multiple orderings or judges, and reported
disagreement. Its output is never visible to a judged policy and is never the
sole primary acceptance criterion.

## Statistical comparison contract

Primary comparisons are paired on prompt sequence and model seed. Resampling
seeds are deterministic and recorded. The confirmatory evaluator uses paired
bootstrap confidence intervals, paired permutation tests, and the preregistered
multiple-comparison correction for required primary comparators.

The neural policy must be compared against `best_static`,
`heuristic_adaptive`, and `random_matched`. Positive validation requires the
preregistered practical and statistical advantage over both serious efficacy
comparators, complete coverage, all substantive guardrails, output-based
recovery, and supporting persistent-state attribution. The random arm does not
replace either serious comparator.

## Final decision states

Every confirmatory experiment returns exactly one state:

| State | Contract meaning |
| --- | --- |
| `VALIDATED_POSITIVE` | The neural controller beats every required serious comparator by preregistered practical and statistical thresholds, all guardrails pass, behavior is not comparator-equivalent, and persistent-state attribution supports a real contribution. |
| `VALIDATED_NEGATIVE` | The run is valid, but the neural controller fails a required advantage, is materially worse, or fails a substantive output guardrail. |
| `INCONCLUSIVE` | The run is valid, but uncertainty crosses the decision boundary, the practical effect remains unresolved, subgroup evidence conflicts, or preregistered optional-metric missingness requires this state. |
| `INVALID_RUN` | Provider identity, schedule, seed, coverage, matching, provenance, execution, database integrity, response evidence, metric reconstruction, or causal invariants fail. |

A clean negative result is preserved and does not automatically open another
tuning cycle. `INCONCLUSIVE` is reserved for valid but unresolved evidence.
Integrity failures produce `INVALID_RUN`; they are not uncertainty.

## Transaction and resume contract

Phase 2 implements one SQLite database as the canonical mutable store, with
unique constraints that prevent a logical request from being committed twice.
Canonical reads are rehashed and integrity-checked rather than trusted as
report text.

For each logical generation, the transaction protocol is:

1. Resolve the condition.
2. Validate prerequisite history.
3. Persist the prepared condition and request.
4. Mark the request `DISPATCHING`.
5. Invoke the provider once.
6. Persist the raw response.
7. Calculate deterministic metrics.
8. Persist policy state and history commitment.
9. Mark the turn `COMMITTED`.

On resume, committed turns are never regenerated. Prepared but never dispatched
turns may resume safely. A dispatched turn lacking a valid persisted response
fails closed unless the provider supports a verified idempotency mechanism.
Uncertain generations are never silently retried.

The closed run has one compact artifact set:

```text
run.sqlite3
manifest.json
results.csv
comparisons.csv
decision.json
report.md
```

Optional plots share one `plots/` directory. There is no per-turn request,
response, hash, or validation directory forest. Reports are derived views and
must not replace the canonical database or mutate the scientific decision.

## Phase gates and claim limits

Phase 1 established the foundation contracts, deterministic identities,
fake-provider behavior, and zero-network test seam. Phase 2 may establish the
provider-to-artifact execution path, transactional resume, deterministic metric
reconstruction, and strict provider contract behavior, but no live-provider
claim without an explicit live run and no policy advantage under any
circumstances. Phase 3 may establish fair baseline and statistical behavior but
no neural result. Phase 4 may establish neural causal activity and clean
persistent-state isolation but no model-backed benefit. Only the frozen Phase 5
confirmatory experiment may produce the final scientific decision.

The five phases execute in order. Production modules and runtime identifiers
remain domain-based, no live model call occurs in default tests, sealed
evaluation data is never used for development selection or tuning, and no
result is overstated beyond the gate that generated it.

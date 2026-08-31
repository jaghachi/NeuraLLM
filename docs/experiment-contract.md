# NeuraLLM experiment contract

## Contract status

This document defines the scientific invariants that implementation and
preregistration must preserve. The source tree reports version `2.0.0b1` and
implements the Phase 5 offline-readiness boundary: the Phase 2 deterministic
experiment kernel, Phase 3 baselines/evaluator, Phase 4 transparent neural
mechanism, and the frozen model-backed tiers, preregistration, durable execution
accounting, confirmatory decision engine, persistence, and reporting path.

The checked-in Phase 3 configurations exercise the baseline evaluator offline;
the Phase 4 causal harness establishes neural controller activity and reset
isolation under the deterministic fake provider. None establishes live
llama.cpp validity, neural benefit, a model-backed persistent-state effect, a
confirmatory result, or a final scientific outcome. `scientific_decision`
remains null.

Phase 5 now has frozen model-backed dataset identities and an exact tiered
protocol. This is readiness evidence only: neither the engineering smoke nor
development pilot supports a scientific claim, and no checked-in identity,
fake-provider run, or offline decision fixture proves that a live request was
dispatched. Until an explicitly authorized live smoke succeeds, the state is
`READY_FOR_LIVE_SMOKE`.

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

### Implemented Phase 3 baseline evaluator

The current population is:

```text
best_static
random_matched
heuristic_adaptive
```

`best_static` is the serious comparator, `random_matched` is a negative
control, and the checked-in evaluator specifications use
`heuristic_adaptive` as the focal policy. Every arm is created from a strict
typed specification before execution. `best_static` and `random_matched`
declare no history access; `heuristic_adaptive` may observe only its own
previous committed response metrics. All three produce exact zero actions at
turn zero.

Static selection accepts only a `development` dataset. Candidate profiles must
have equal development-unit coverage; the winner is the highest mean
sequence-task-score candidate with lexical profile ID as the deterministic tie
break. The complete candidate grid, development dataset hash, winner, rule, and
selection-result hash are frozen before an evaluation or synthetic run is
planned. The winning profile must exactly equal the experiment's base decoding
profile.

The Phase 3 evaluator rejects development-purpose data. Evaluation-purpose data
requires a matching external seal; synthetic-purpose data must be unsealed. The
checked-in offline fixtures are frozen as:

| Purpose | Dataset | Canonical identity |
| --- | --- | --- |
| Development | `datasets/development/phase3-baseline-development-v1.yaml` (6 sequences, 24 prompt turns) | `a6c41a046cb84bc9a806866a7393196784eb769118f74cbe4d44d0f3e247df97` |
| Evaluation | `datasets/evaluation/phase3-baseline-evaluation-v1.yaml` (8 sequences, 32 prompt turns) | Dataset `de4c415d71cc3ed0177b189880fa9da040464f41ab14b192bab01cb4eed09199`; seal `89e794e8c80094c15ba9be801306f9ca8090fbd45ab9462b92c172a4a3b65847` |
| Synthetic | `datasets/synthetic/phase3-evaluator-validation-v1.yaml` (4 sequences, 16 prompt turns) | `192d7f5f092eb628cbbc25316aefcbbabd89e4e18dd5180be9e72e5ad426ffbf` |

The frozen development selection compares three profiles over the same 12
explicitly keyed `prompt sequence x model seed` units, holds `max_tokens` fixed
at 192, selects `static-balanced-v1`, and has selection-result SHA-256
`19be248a50cf6504011168d1e79e3e3cd24d1027017a6cbec443b9019a0bf301`.

This is an offline fixture identity, not proof that a confirmatory dataset was
evaluated with a live model.

### Implemented policy, future end-to-end efficacy

The intended efficacy population contains:

```text
best_static
random_matched
heuristic_adaptive
neural_persistent
```

The first three arms retain their Phase 3 behavior, and
`neural_persistent` is now implemented. A future efficacy run must keep every
policy independent:

- it produces its own response;
- it observes metrics only from its own previous response;
- it carries only its own explicitly declared state; and
- its next action cannot depend on another policy's output or state.

The required serious primary comparators would be `best_static` and
`heuristic_adaptive`. `random_matched` remains a negative-control and
structured-action sanity comparison. Phase 4 does not place
`neural_persistent` into the Phase 3 evaluator or make an efficacy claim.

### Implemented matched-history attribution mechanism

The Phase 4 attribution harness pairs:

```text
neural_persistent
neural_matched_history_state_reset
```

For each attribution turn, the reset arm:

- receives the exact committed previous-turn `ResponseMetrics` tuple from the
  paired focal `neural_persistent` condition;
- preserves the real prompt-sequence turn index;
- preserves the focal history-presence semantics;
- loads the same focal controller-state envelope, then resets only the declared
  five-variable neural substrate at the intervention boundary;
- uses the same current prompt, model seed, action decoder, provider, base
  decoding parameters, and action bounds; and
- never feeds its own response metrics into focal history.

`neural_matched_history_state_reset` is attribution-only. It must not enter the
independent efficacy population, be summarized as an operating baseline, or
replace any required efficacy comparator.

At turn zero the paired attribution arms must be byte-equivalent before the
state-reset intervention can have meaning. At later turns, focal-history hashes
must match and tests must establish that only declared persistent state differs.
Phase 4 tests establish this mechanism-level isolation and deterministic
fake-provider activity. They do not establish a beneficial model-output effect.

## Frozen model-backed population and schedules

The model-backed protocol contains exactly five arms, in this reporting order:

| Arm | Analysis role |
| --- | --- |
| `best_static` | Independent efficacy arm and serious comparator |
| `random_matched` | Independent efficacy arm and negative control |
| `heuristic_adaptive` | Independent efficacy arm and serious comparator |
| `neural_persistent` | Independent efficacy focal arm |
| `neural_matched_history_state_reset` | Persistent-state attribution only |

The first four arms participate in end-to-end efficacy. The fifth consumes the
matched focal history required for the declared reset intervention and is
excluded from all efficacy estimates and operating-baseline summaries.

Request accounting is the exact Cartesian product below. The single
controller seed is an explicit nested replicate and does not enlarge the
statistical unit.

| Tier | Sequences | Turns | Model seeds | Controller seeds | Arms | Logical generations | Claim boundary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Engineering smoke | 2 | 2 | 1 | 1 | 5 | 20 | Engineering validation only; no scientific claim |
| Development pilot | 6 | 4 | 2 | 1 | 5 | 240 | Development calibration only; no scientific claim |
| Confirmatory | 24 | 4 | 5 | 1 | 5 | 2,400 | Eligible for one final decision only if all frozen identities and integrity gates pass |

The checked-in model-backed datasets are frozen as:

| Purpose | Dataset | Exact shape | Canonical identity |
| --- | --- | --- | --- |
| Development | `datasets/development/model-backed-engineering-smoke-v1.yaml` | 2 sequences by 2 turns | `14c382a04acbe9394474f05cf84d8389833058afc2dc6feda21a023d46e45ef3` |
| Evaluation | `datasets/evaluation/model-backed-confirmatory-v1.yaml` | 24 sequences by 4 turns | `7cf2d3a9fa35735aadc9186438277d2b5f6b7beb9f96e9fc9bbeb400da2b5d72`; must match `model-backed-confirmatory-v1.seal.yaml` |

The pilot may use development data only. It may identify broken metrics,
calibrate thresholds, select the static baseline, and finalize validators and
bounds, but it cannot inspect or tune against the sealed confirmatory
responses. All confirmatory identities are immutable once execution begins.

## Experimental unit and condition identity

The primary statistical unit is:

```text
prompt sequence x model seed
```

Turns within one sequence are correlated observations and must not be treated as
independent samples. The implemented aggregation is
`mean-controller-seed-then-turn-v1`: metrics are averaged across turns within
each controller seed, then the controller-seed means are averaged within each
`prompt sequence x model seed` unit. Turns and controller seeds are nested
replicates and never increase the paired sample size.

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
retry count are not scientific identity fields. Before statistics, the evaluator
materializes and compares the exact Cartesian grid of prompt-sequence turns,
policies, model seeds, and controller seeds. Missing, unexpected, duplicate,
dataset-mismatched, or provider-mismatched evidence yields the Phase 3 verdict
`invalid` with zero statistical calls; it is never silently repaired.

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

- the same policy's turn `t - 1` response for the implemented baseline paths
  and independent efficacy; or
- the paired focal persistent arm's exact committed turn `t - 1` metric tuple
  for the implemented matched-history attribution mechanism.

The current response can affect only a future action. Stale, uncommitted,
wrong-policy, wrong-turn, wrong-sequence, or hash-mismatched history fails closed.

The run manifest records the only permitted cross-policy source:
`neural_matched_history_state_reset -> neural_persistent`. Plans with this edge
are turn-interleaved and their complete causal predecessor graph is validated
before any provider dispatch. Both later neural arms record the focal condition,
commitment, and canonical metrics hashes in a Phase 4 causal trace.

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
identical provider-visible prompt, decoding, provider-identity, and provider-
configuration inputs with zero network activity. Policy IDs, controller seeds,
and other orchestration-only condition fields cannot affect its response bytes;
the full canonical request hash remains retained as provenance metadata. Phase
2 implements the complete planner, ordered schedule, dry-run and artifact identities, runner,
deterministic metric path, and compact artifact publication. Phase 3 adds typed
policy-runtime construction, matched-unit expectations, prompt-side evaluator
evidence, closed-run analysis, and durable comparison evidence. `validate`,
`plan`, and `run --dry-run` validate the development-selection record,
dataset identity and seal when required, policy/evaluator specifications, and
the full schedule without constructing any provider or making HTTP calls.

The checked-in development, evaluation, and synthetic fixtures are intentionally
offline. A provider-free scenario harness consumes the synthetic dataset's
known-superior, known-inferior, identical/equivalent, and length-confound codes
as four independent evaluator runs. Pure evaluator tests additionally prove
incomplete coverage is invalid before any statistical function is called. The
ordinary fake-provider workflow remains an engineering/replay check and does
not claim to manufacture those known outcomes.
Fake output can establish the provider-to-artifact and evaluator engineering
paths and the Phase 4 mechanism-level reset isolation. It cannot prove live
llama.cpp validity, neural efficacy, a beneficial model-backed persistent-state
effect, or a scientific model result.

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

For Phase 3, the plan also binds the dataset purpose, canonical seal when
required, `EvaluationSpec` and its hash, the keyed development-only
`StaticSelectionRecord`, and exact matched-unit expectations. Before execution,
`RunManifest.phase3_analysis_contract_sha256` freezes the plan, evaluator,
selection, evaluation design, purpose, dataset, and seal identities. The
schema-v2 `AnalysisManifest` repeats that evidence with the finalized run,
scientific-result, and canonical evaluator-input hashes; persistence and reads
recompute the contract digest and fail closed on any foreign evidence.

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

The exact response-level tokenization, formulas, validator behavior, versions,
availability rules, and Phase 3 aggregation/statistical rules are recorded in
[the metric definitions](metrics.md).

The implemented Phase 3 primary metric is `task_score`. Required evaluator
inputs also include `instruction_adherence`, `response_length_tokens`,
`repetition_ratio`, normalized action magnitude, action-bound compliance, and
action saturation. These are aggregated at the prompt-sequence by model-seed
unit. Guardrails gate the result rather than being blended into an opaque
weighted score. Controller movement is diagnostic only.

The implemented confirmatory endpoint is `guardrail_clean_task_score`: the raw
task score remains auditable, but its gated value is available only when the
declared guardrails pass. The implemented recovery measures are
`post_stressor_task_score_change`,
`post_stressor_repetition_change`, and
`time_to_return_to_target_band`. A unit that does not return during the frozen
recovery window is retained as right-censored at window length plus one; it is
never silently dropped. These Phase 5 endpoints are separate from and are not
computed by the Phase 3 evaluator.

The stored response tuple still includes repetition, repeated 3-gram and 4-gram
ratios, distinct 2-grams and 3-grams, late-window repetition, response length,
and explicitly unavailable semantic similarity. Phase 3 uses response length
and repetition to reject an apparent repetition improvement that is explained
by excessive output shortening.

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

No LLM judge is implemented or invoked in Phase 3. Any future optional judge
would be secondary, use a separate explicit identity, remain invisible to the
judged policy, and never become the sole primary acceptance criterion.

## Statistical comparison contract

Phase 3 comparisons are focal-minus-comparator differences paired on prompt
sequence and model seed. The `EvaluationSpec` records every resampling seed,
resample count, confidence level, practical-effect threshold, equivalence
margin, multiplicity method, and guardrail threshold.

For valid exact coverage, the evaluator computes:

1. a seeded paired-bootstrap percentile interval over matched-unit differences;
2. a two-sided paired sign-flip permutation test, exact for at most 20 units
   when all sign patterns fit within the configured resample budget and
   otherwise deterministic Monte Carlo with an add-one correction; and
3. Holm-adjusted p-values over required serious comparators only.

Negative controls receive the raw permutation result and do not enter the Holm
family. A pair is `superior` only when its mean improvement meets the practical
threshold, its bootstrap lower bound is positive, and its applicable p-value is
at most alpha. The symmetric rule yields `inferior`; equivalence-margin or
behavioral-alias evidence yields `equivalent`; otherwise the pair is
`inconclusive`. Substantive adherence, response-length, or focal-saturation
failure yields `inferior`. The overall verdict considers serious comparators:
all superior yields `superior`, all equivalent yields `equivalent`, any
inferior yields `inferior`, and all other valid combinations yield
`inconclusive`.

This remains the Phase 3 decision skeleton. The separate implemented Phase 5
confirmatory evaluator compares `neural_persistent` against both serious
efficacy comparators and reports `random_matched` only as the negative control;
it never substitutes that control for either serious comparator. The Phase 4
mechanism harness alone does not support neural benefit or a model-backed
persistent-state effect.

Phase 5 recovery reduction is also comparator-exact. For each preregistered
recovery event and model seed, each endpoint is oriented so a positive margin
favors `neural_persistent`: focal minus comparator for both post-stressor change
endpoints, and comparator minus focal for time to return. The analysis retains
the minimum oriented margin across `best_static` and `heuristic_adaptive` for
that event/model-seed unit. It never averages the two serious comparators.

The stochastic negative-side evidence used to reach `VALIDATED_NEGATIVE` has
one exact seven-member family: all three efficacy comparisons, all three
recovery endpoints, and the persistent-state attribution comparison. A
Bonferroni simultaneous two-sided bootstrap uses familywise alpha `0.05` and
confidence `1 - 0.05 / 7 = 0.9928571428571429`. Substantive deterministic
guardrail failures, behavioral-alias findings, and focal right-censor failures
are direct decision gates; they are not members of this stochastic
multiplicity family. The adjustment is negative-side only: positive gates keep
their nominal 95% bootstrap intervals, and positive efficacy tests across
`best_static` and `heuristic_adaptive` keep the preregistered Holm correction.

The confirmatory result persists the complete preregistered analysis spec and
its canonical SHA-256. On model load, all nominal and adjusted bootstrap seeds,
resample counts, confidence levels, permutation settings, and practical
thresholds are checked against that embedded spec. On durable persistence, the
embedded spec and hash must also equal the pre-execution spec in the scientific
analysis manifest. Prompt-family sensitivity bootstraps are persisted as typed
subgroup results tied to the exact unit-level family assignments; the reported
statistical-call count is derived from those and the other enclosed statistical
objects.

Before any confirmatory request is dispatched, all scheduled turns must produce
one exact `prompt_sequence_id -> prompt_family` mapping, with exactly one family
for each sequence. Canonical mapping bytes and their SHA-256 are frozen in the
analysis contract, repeated in the scientific manifest and result, and exposed
in `decision.json`. The result's unit keys and family labels must cover that
mapping exactly.

The v2 result also closes the path from raw evidence to the decision. It stores
unit-level efficacy scores, recovery margins and censor indicators,
persistent-minus-reset attribution differences, and optional-metric
availability. Aggregate evidence is recomputed from those values. The limitation
tuple must then equal the complete derived optional-metric, right-censoring, and
prompt-family-conflict set; a caller cannot add, remove, or edit a limitation to
change the decision.

The current confirmatory boundary uses decision rule
`confirmatory-scientific-decision-v2`, `confirmatory-analysis-v2`,
`confirmatory-evaluation-v2`,
`confirmatory-scientific-analysis-storage-v2`, and Phase 5 `decision.json`
schema 2. These versions move together and are incompatible with provisional v1
scientific envelopes. SQLite's physical schema remains version 2, but v1
scientific-analysis rows are rejected rather than migrated. Their run manifests
remain v1-bound, so replacement requires a new v2 confirmatory workflow in a
fresh run directory rather than an in-place offline reanalysis.

The run manifest also binds canonical `EvaluationSpec` JSON and its SHA-256,
plus the SHA-256 of the complete condition-keyed `TurnInputEvidence` tuple.
Persistence requires exact prompt-side input coverage, recomputes deterministic
response metrics from each committed input and response, and reconstructs
guardrail status and thresholds from committed traces and metrics under that
pre-execution spec before accepting a scientific result.

## Phase 3 and final decision states

The implemented Phase 3 verdict set is:

| Verdict | Contract meaning |
| --- | --- |
| `superior` | The focal baseline clears the configured practical, interval, and p-value rules against every serious comparator without a substantive guardrail failure. |
| `inferior` | A serious comparison is statistically/practically inferior or a substantive guardrail fails. |
| `equivalent` | Every serious comparison is within the equivalence rule or behaviorally aliased. |
| `inconclusive` | Evidence is valid but does not satisfy the superior, inferior, or all-equivalent rule. |
| `invalid` | Exact coverage or an integrity guardrail fails before statistics. |

These values are stored under claim scope
`phase-3-statistical-behavior-only`. They never populate
`scientific_decision`.

The frozen Phase 5 confirmatory protocol permits exactly one final state:

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

One SQLite database remains the canonical mutable store, with unique constraints
that prevent a logical request from being committed twice. The current physical
schema is version 2 and migrates older stores; Phase 2 manifests declaring
database schema version 1 remain supported. Version 2 adds immutable prompt-side
turn inputs and durable Phase 3 analysis tables. Canonical reads are rehashed
and integrity-checked rather than trusted as report text.

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

After a schema-v2 Phase 3 run is finalized, offline analysis:

1. verifies the database, run manifest, run finalization, exact plan conditions,
   and stored prompt-side evidence;
2. reconstructs typed evaluator records only from committed store evidence;
3. validates exact coverage and integrity before any statistical call;
4. persists the analysis manifest, comparison rows, guardrail rows, Phase 3
   decision skeleton, and analysis finalization atomically; and
5. reads back and hash-validates the finalized analysis before export.

Repeated persistence of identical evidence is idempotent. A different analysis
after finalization, a missing finalized run, or hash-mismatched analysis evidence
fails closed.

A confirmatory analysis additionally requires a clean-tree llama.cpp manifest
whose identity binds the measured model-artifact SHA-256, the exact five-arm
schedule and matched-history edge, zero uncertain dispatches,
complete planned/dispatched/successful/committed durable accounting, the frozen
preregistration and analysis-contract hashes, and the sealed evaluation dataset.
It reconstructs metrics and causal evidence from committed records, computes
the three efficacy comparisons plus the attribution-only comparison and
recovery evidence, derives exactly one typed decision, and atomically persists
and reads back the bound scientific result before export. Smoke and pilot tiers
are ineligible for this path.

The closed run has one compact artifact set:

```text
run.sqlite3
manifest.json
results.csv
comparisons.csv
decision.json
report.md
```

The current exporter rejects every additional file. There is no plot directory
or per-turn request, response, hash, or validation forest. Reports are derived
views and must not replace the canonical database or mutate the Phase 3 verdict
or final scientific decision.

For historical Phase 2 runs, `comparisons.csv` remains header-only and
`decision.json` retains the `engineering_validation_only` claim scope. For a
finalized Phase 3 analysis, `comparisons.csv` records the paired estimates,
bootstrap bounds, sign-flip results, Holm values where applicable, alias flag,
guardrail statuses, and pair verdict. `decision.json` records the Phase 3
verdict and canonical analysis identities, and `report.md` separates baseline
evaluator validation, controller activity, guardrails, end-to-end efficacy,
persistent-state attribution, limitations, and the Phase 3 result.
`scientific_decision` remains null.

A final Phase 5 `report.md` must preserve separate, plainly labeled sections
for `engineering validity`, `controller activity`, `end-to-end efficacy`,
`persistent-state attribution`, `guardrail outcomes`, `limitations`, and
`final decision`. The final decision vocabulary is limited to
`VALIDATED_POSITIVE`, `VALIDATED_NEGATIVE`, `INCONCLUSIVE`, and `INVALID_RUN`;
Phase 3 verdict words and readiness labels are not substitutes.

## Phase gates and claim limits

Phase 1 established the foundation contracts, deterministic identities,
fake-provider behavior, and zero-network test seam. Phase 2 established the
provider-to-artifact execution path, transactional resume, deterministic metric
reconstruction, and strict provider contract behavior, but no live-provider
claim without an explicit live run and no policy advantage under any
circumstances. Phase 3 establishes typed baseline and statistical-evaluator
behavior under offline fake fixtures, frozen dataset identities, and synthetic
known-outcome tests. It establishes no live-provider, neural efficacy, or
model-backed persistent-state result. Phase 4 establishes deterministic neural
mechanism activity and clean matched-history substrate-reset isolation under the
fake provider, but no model-backed benefit. Phase 5 implements the frozen tier,
analysis, persistence, and report contracts offline; only an actually executed,
claim-eligible frozen confirmatory experiment may produce an observed final
scientific decision.

The five phases execute in order. No live model call occurs in default tests,
sealed evaluation data is never accepted for development selection or tuning,
and no result is overstated beyond the gate that generated it.

Live llama.cpp execution additionally requires a successful explicit preflight
that hashes the client-local model artifact and makes no generation request,
`neurallm preflight --provider-config <path>`, against `/health` and `/props`,
followed by the double CLI gate: both `--execute` and
`--allow-live-provider` on `neurallm run --config <path>`. Preflight alone or
either execution flag alone authorizes no generation; no environment fallback
may fill in these choices.

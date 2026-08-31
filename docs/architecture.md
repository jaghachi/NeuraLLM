# NeuraLLM architecture

## Status and purpose

This living contract describes the NeuraLLM 2.0 architecture through the Phase
5 offline-readiness boundary. The source tree reports version `2.0.0b1`. It
preserves the Phase 2 experiment kernel, Phase 3 baseline evaluator, and Phase 4
transparent five-state simulated neural mechanism, then adds the frozen Phase 5
tier protocol, preregistration and provider preflight surfaces, durable logical-
generation accounting, confirmatory analysis, atomic scientific persistence,
and deterministic final reporting.

Phase 4 establishes deterministic controller activity and causal isolation
under the fake provider. The checked-in Phase 5 protocol, engineering fixture,
sealed confirmatory dataset, and offline tests establish execution readiness
only. No live llama.cpp smoke, pilot, or confirmatory run has occurred, so the
repository's truthful state is `READY_FOR_LIVE_SMOKE`; it contains no observed
model-backed benefit, persistent-state effect, or final scientific outcome.

NeuraLLM asks one falsifiable question:

> Can a simulated neural controller regulate LLM decoding more effectively
> than a strong fixed sampler, bounded random perturbation, and a competent
> non-neural adaptive controller under matched model, prompt, seed, and
> generation-budget conditions?

A valid negative result is a successful scientific outcome. The architecture
therefore prioritizes causal isolation, explicit identity, deterministic
reproduction, and fail-closed behavior over maximizing a controller score.

The historical `jaghachi/neurollm` repository is a read-only scientific and
design reference. NeuraLLM is a new implementation: it does not import the
legacy runtime, retain phase-numbered runtime APIs, or copy the old controller
monolith. Any legacy mechanism must pass the porting gate in
`docs/legacy-lessons.md` before it is reimplemented.

## Architectural shape

Production code is organized by domain, not by implementation phase.

| Domain | Responsibility | Must not own |
| --- | --- | --- |
| `domain` | Immutable models, identifiers, canonical serialization, and hashes | Provider clients, policy logic, storage side effects |
| `providers` | The common generation boundary, deterministic fake generation, and the strict llama.cpp adapter | Policy selection, metrics, retries, fallback routing |
| `control` | Shared policy protocol, action bounds, static/random/heuristic baselines, and the composed simulated neural controller | Provider construction, experiment scheduling, evaluation decisions |
| `metrics` | Deterministic validators and versioned output metrics | Controller state transitions or policy feedback routing |
| `experiments` | Deterministic planning, matching, scheduling, execution, resume, preregistration, and hash-bound analysis orchestration | Policy-specific mode dispatch, generation transport, or alternate decision rules |
| `evaluation` | Exact coverage, sequence-level aggregation, guardrails, paired statistics, recovery, attribution, and typed Phase 3 and Phase 5 decisions | Generation or mutation of source run evidence |
| `storage` | Transactional run and analysis persistence, manifests, integrity checks, and crash-safe resume | Scientific policy or retry decisions |
| `reporting` | Compact, reproducible views derived from the canonical run store | Recomputing or changing scientific truth |
| `cli` | Explicit composition root and command surface | Import-time clients, hidden defaults, or alternate scientific paths |

Dependencies point toward typed domain contracts. The runner composes policies,
providers, metrics, and storage; those components do not reach back into the CLI
or choose one another through mode strings. Provider and policy runtime
identifiers are domain-based; the explicitly scoped Phase 3 evaluator and
decision-rule versions retain Phase 3 in their names so their limited claim
scope cannot be mistaken for a final scientific decision.

## Shared policy boundary

Every experimental arm implements the same policy protocol:

```python
class ControlPolicy(Protocol):
    policy_id: str

    def initial_state(self, context: PolicyContext) -> PolicyState: ...

    def act(
        self,
        observation: ControllerObservation,
        state: PolicyState,
    ) -> tuple[ControllerAction, PolicyState, PolicyTrace]: ...
```

The composition root maps the strict discriminated `PolicySpec` union to a
runtime before execution; the runner does not branch on legacy policy mode
strings in the turn loop. The implemented variants are:

- `BestStaticPolicySpec` / `best_static`: stateless, no history access, and
  zero deltas from the development-selected base profile.
- `RandomMatchedPolicySpec` / `random_matched`: no history access, an exact
  zero action at turn zero, and deterministic SHA-256-derived draws within all
  four shared action bounds on later turns.
- `HeuristicAdaptivePolicySpec` / `heuristic_adaptive`: explicit typed state,
  an exact zero action at turn zero, access only to its own previous response
  metrics, declared repetition/adherence/length reactions, and clean-response
  decay toward zero.
- `NeuralPersistentPolicySpec` / `neural_persistent`: a five-variable bounded
  substrate, own-previous-response history, and persistent serialized state.
- `NeuralMatchedHistoryStateResetPolicySpec` /
  `neural_matched_history_state_reset`: the same mechanism and decoder, exact
  focal previous history, and one declared substrate-only reset intervention.

Policy state is explicit and passed through the common interface; global mutable
controller state is prohibited. Policy-specific traces are nested inside one
common applied-action trace in schema-v2 runs.

The neural policy is composed rather than implemented as a controller monolith:

```text
ControllerObservation
    -> ObservationEncoder
    -> EncodedObservation
    -> NeuralSubstrate
    -> ActionDecoder
    -> ControllerAction
```

The substrate state is exactly `excitation`, `inhibition`, `adaptation`,
`fatigue`, and `context`. Initial values and per-turn seed drive are derived
from the declared controller seed through canonical SHA-256. Five fixed linear
update equations are clipped and quantized to explicit bounds; there are no
learned weights, global RNG, mutable object state, I/O, or hidden feedback.
The trace records the encoding, stored/effective/pre/post substrate states,
per-variable saturation, decoder activations, normalized action magnitude, and
reset marker. Turn zero gates the action to exact zero without inventing prior
metrics.

## Shared provider boundary

All generation implementations satisfy one typed `GenerationProvider`
protocol. At the architecture level, that boundary accepts a validated,
identity-bound generation request and returns a validated raw generation result.
It does not expose controller-specific methods.

Provider construction occurs only in the explicit composition root after
configuration validation. There is no import-time construction, implicit
current-working-directory lookup, hidden environment-variable fallback,
automatic model download, provider fallback, or automatic retry after a request
has been dispatched.

The deterministic `FakeProvider` is an explicitly selected testing
implementation, never a fallback for a failed live provider.
Its response bytes hash only provider-visible prompt, decoding parameters,
provider identity, and provider configuration. Policy IDs, controller seeds,
and other orchestration-only condition metadata cannot affect the response;
the full canonical request hash is still retained in generation metadata. It
makes no network call, giving domain, policy, identifier, and CLI contracts a
complete zero-network test seam.

The Phase 2 generation adapter is a strict llama.cpp completion provider. Its
explicit configuration supplies the server URL, expected model alias, model
path, build ID, prompt-template hash, and four HTTP timeouts. Construction
inspects `/health` and `/props`; every generation repeats that identity check
before one `/completion` dispatch. Drift, malformed payloads, and effective
setting mismatches fail closed. HTTP environment integration and redirects are
disabled, and no retry, fallback, or model download exists. See
[the provider runbook](provider-runbook.md). Contract tests do not establish
live-provider validity. Ollama compatibility is not part of the initial
rebuild. An optional blinded judge, if later enabled, uses a separate explicit
evaluation-provider identity and never becomes an implicit generation fallback.

The live boundary has three explicit steps. First,
`neurallm preflight --provider-config <path>` may inspect only the llama.cpp
identity endpoints (`/health` and `/props`) and performs no generation. A live
run is then authorized only when `neurallm run --config <path>` includes both
`--execute` and `--allow-live-provider`. Preflight is not an execution gate by
itself, either execution flag alone is insufficient, and no environment
fallback supplies any of these choices.

## Domain and identity contracts

Core boundaries use strict immutable models rather than untyped dictionaries.

| Model | Contract |
| --- | --- |
| `DecodingParameters` | `temperature`, `top_p`, `top_k`, `presence_penalty`, fixed `max_tokens`, and `seed` |
| `ControllerAction` | Bounded deltas for the four controllable parameters; it cannot change `max_tokens` |
| `ControllerObservation` | Only the current turn index, prompt family, current prompt features, nullable prior-response metrics, and an explicit history-presence flag |
| `ResponseMetrics` | Output measures whose entries carry value, availability, metric version, and input hash |
| `ExperimentCondition` | Unique binding of experiment, dataset, prompt sequence, turn, policy, model seed, controller seed, provider identity, and base decoding profile |
| `RunManifest` | Source and clean-tree provenance; configuration, data, provider, policy, metric, seed, bounds, decision rule, and database schema; the matched-history edge; and applicable pre-execution Phase 3 or Phase 5 identities |
| `DatasetSeal` | Evaluation purpose, dataset ID/version, dataset SHA-256, and canonical seal identity |
| `StaticSelectionRecord` | Development-only candidate grid, fixed `max_tokens`, canonical matched-unit keys, aligned score vectors, deterministic winner, and selection-result hash |
| `EvaluationSpec` | Focal/comparator roles, aggregation and statistical methods, seeds, thresholds, and guardrail limits |
| `AnalysisManifest` | Hash-bound connection between a finalized run, plan, evaluator input, static selection, dataset purpose, and seal |
| `ScientificAnalysisManifest` | Hash-bound connection between one finalized confirmatory run, its preregistration, frozen confirmatory-analysis contract, sealed dataset, and evaluation result |

Canonical scientific serialization is UTF-8 JSON with sorted keys, compact
separators, and non-finite values rejected (`allow_nan = false`). Hashes are
lowercase SHA-256. Deterministic condition identifiers are derived from the
complete condition identity, not filesystem paths, timestamps, process order,
or incidental machine state.

Dataset purpose is a typed boundary: `development` is accepted for static
selection but rejected by the statistical evaluator; `evaluation` requires a
matching seal; and `synthetic` is explicitly unsealed. The checked-in Phase 3
evaluation fixture binds dataset SHA-256
`de4c415d71cc3ed0177b189880fa9da040464f41ab14b192bab01cb4eed09199` and seal
SHA-256 `89e794e8c80094c15ba9be801306f9ca8090fbd45ab9462b92c172a4a3b65847`.
Both Phase 3 experiment fixtures use `FakeProvider`; the seal does not turn an
offline fixture into a confirmatory or live-provider run.

## Strict causal observation surface

`ControllerObservation` contains only information available before the current
response is generated:

```text
turn_index
prompt_family
current_prompt_features
previous_response_metrics | null
has_previous_response
```

It never exposes future or current-response metrics, comparator outcomes, final
objective labels, response hashes as optimization targets, another policy's
state, or aggregate results from the sealed confirmatory set.

At turn zero the representation is exact:

```text
previous_response_metrics = null
has_previous_response = false
```

Neutral numeric values are not prior history. A policy may derive internal
calculation defaults only after observing the null value, and traces must keep
those defaults distinguishable from measured history.

## Controlled generation loop

The intended logical order for one turn is:

1. Resolve one deterministic condition and its prerequisite committed history.
2. Build the causally valid observation.
3. Invoke the selected policy with explicit state.
4. Step-clamp the action, then legally clamp the resulting decoding parameters.
5. Preserve the plan-bound `max_tokens`; the controller cannot alter it.
6. Invoke the selected provider once.
7. Validate and retain the raw response.
8. Calculate deterministic, versioned output metrics.
9. Commit the resulting metrics and policy state as the only allowed history for
   a later turn.

Raw action, step-clamped action, final legal parameters, and saturation
indicators remain distinct trace fields. Output behavior is the primary
evidence; controller activity is diagnostic mechanism evidence.

Phase 2 implements this loop for the explicitly configured `kernel_fixed`
policy with the fake or strict llama.cpp provider. Phase 3 uses the same loop
through a typed runtime factory for `best_static`, `random_matched`, and
`heuristic_adaptive`, without policy-specific execution branches. Schema-v2
runs additionally bind immutable prompt-side evidence needed to reconstruct
evaluation inputs. Fake-provider execution establishes an offline
provider-to-artifact and evaluator-validation path; it must not be described as
live-provider validation or model efficacy.

Phase 4 preserves that loop and adds a manifest-declared causal predecessor.
Plans containing matched history are deterministically interleaved by logical
turn. Before any dispatch, the runner verifies that every declared predecessor
exists and is scheduled earlier. At turn `t > 0`, both neural arms bind the
exact committed `neural_persistent[t-1]` condition for metrics and policy-state
rehydration. The reset policy then substitutes only the seed-derived substrate;
controller seed, action bounds, and real logical turn index remain unchanged.
The request is fully determined before the current provider response exists.

## Baseline efficacy and matched-history attribution are different experiments

The implemented Phase 3 baseline population is `best_static`,
`random_matched`, and `heuristic_adaptive`. Each policy generates its own
response and carries only its declared state. Only `heuristic_adaptive` may
observe previous-response metrics, and those metrics come from its own committed
trajectory. The checked-in Phase 3 specifications use
`heuristic_adaptive` as the evaluator focal policy, `best_static` as the
required serious comparator, and `random_matched` as a negative control. This
validates baseline and evaluator behavior; it is not a neural efficacy
experiment.

Phase 4 implements `neural_persistent` for future independent-history efficacy
and separately pairs it with `neural_matched_history_state_reset` for mechanism
attribution. The reset arm is rejected from the current Phase 3 efficacy
evaluator. Its own response and stored post-reset state are never selected as
future focal history. SQLite permits only the manifest-declared
reset-to-persistent edge while requiring every other condition axis and exact
turn `t-1` to match.

The fake-provider causal harness establishes controller activity, turn-zero
provider-visible equivalence, exact focal-history matching, and isolation of the
declared substrate reset. It does not establish better model output or a
beneficial persistent-state effect; those remain Phase 5 questions.

## Storage and artifact boundary

One SQLite database is the canonical mutable record. The Phase 2 transaction,
resume, and finalization behavior remains supported. Schema version 2 adds
`turn_inputs`, `analysis_manifest`, `comparison_results`,
`guardrail_results`, `analysis_decision`, and `analysis_finalization`.
Discriminated typed records in those analysis tables support both the limited
Phase 3 result and the Phase 5 confirmatory result without creating an alternate
source of scientific truth.
Prompt-side records are immutable once the run is finalized; comparison and
guardrail rows are immutable once analysis is finalized. The analysis manifest
binds the closed run, experiment plan, evaluator spec, development-only static
selection, dataset identity and seal when required, and evaluator input hash.
The run manifest freezes a digest of that Phase 3 analysis contract before the
first provider call. Analysis persistence and reads recompute the digest, so a
foreign plan, selection, evaluator design, purpose, dataset, or seal fails
closed. Analysis members and their finalization are canonical-hash-validated,
transactional, idempotent, and rechecked on read.

Unique constraints prevent duplicate logical requests. A request moves through
prepared, dispatching, response-persisted, metrics-computed, and committed work
within explicit transactions. Committed turns are never regenerated. A
dispatched request with uncertain outcome fails closed unless the provider
offers a verified idempotency mechanism; it is never silently retried.
The existing predecessor columns also represent Phase 4 focal-history edges;
no schema migration is needed. Their source policy is authorized by the
immutable run manifest, and the predecessor commitment continues to bind the
request, response, metrics, policy state, trace, and earlier commitment.

Closed runs export only:

```text
run.sqlite3
manifest.json
results.csv
comparisons.csv
decision.json
report.md
```

The five non-database files are deterministic derived views. Phase 2 retains a
header-only `comparisons.csv` and an engineering-only null decision. Finalized
Phase 3 analysis emits deterministic comparison rows and a limited Phase 3
verdict while keeping `scientific_decision` null. Smoke and pilot exports also
keep that field null. A claim-eligible finalized confirmatory analysis emits
exactly three efficacy rows and one attribution-only row, the frozen final
decision and evidence identities in `decision.json`, and the seven required
report sections. The exporter rejects every additional file, including plot
directories and per-turn request/response forests. `scientific_identity_sha256`
covers canonical scientific inputs and excludes incidental timestamps and
machine-local output paths.

## Evaluation boundary

The implemented Phase 3 primary metric is output-based `task_score`. Turns are
first averaged within a controller seed, then controller-seed means are averaged
at the `prompt sequence x model seed` unit. Only those unit values enter paired
statistics. The evaluator requires the exact Cartesian grid of policies, turns,
model seeds, and controller seeds before aggregation or statistical calls;
missing, unexpected, duplicate, dataset-drifted, provider-drifted, out-of-bounds,
turn-zero-history-invalid, or required-metric-missing evidence returns the Phase
3 verdict `invalid` with zero statistical calls.

For valid evidence, focal-minus-comparator differences feed deterministic seeded
paired-bootstrap percentile intervals and two-sided paired sign-flip tests.
Sign patterns are enumerated exactly for at most 20 units when the configured
resample budget permits; otherwise a deterministic Monte Carlo stream with an
add-one correction is used. Holm correction applies to the serious-comparator
family. Adherence non-regression, response-length confounding, action
saturation, and behavioral aliasing remain explicit guardrails rather than
components of a weighted score.

Phase 3 results use only `superior`, `inferior`, `equivalent`,
`inconclusive`, or `invalid` with claim scope
`phase-3-statistical-behavior-only`. These validate the evaluator skeleton and
baseline behavior. They are not the final Phase 5 states
`VALIDATED_POSITIVE`, `VALIDATED_NEGATIVE`, `INCONCLUSIVE`, or
`INVALID_RUN`; `scientific_decision` remains null.

## Model-backed protocol boundary

Phase 5 freezes exactly five arms in this reporting order:

1. `best_static`
2. `random_matched`
3. `heuristic_adaptive`
4. `neural_persistent`
5. `neural_matched_history_state_reset`

The first four form the independent end-to-end efficacy population. The fifth
is a matched-focal-history, state-reset intervention used only for
persistent-state attribution; it must never enter efficacy estimates.

The three frozen run tiers have exact request-accounting schedules:

| Tier | Sequences | Turns per sequence | Model seeds | Controller seeds | Arms | Logical generations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Engineering smoke | 2 | 2 | 1 | 1 | 5 | 20 |
| Development pilot | 6 | 4 | 2 | 1 | 5 | 240 |
| Confirmatory | 24 | 4 | 5 | 1 | 5 | 2,400 |

The engineering dataset is a 2-sequence by 2-turn development fixture with
SHA-256
`14c382a04acbe9394474f05cf84d8389833058afc2dc6feda21a023d46e45ef3`.
The confirmatory dataset is a sealed 24-sequence by 4-turn evaluation fixture
with SHA-256
`7cf2d3a9fa35735aadc9186438277d2b5f6b7beb9f96e9fc9bbeb400da2b5d72`.

Smoke and pilot runs validate engineering behavior and development choices;
neither may make a scientific claim. Only a valid frozen confirmatory run may
emit exactly one final state: `VALIDATED_POSITIVE`, `VALIDATED_NEGATIVE`,
`INCONCLUSIVE`, or `INVALID_RUN`. Its report must keep seven distinct sections:
engineering validity, controller activity, end-to-end efficacy,
persistent-state attribution, guardrail outcomes, limitations, and final
decision.

## Five phase boundaries

The implementation has exactly five major phases. Phase names guide work and
release gates; they are not runtime architecture.

| Phase | Authorized outcome | Not yet claimed at that gate |
| --- | --- | --- |
| 1. Clean foundation and contracts | Typed domain and protocol surfaces, canonical identities, fake provider, minimal CLI, zero-network tests, architecture documents | SQLite execution, resume, llama.cpp transport, scientific policies or results |
| 2. Experiment kernel, storage, metrics, and llama.cpp | Complete provider-to-compact-artifact path, deterministic validators, strict llama.cpp, transaction/resume behavior | Comparator efficacy or neural claims |
| 3. Baselines and statistical evaluator | Typed static, random, and heuristic comparators; development-only selection; sealed-data identity; exact matching; paired evaluator, guardrails, and durable decision skeleton | Live-provider validity, neural activity or benefit, persistent-state attribution, or a final scientific decision |
| 4. Simulated neural controller and attribution | One transparent persistent neural policy, causally clean matched-focal-history substrate reset, and fake-provider mechanism proof | Live/model-backed efficacy, beneficial persistent-state attribution, or a final scientific decision |
| 5. Model-backed evaluation and closeout | Smoke, development pilot, frozen confirmatory run, and one declared final decision | Opportunistic retuning after confirmatory execution |

Work advances in order only after the current phase gate passes. No additional
numbered phase, compatibility layer, or out-of-scope feature may be inserted to
bypass a gate.

## Initial non-goals and hard restrictions

The initial rebuild excludes Ollama compatibility, legacy Phase 1–6 APIs and
mode strings, token- or chunk-level intervention, reinforcement learning,
automated gain or Bayesian optimization, live CL1 integration, bundled NEURON
source trees, automatic model downloads, automatic provider fallback, silent
generation retries, and direct controller control of generation length.

Default tests perform no live HTTP or model calls. The live llama.cpp test
requires the explicit `live` marker and a complete JSON payload; CLI model
execution separately requires successful explicit preflight followed by both
`run --execute` and `--allow-live-provider`. Mocks can establish contract
behavior but never live-provider validity. If a machine-local llama.cpp runtime
is unavailable after all code and preflight work is complete, the truthful state
is `READY_FOR_LIVE_SMOKE`, not a fabricated live result.

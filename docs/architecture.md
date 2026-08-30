# NeuraLLM architecture

## Status and purpose

This living contract describes the NeuraLLM 2.0 architecture through the Phase
2 experiment kernel. The current implementation includes strict configuration
and datasets, deterministic planning, bounded action application, a fixed kernel
policy, deterministic response metrics, fake and llama.cpp provider adapters,
transactional SQLite execution and resume, compact exports, and the explicit
CLI. Their existence is not evidence of live llama.cpp validity, comparator
fairness, neural activity, or scientific efficacy.

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
| `control` | Shared policy protocol, action bounds, baselines, and the composed neural controller | Provider construction, experiment scheduling, evaluation decisions |
| `metrics` | Deterministic validators and versioned output metrics | Controller state transitions or policy feedback routing |
| `experiments` | Deterministic planning, matching, scheduling, execution, and resume orchestration | Policy-specific mode dispatch or statistical claims |
| `evaluation` | Sequence-level scoring, guardrails, paired statistics, and scientific decisions | Generation or mutation of source run evidence |
| `storage` | Transactional run persistence, manifests, integrity checks, and crash-safe resume | Scientific policy or retry decisions |
| `reporting` | Compact, reproducible views derived from the canonical run store | Recomputing or changing scientific truth |
| `cli` | Explicit composition root and command surface | Import-time clients, hidden defaults, or alternate scientific paths |

Dependencies point toward typed domain contracts. The runner composes policies,
providers, metrics, and storage; those components do not reach back into the CLI
or choose one another through mode strings. Production package paths and runtime
identifiers must not contain chronological phase names.

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

The runner may select and construct a policy, but it must not branch on policy
mode strings during execution. Static, bounded-random, heuristic-adaptive,
persistent-neural, and matched-history reset arms all use this interface and the
same `ControllerAction` type. Policy state is explicit and passed through the
interface; global mutable controller state is prohibited.

The neural policy is composed from separable parts:

```text
ControllerObservation
    -> ObservationEncoder
    -> NeuralStimulus
    -> NeuralSubstrate
    -> NeuralReadout
    -> ActionDecoder
    -> ControllerAction
```

The initial substrate will be a small deterministic dynamical system with
interpretable, bounded state and explicit equations. It is not a learned opaque
network. This separation permits a future substrate adapter without changing
the experiment runner or evaluation system, but live CL1 integration is outside
the initial rebuild.

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
For identical canonical inputs it must return identical outputs and it must make
no network call. This gives domain, policy, identifier, and CLI contracts a
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

## Domain and identity contracts

Core boundaries use strict immutable models rather than untyped dictionaries.

| Model | Contract |
| --- | --- |
| `DecodingParameters` | `temperature`, `top_p`, `top_k`, `presence_penalty`, fixed `max_tokens`, and `seed` |
| `ControllerAction` | Bounded deltas for the four controllable parameters; it cannot change `max_tokens` |
| `ControllerObservation` | Only the current turn index, prompt family, current prompt features, nullable prior-response metrics, and an explicit history-presence flag |
| `ResponseMetrics` | Output measures whose entries carry value, availability, metric version, and input hash |
| `ExperimentCondition` | Unique binding of experiment, dataset, prompt sequence, turn, policy, model seed, controller seed, provider identity, and base decoding profile |
| `RunManifest` | Source, configuration, data, provider, policy, metric, seed, bounds, decision-rule, and database-schema identities |

Canonical scientific serialization is UTF-8 JSON with sorted keys, compact
separators, and non-finite values rejected (`allow_nan = false`). Hashes are
lowercase SHA-256. Deterministic condition identifiers are derived from the
complete condition identity, not filesystem paths, timestamps, process order,
or incidental machine state.

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
indicators remain distinct trace fields. Output behavior is the primary evidence;
controller and neural trajectories are diagnostic mechanism evidence.

The Phase 2 kernel implements this loop for the explicitly configured
`kernel_fixed` policy with the fake or strict llama.cpp provider. It stores the
applied action stages, response, metrics, state, and history commitments in one
transactional SQLite run store. Fake-provider execution establishes the offline
provider-to-artifact engineering path; it must not be described as live-provider
validation or policy efficacy.

## Efficacy and attribution are different experiments

End-to-end efficacy gives each policy its own causal trajectory. Each policy
generates its own response, observes metrics only from its own previous response,
and carries only its own declared state. The efficacy policies are
`best_static`, `random_matched`, `heuristic_adaptive`, and `neural_persistent`.

Persistent-state attribution instead pairs `neural_persistent` with
`neural_matched_history_state_reset`. The reset arm receives the exact committed
previous-turn metric tuple from the focal persistent arm, preserves the real
turn index and history-presence semantics, and resets substrate/controller state
at the intervention boundary. It never inserts its own response metrics into the
focal history. The reset arm is attribution-only and must not be reported as an
independently operating efficacy baseline.

These paths share current prompt, model seed, action decoder, provider, base
parameters, and action bounds. At turn zero they must be byte-equivalent before
persistent state can meaningfully differ. Later tests must prove that only the
declared persistent state differs and that no comparator-history leakage exists.

## Storage and artifact boundary

Phase 2 uses one SQLite database as the canonical mutable run record. Migrations,
integrity verification, transactional checkpoints, history commitments, and
crash-safe resumption are implemented without an alternate ad hoc JSON store.
Unique constraints prevent duplicate logical requests. A request moves
through prepared, dispatching, response-persisted, metrics-computed, and
committed work within explicit transactions. Committed turns are never
regenerated. A dispatched request with uncertain outcome fails closed unless the
provider offers a verified idempotency mechanism; it is never silently retried.

Closed runs export only:

```text
run.sqlite3
manifest.json
results.csv
comparisons.csv
decision.json
report.md
```

Optional plots share one `plots/` directory. Per-turn directory forests are
prohibited. `scientific_identity_sha256` covers canonical scientific inputs and
excludes incidental timestamps and machine-local output paths.

## Evaluation boundary

The primary endpoint is preregistered, output-based
`guardrail_clean_task_score`, aggregated at the prompt-sequence by model-seed
unit rather than treating turns as independent samples. Guardrails gate the
result; they are not hidden inside a weighted score. Internal controller drift
cannot establish efficacy.

Every confirmatory run terminates in exactly one state:

- `VALIDATED_POSITIVE`
- `VALIDATED_NEGATIVE`
- `INCONCLUSIVE`
- `INVALID_RUN`

A clean negative is preserved. An invalid run is not relabeled inconclusive,
and uncertainty is not relabeled negative. Decision execution and statistical
testing are later-phase responsibilities. Phase 2 exports a null scientific
decision with an engineering-only claim scope rather than simulating a later
phase result.

## Five phase boundaries

The implementation has exactly five major phases. Phase names guide work and
release gates; they are not runtime architecture.

| Phase | Authorized outcome | Not yet claimed at that gate |
| --- | --- | --- |
| 1. Clean foundation and contracts | Typed domain and protocol surfaces, canonical identities, fake provider, minimal CLI, zero-network tests, architecture documents | SQLite execution, resume, llama.cpp transport, scientific policies or results |
| 2. Experiment kernel, storage, metrics, and llama.cpp | Complete provider-to-compact-artifact path, deterministic validators, strict llama.cpp, transaction/resume behavior | Comparator efficacy or neural claims |
| 3. Baselines and statistical evaluator | Serious static, random, and heuristic comparators; sealed-data discipline; paired evaluator and decision skeleton | Neural activity or benefit |
| 4. Simulated neural controller and attribution | One transparent persistent neural policy and causally clean matched-history reset control | Model-backed scientific conclusion |
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
execution separately requires `run --execute`. Mocks can establish contract
behavior but never live-provider validity. If a machine-local llama.cpp runtime
is unavailable after all code and preflight work is complete, the truthful state
is `READY_FOR_LIVE_SMOKE`, not a fabricated live result.

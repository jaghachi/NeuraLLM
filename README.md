# NeuraLLM

NeuraLLM is a greenfield research system built to answer one falsifiable
question:

> Can a simulated neural controller regulate LLM decoding more effectively than
> a strong fixed sampler, bounded random perturbation, and a competent
> non-neural adaptive controller under matched model, prompt, seed, and
> generation-budget conditions?

A clean negative result counts as a successful experiment. Engineering
completion never depends on producing a positive scientific result.

## Current status

The source tree reports package version `2.0.0b1` and implements the Phase 4
engineering boundary: the Phase 3 baselines and paired statistical evaluator,
plus one transparent five-state simulated neural controller and its
matched-focal-history substrate-reset attribution arm. Phase 2's
provider-to-artifact kernel remains supported, including strict llama.cpp
contract behavior and the deterministic fake provider.

Phase 4 proves controller activity and causal isolation with the deterministic
fake provider. It does not establish live-provider validity, neural-controller
benefit, a model-backed result, or a final persistent-state effect on model
quality. The matched-reset arm is never admitted to the Phase 3 efficacy
evaluator, and `scientific_decision` remains `null`.

## Implemented Phase 3 boundary

Every Phase 3 arm is constructed from a strict typed specification and uses the
same control-policy interface and action bounds:

- `best_static` applies no deltas and preserves the decoding profile selected
  and frozen from development-only evidence.
- `random_matched` produces deterministic SHA-256-derived perturbations within
  the shared bounds. Its turn-zero action is exactly zero and it has no response
  history access.
- `heuristic_adaptive` uses only its own previous committed response metrics,
  reacts through declared thresholds, and decays its prior action toward zero
  when the previous response is clean. Its turn-zero action is exactly zero.

Static selection accepts only a `development` dataset, binds every candidate
score vector to the same canonical `prompt sequence x model seed` key order,
requires one fixed `max_tokens` budget, uses mean sequence-unit task score with
a lexical profile-ID tie break, and records a canonical selection hash. The
selected profile is bound as the Phase 3 base profile before evaluation.

Before execution, the run manifest freezes a Phase 3 analysis-contract digest
covering the plan, evaluator, static selection, evaluation design, dataset
purpose/hash, and seal. Analysis persistence and reads recompute that digest and
reject foreign provenance.

Evaluation and synthetic datasets have explicit purposes and canonical content
hashes. An `evaluation` dataset additionally requires a matching external seal.
The checked-in evaluation fixture is bound to
`datasets/evaluation/phase3-baseline-evaluation-v1.yaml` with dataset SHA-256
`de4c415d71cc3ed0177b189880fa9da040464f41ab14b192bab01cb4eed09199` and seal
SHA-256 `89e794e8c80094c15ba9be801306f9ca8090fbd45ab9462b92c172a4a3b65847`.
That identity is frozen evidence for the offline fixture, not evidence that a
confirmatory live-model evaluation has run.

The primary statistical unit is `prompt sequence x model seed`. Turns are
averaged within each controller seed, then controller-seed means are averaged
within the unit; neither turns nor controller seeds inflate the paired sample
size. Exact policy, turn, model-seed, and controller-seed coverage is checked
before statistics. Valid evidence is evaluated with seeded paired bootstrap
confidence intervals, paired sign-flip permutation tests, and Holm correction
over serious comparators. Explicit guardrails cover coverage and dataset
identity, provider identity, turn-zero history semantics, action bounds,
required metric availability, adherence non-regression, response-length
confounding, focal action saturation, and behavioral aliasing.

## Implemented Phase 4 causal boundary

Both neural arms use the same `ObservationEncoder`, deterministic
`NeuralSubstrate`, `ActionDecoder`, shared `ControllerAction`, and run-level
action bounds:

- `neural_persistent` consumes only its own previous committed response metrics
  and carries a five-variable bounded substrate.
- `neural_matched_history_state_reset` is attribution-only. At turn `t > 0`, it
  binds the exact `neural_persistent[t-1]` condition, commitment, metrics, and
  controller-state envelope, then resets only the declared substrate before
  applying the same transition equations and decoder.

The run manifest records the sole authorized cross-policy history edge. SQLite
rejects any other source policy or mismatch in experiment, dataset, sequence,
turn, model seed, controller seed, provider identity, or base profile. The
neural trace records encoding, stored and effective substrate states, transition
equation and saturation evidence, four decoder activations, normalized action
magnitude, reset status, focal condition/commitment/metrics hashes, and the
existing step/legal action-clamp evidence.

The checked-in `phase4-neural-causal-smoke.yaml` harness is turn-interleaved and
provider-free until explicit execution. Tests prove identical turn-zero
provider-visible inputs and fake responses, same focal metrics at later turns,
reset-only state intervention, later decoding/response divergence, bounded
actions, deterministic serialization, and zero provider calls on replay. This
is mechanism evidence, not efficacy evidence.

## Install

Python 3.12 is required. The intended local environment is:

```powershell
conda create -n neurallm2 python=3.12 -y
conda activate neurallm2
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
```

For a runtime-only editable install:

```powershell
python -m pip install -e .
```

## Current command surface

The command surface is machine-readable and supports the historical Phase 2
smoke configuration, the Phase 3 offline fixtures, and the Phase 4 causal
harness:

```powershell
neurallm status
neurallm validate --config configs/experiments/phase3-baseline-evaluation.yaml
neurallm plan --config configs/experiments/phase3-baseline-evaluation.yaml
neurallm run --config configs/experiments/phase3-baseline-evaluation.yaml --dry-run
neurallm run --config configs/experiments/phase3-synthetic-evaluator.yaml --execute
neurallm run --config configs/experiments/phase4-neural-causal-smoke.yaml --dry-run
neurallm analyze --run-dir runs/phase3-synthetic-evaluator-validation
neurallm report --run-dir runs/phase3-synthetic-evaluator-validation
```

`validate`, `plan`, and `run --dry-run` validate the complete schedule,
development-selection evidence, dataset identity, seal when required, policy
specifications, and evaluator identity without constructing a generation
provider or requesting network access. `run --execute` is the explicit provider
boundary. For a Phase 3 configuration it executes or safely resumes the run,
derives evaluator records from the closed SQLite evidence, finalizes the
analysis, and exports the compact views. The checked-in Phase 3 configurations
select only the deterministic fake provider.

`analyze` and `report` re-verify the canonical store and reproduce the derived
views. After argparse accepts a command, successful application output is
canonical JSON on stdout; runtime or validation failures are canonical JSON on
stderr with exit code 2. `--help`, `--version`, and argparse usage errors retain
argparse's plain-text interface.

Every closed run contains exactly:

```text
run.sqlite3
manifest.json
results.csv
comparisons.csv
decision.json
report.md
```

`run.sqlite3` is canonical. The other five files are deterministic derived
views. For a Phase 2 run, `comparisons.csv` remains header-only and
`decision.json` retains the historical `engineering_validation_only` scope. A
finalized Phase 3 analysis populates comparison rows when comparisons are
available and emits the Phase 3 verdict and analysis hashes in `decision.json`
and `report.md`; it still emits `scientific_decision: null`.

## Validate the implementation

```powershell
ruff check .
ruff format --check .
mypy src
python -m pytest -q
```

Default tests are network-blocked and exclude the `live` marker. A provider-free
scenario harness consumes the checked-in synthetic fixture's known-outcome
codes in independent evaluator runs covering superior, inferior,
equivalent/aliased, and length-confound behavior; pure tests also cover invalid
coverage without a model call.

## Scientific claim boundary

Phase 4 adds deterministic neural mechanism activity and a hash-bound,
matched-focal-history reset isolation test to the Phase 3 comparator and
evaluator evidence. It does not establish neural efficacy, a beneficial
persistent-state effect on model output, live llama.cpp validity, or a final
scientific outcome. The Phase 3 verdict vocabulary
(`superior`, `inferior`, `equivalent`, `inconclusive`, `invalid`) validates the
baseline evaluator under its fixture protocol; it is not any Phase 5 decision
state.

See [metric definitions](docs/metrics.md), the
[llama.cpp provider runbook](docs/provider-runbook.md), the
[architecture](docs/architecture.md), and the
[experiment contract](docs/experiment-contract.md) for the exact boundaries.

The historical `jaghachi/neurollm` repository is a frozen reference, not the
implementation base. See [legacy lessons](docs/legacy-lessons.md) for its exact
commit, archive tag, and component dispositions.

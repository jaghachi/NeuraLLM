# NeuraLLM

NeuraLLM is a greenfield research system built to answer one falsifiable
question:

> Can a simulated neural controller regulate LLM decoding more effectively than
> a strong fixed sampler, bounded random perturbation, and a competent
> non-neural adaptive controller under matched model, prompt, seed, and
> generation-budget conditions?

A clean negative result counts as a successful experiment. Engineering
completion never depends on producing a positive scientific result.

## How the system is designed

Every experimental arm implements one typed control-policy interface and acts
only on causally available observations. Controllers can adjust temperature,
top-p, top-k, and presence penalty within shared bounds; the generation length
budget remains fixed. A deterministic fake provider supports complete offline
testing. The implemented llama.cpp adapter is isolated behind an explicit,
identity-bound, fail-closed provider boundary.

End-to-end efficacy uses independent policy histories. Persistent-state causal
attribution is a separate matched-history experiment and is never reported as
an independent efficacy baseline.

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

Phase 2 exposes a machine-readable command surface:

```powershell
neurallm status
neurallm validate --config configs/experiments/smoke.yaml
neurallm plan --config configs/experiments/smoke.yaml
neurallm run --config configs/experiments/smoke.yaml --dry-run
neurallm run --config configs/experiments/smoke.yaml --execute
neurallm analyze --run-dir runs/phase2-fake-smoke
neurallm report --run-dir runs/phase2-fake-smoke
```

`validate`, `plan`, and `run --dry-run` validate the complete schedule and
identities without constructing a generation provider or requesting network
access. `run --execute` is the explicit execution boundary: it constructs only
the configured provider, safely resumes the SQLite run, and exports the closed
run. The included smoke configuration selects the deterministic fake provider.

`analyze` verifies a closed `run.sqlite3` and deterministically derives the
compact artifact set. `report` re-verifies the same canonical store and
reproduces its derived views, including `report.md`. After argparse accepts a
command, successful application output is canonical JSON on stdout, and
runtime or validation failures are canonical JSON on stderr with exit code 2.
`--help`, `--version`, and argparse usage errors retain argparse's plain-text
interface.

A closed Phase 2 run contains exactly:

```text
run.sqlite3
manifest.json
results.csv
comparisons.csv
decision.json
report.md
```

`run.sqlite3` is canonical. The other five files are reproducible derived
views. `comparisons.csv` is intentionally empty and `decision.json` has a null
scientific decision because comparator evaluation begins in Phase 3.

## Validate the foundation

```powershell
ruff check .
ruff format --check .
mypy src
python -m pytest -q
```

Default tests are network-blocked and exclude the `live` marker.

## Scientific status

Phase 2 establishes an engineering path from explicit configuration through a
provider, transactional storage, deterministic response metrics, and compact
artifacts. It does not estimate policy efficacy, establish comparator advantage,
demonstrate neural activity, or select a scientific outcome. No confirmatory
experiment has run, and there is no live llama.cpp validation claim. A future
confirmatory run must terminate as exactly one of `VALIDATED_POSITIVE`,
`VALIDATED_NEGATIVE`, `INCONCLUSIVE`, or `INVALID_RUN`.

See [metric definitions](docs/metrics.md), the
[llama.cpp provider runbook](docs/provider-runbook.md), the
[architecture](docs/architecture.md), and the
[experiment contract](docs/experiment-contract.md) for the exact boundaries.

The historical `jaghachi/neurollm` repository is a frozen reference, not the
implementation base. See [legacy lessons](docs/legacy-lessons.md) for its exact
commit, archive tag, and component dispositions.

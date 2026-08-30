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
testing, while llama.cpp will be isolated behind an explicit fail-closed
provider boundary.

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

Phase 1 exposes a provider-free status command:

```powershell
neurallm status
```

The experiment `validate`, `plan`, `run --dry-run`, `run --execute`, `analyze`,
and `report` commands are Phase 2 deliverables and are not claimed as available
yet.

## Validate the foundation

```powershell
ruff check .
ruff format --check .
mypy src
python -m pytest -q
```

Default tests are network-blocked and exclude the `live` marker.

## Scientific status

No confirmatory experiment has run. There is currently no scientific decision
and no live-provider validation claim. A future confirmatory run must terminate
as exactly one of `VALIDATED_POSITIVE`, `VALIDATED_NEGATIVE`, `INCONCLUSIVE`, or
`INVALID_RUN`.

The historical `jaghachi/neurollm` repository is a frozen reference, not the
implementation base. See [legacy lessons](docs/legacy-lessons.md) for its exact
commit, archive tag, and component dispositions.

# NeuraLLM contributor contract

Work in the five major phases defined by the project contract and keep production
modules organized by domain, never by phase number.

Before advancing a phase, run:

```powershell
ruff check .
ruff format --check .
mypy src
python -m pytest -q
```

Non-negotiable rules:

- Treat `jaghachi/neurollm` as a frozen read-only reference; never import or copy
  its runtime wholesale.
- Use strict immutable models at domain boundaries and one shared controller
  interface for every policy.
- Keep `max_tokens` fixed by the experiment plan; controllers may change only
  temperature, top-p, top-k, and presence penalty within declared bounds.
- Turn zero has null prior metrics and `has_previous_response = false`.
- Keep independent-history efficacy separate from matched-history attribution.
- No automatic provider fallback, hidden environment fallback, or silent retry
  after dispatch.
- Default tests and dry runs make zero live model calls. Never claim live
  provider validity from mocks.
- Preserve valid negative and inconclusive results; do not tune on sealed
  evaluation data.
- Record exact commands, results, limitations, and the independent review at
  each major phase closeout.

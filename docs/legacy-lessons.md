# Legacy lessons and disposition

NeuraLLM 2.0 is a greenfield implementation. The historical repository is a
scientific and design reference, not a source tree to refactor or copy.

## Frozen reference

- Repository: `jaghachi/neurollm`
- Default branch: `main`
- Reference commit: `9b983dbe3781c1eef715d895c3bb3f68bfcccebb`
- Reference subject: `SCRUM-40: Reconcile SCRUM-40 candidate-v2 submission readiness`
- Remote archive tag: `v1-research-archive-2026-08-26`

The tag was created and verified on the GitHub remote at the exact commit above.
A pre-existing local checkout may not list it until tags are fetched. The remote
tag and commit identify the legacy population. Existing code, datasets, reports,
and evidence remain in that repository and must not be regenerated or presented
as NeuraLLM 2.0 evidence.

## Classification rules

`PRESERVE_CONCEPT` means retain a scientific or integrity principle while
designing a new implementation. `REIMPLEMENT` means write new domain-based code
and independently test it. `OPTIONAL_REFERENCE` means inspect only when a
specific Phase 2-5 need is approved. `RETIRE` means the component is outside the
2.0 architecture and must not be copied.

The default disposition for legacy runtime code is `RETIRE`.

| Candidate legacy component | Original path at the frozen commit | Classification | Scientific use and independent validation required in 2.0 |
| --- | --- | --- | --- |
| Fail-closed provider identity | `src/neurollm/diagnostics/provider_identity.py`; `src/neurollm/phase6/phase6_6d_llama_cpp_protocol.py` | `PRESERVE_CONCEPT` | Preserve explicit identity binding and drift rejection. New contract tests must cover alias, model path, build, template, configuration, and mid-run drift. |
| llama.cpp transport | `src/neurollm/llm/llama_cpp_provider.py` | `REIMPLEMENT` | A strict provider remains useful, but 2.0 needs a new typed interface, explicit configuration, no fallback, no automatic retry, and mock-transport tests for every failure mode. |
| Canonical manifests and evidence hashes | `src/neurollm/phase6/gain_profile_manifest.py`; `src/neurollm/phase6/phase6_6d_llama_cpp_evidence.py` | `PRESERVE_CONCEPT` | Preserve canonical, fail-closed scientific identity. New serialization tests must prove UTF-8, sorted compact JSON, finite values, lowercase SHA-256, and timestamp/path exclusion where required. |
| Objective scoring and deterministic validators | `src/neurollm/phase6/objective_scoring.py` | `REIMPLEMENT` | Objective validation remains scientifically useful. New unit and property tests must validate each prompt contract without importing or calling the legacy implementation. |
| Repetition, diversity, and behavioral metrics | `src/neurollm/core/metrics.py`; `src/neurollm/phase5/behavioral_metrics.py`; `src/neurollm/phase3/metrics.py` | `REIMPLEMENT` | Recreate only preregistered formulas with explicit metric versions and input hashes. Independently test edge cases, length confounds, determinism, and non-finite rejection. |
| Matched-history state-reset idea | `src/neurollm/phase6/matched_history_state_reset.py` | `PRESERVE_CONCEPT` | Preserve the causal distinction between focal committed history and reset state. New tests must prove turn-zero equivalence, focal-history identity, no comparator-history leakage, and correct intervention boundaries. |
| Gain-profile mappings and accepted formulas | `src/neurollm/phase6/gain_profile_mapping.py`; `src/neurollm/phase6/gain_profile_candidate_set.py` | `OPTIONAL_REFERENCE` | No mapping is automatically carried forward. A future port must name the exact formula, justify its relevance to the four-parameter action space, and add independent boundedness and determinism tests. |
| Legacy prompts and datasets | `benchmarks/phase3/prompts/`; `benchmarks/phase4/prompts/`; `src/neurollm/data/benchmark_prompts.json` | `OPTIONAL_REFERENCE` | A prompt may be adapted only with provenance, licensing review, a new dataset version/hash, and an independently tested objective validator. It must not enter the sealed evaluation set by convenience. |
| Historical reports and generated evidence | `experiments/archive/`; `benchmarks/phase6/`; `docs/phase5_*`; `docs/phase6_*` | `OPTIONAL_REFERENCE` | These records may inform risks and design reviews, but they remain version 1 evidence and cannot qualify 2.0 behavior or provider validity. |
| Phase-numbered runtime architecture | `src/neurollm/phase3/`; `src/neurollm/phase4/`; `src/neurollm/phase5/`; `src/neurollm/phase6/` | `RETIRE` | Production code is organized by domain in 2.0. No phase-numbered package or runtime identifier is retained. |
| Controller monolith and mode-string dispatch | `src/neurollm/core/neural_controller.py`; legacy benchmark runners and mode configurations | `RETIRE` | All policies use one typed protocol and composition. No wholesale port, compatibility dispatch tree, or obsolete mode string is allowed. |
| Ollama runtime and evidence | `src/neurollm/llm/ollama_provider.py`; `configs/llm_provider.ollama_legacy.example.json`; Ollama evidence records | `RETIRE` | Ollama is excluded from the initial rebuild. Historical Ollama artifacts remain intact and are never described as llama.cpp or NeuraLLM 2.0 evidence. |
| Bundled NEURON simulator tree and live CL1 work | `nrn/`; `experiments/archive/phase1/` | `RETIRE` | The first controller is a small deterministic simulated substrate. Bundled simulator binaries, source trees, and live CL1 integration are outside the initial rebuild. |

## Porting gate

Before any legacy formula, dataset, prompt, metric, or mechanism is introduced,
the implementing change must record:

1. the exact path above (or a newly discovered exact path);
2. the frozen commit SHA;
3. the scientific reason to retain it;
4. the new tests that validate the implementation independently; and
5. confirmation that no legacy runtime module is imported or copied wholesale.

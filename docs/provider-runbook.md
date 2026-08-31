# llama.cpp provider runbook

## Status and claim boundary

NeuraLLM Phase 2 implements a strict synchronous adapter for the llama.cpp HTTP
completion server. Fake-transport contract tests cover its success and failure
paths. No live llama.cpp run has been validated for this release; mocks and the
deterministic fake provider cannot establish live validity.

The adapter targets one explicit server contract. It is not a compatibility
layer for every llama.cpp version, Ollama, OpenAI-compatible routes, or another
provider.

## Required server contract

Construction first streams the configured local model artifact and requires its
lowercase SHA-256 to equal `model_sha256`. It then performs these requests:

1. `GET /health`, which must return exactly `{"status":"ok"}`.
2. `GET /props`, which must return a JSON object containing:
   - non-blank `model_path`, `build_info`, and `chat_template` strings;
   - positive integer `total_slots`; and
   - `default_generation_settings.params` with finite-float `temperature`,
     finite-float `top_p`, integer `top_k`, finite-float `presence_penalty`,
     integer `n_predict`, and integer `seed`.

Before every generation, the adapter cheaply compares the local artifact's
device, file identity, size, and modification time with the values bound at
construction. A changed fingerprint triggers a full digest remeasurement. This
metadata check catches ordinary replacements but is not a cryptographic
measurement of every byte at every dispatch. The adapter then repeats `/health`
and `/props` and requires the complete effective configuration and derived
identity to equal the values bound at construction. It then sends exactly one
non-streaming
`POST /completion` with:

```text
prompt
model
temperature
top_p
top_k
presence_penalty
n_predict
seed
stream = false
cache_prompt = false
```

The deterministic seed must be in `0..4294967294`. A successful completion must
contain non-blank `content`, `stop = true`, the bound model alias, and effective
`generation_settings` that reproduce temperature, top-p, top-k, presence
penalty, both `n_predict` and `max_tokens`, and seed. Missing, malformed, or
different settings fail closed.

For an accepted completion, `GenerationMetadata` retains canonical parsed JSON
for the exact provider request and response plus a lowercase SHA-256 for each.
The transactional run store persists that metadata inside the canonical
generation response. This is protocol evidence after JSON parsing, not a claim
to preserve HTTP framing or byte-for-byte wire data.

## Explicit provider configuration

Copy `configs/providers/llama_cpp.example.yaml` to a machine-local path and
replace every placeholder. Do not put credentials in `base_url` or commit local
machine identity accidentally.

| Field | Requirement |
| --- | --- |
| `base_url` | Absolute HTTP(S) URL with no credentials, query, or fragment |
| `model_alias` | Exact alias sent to and returned by `/completion` |
| `model_path` | Absolute, readable client-local regular file path that is returned exactly by `/props` |
| `model_sha256` | Lowercase SHA-256 measured from the complete local model artifact |
| `build_id` | Exact non-blank `build_info` returned by `/props` |
| `chat_template_sha256` | Lowercase SHA-256 of the raw `/props` `chat_template` UTF-8 text |
| `connect_timeout_seconds` | Explicit positive finite connect timeout |
| `read_timeout_seconds` | Explicit positive finite read timeout |
| `write_timeout_seconds` | Explicit positive finite write timeout |
| `pool_timeout_seconds` | Explicit positive finite pool timeout |

The provider reads none of these fields from environment variables. Its HTTP
client uses `trust_env = false` and `follow_redirects = false`; proxy environment
variables and redirect targets cannot silently change the route. There is no
automatic download, provider fallback, or retry transport.

## Deliberate identity preflight

An experiment configuration must contain both the provider-config path and the
exact expected `ProviderIdentity`. Inspect that identity with the first-class
preflight command, using only an explicit machine-local file:

```powershell
neurallm preflight --provider-config configs/providers/llama_cpp.local.yaml
```

This is a deliberate network operation, but it first hashes the local artifact,
then performs exactly one `GET /health` and one `GET /props`; it never requests
`/completion`. The provider configuration is not read from an environment
variable. Success emits one
canonical JSON object containing `expected_identity`, `provider_identity_id`,
`expected_effective_configuration_json`, and `completion_requested: false`.

Copy `expected_identity` and the exact one-line
`expected_effective_configuration_json` value under `provider`, and point
`config_path` to the same provider file, relative to the experiment config:

```yaml
provider:
  kind: llama_cpp
  config_path: ../providers/llama_cpp.local.yaml
  expected_identity:
    provider_type: llama_cpp
    implementation_version: llama-cpp-completion-http-v1
    model_alias: replace-with-preflight-output
    build_id: replace-with-preflight-output
    provider_config_hash: replace-with-64-character-preflight-output
    model_path: replace-with-preflight-output
    model_sha256: replace-with-64-character-preflight-output
    chat_template_sha256: replace-with-preflight-output
  expected_effective_configuration_json: >-
    {"replace":"with the exact single-line canonical preflight output"}
```

The `>-` block scalar preserves the pasted one-line JSON without adding a
trailing newline. Replace its entire example line; do not pretty-print, reorder,
edit, or re-encode the preflight output. Validation requires that exact
canonical string to hash to `expected_identity.provider_config_hash`. The
top-level `provider_identity_id` is verification output and is not a separate
experiment-config field.

`provider_config_hash` binds the explicit client configuration and the inspected
effective server configuration, including `model_sha256`; `ProviderIdentity`
also records that artifact digest directly. If the model bytes, server,
timeouts, defaults, slots, template, path, build, or alias changes, perform a
new preflight and treat it as a new provider identity. Never edit the expected
identity merely to bypass a drift failure.

The digest establishes the artifact readable at the client-side `model_path`.
NeuraLLM also requires `/props` to return that exact path, but llama.cpp does not
attest over HTTP that its already-loaded weights came from those bytes. Therefore
claim-eligible use requires a same-host path or a trusted shared mount and an
operator-controlled server whose path-to-loaded-artifact relationship is trusted.
A remote-only or container-internal path that the client cannot read fails
closed; there is no remote attestation or download fallback.

## Provider-free commands and explicit execution

These commands validate declared inputs and identities without constructing the
provider or making HTTP requests:

```powershell
neurallm validate --config path/to/experiment.yaml
neurallm plan --config path/to/experiment.yaml
neurallm run --config path/to/experiment.yaml --dry-run
```

Only this command crosses the llama.cpp provider-construction and generation
boundary:

```powershell
neurallm run --config path/to/experiment.yaml --execute --allow-live-provider
```

For `kind: llama_cpp`, both `--execute` and `--allow-live-provider` are required.
Omitting the second flag fails before provider construction or HTTP. Fake-provider
execution remains available with `--execute` alone. Authorized execution
constructs exactly the selected provider and requires its inspected identity to
equal `expected_identity` before the run manifest is bound. Each pending logical
turn gets at most one `/completion` dispatch. If transport fails after dispatch
begins, the SQLite turn becomes uncertain and resume fails closed instead of
silently generating again.

For confirmatory execution, the construction hash and the full hash immediately
before scientific persistence/export are point-in-time measurements. A
replacement between preflight and execution fails during construction, and an
ordinary mid-run replacement changes the per-dispatch metadata fingerprint and
triggers a full rehash. These checks are not continuous remote attestation: a
transient same-size rewrite whose file identity and modification time are
restored between checks is outside what the fingerprint can prove, and the final
hash cannot retrospectively identify it. Claim-eligible use therefore assumes a
trusted same-host operator (or equivalently trusted shared mount) who excludes
that rewrite-and-restore behavior.

## Explicit live test

The live test is excluded by default. It requires both the `live` marker and the
complete `NEURALLM_LIVE_LLAMA_CONFIG_JSON` payload. That variable belongs only
to the opt-in test harness; the provider itself does not use it as fallback
configuration.

```powershell
$livePayload = @{
    provider = @{
        base_url = "http://127.0.0.1:8080"
        model_alias = "replace-with-explicit-alias"
        model_path = "C:/models/replace-with-model.gguf"
        model_sha256 = "replace-with-64-character-model-hash"
        build_id = "replace-with-build-id"
        chat_template_sha256 = "replace-with-64-character-hash"
        connect_timeout_seconds = 5.0
        read_timeout_seconds = 120.0
        write_timeout_seconds = 10.0
        pool_timeout_seconds = 5.0
    }
    prompt = "Reply with the word ready."
    decoding_parameters = @{
        temperature = 0.7
        top_p = 0.9
        top_k = 40
        presence_penalty = 0.0
        max_tokens = 16
        seed = 11
    }
    experiment_id = "explicit-live-smoke"
    dataset_version = "local-smoke-v1"
    prompt_sequence_id = "live-sequence-1"
    policy_id = "kernel_fixed"
    controller_seed = 21
    base_decoding_profile_id = "live-base-v1"
} | ConvertTo-Json -Depth 8 -Compress

$env:NEURALLM_LIVE_LLAMA_CONFIG_JSON = $livePayload
python -m pytest -q -m live tests/live/test_llama_cpp_live.py
Remove-Item Env:NEURALLM_LIVE_LLAMA_CONFIG_JSON
```

With no payload, the explicitly selected live test skips before HTTP. A passing
fake-transport suite, provider-free command, or fake-provider run is not a
substitute for recording a passing live command against the exact bound server.

## Failure semantics

| Failure | Result |
| --- | --- |
| Invalid explicit config or prohibited URL shape | Validation fails before HTTP |
| llama.cpp `run --execute` without `--allow-live-provider` | Authorization fails before provider construction or HTTP |
| Timeout, connection error, or non-200 response | `LlamaCppTransportError`; no retry |
| Invalid JSON, response shape, types, or effective settings | `LlamaCppProtocolError` |
| Changed model path, build, template, defaults, slots, alias, or effective identity | `LlamaCppIdentityDriftError` |
| Request bound to another provider identity | `ProviderIdentityMismatchError` |
| Dispatched turn without a valid persisted response | Uncertain dispatch; automatic resume is refused |

Do not convert these failures into fake-provider execution, loosen identity
checks, or describe contract-test success as live validation.

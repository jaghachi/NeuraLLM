"""Five-arm v2 smoke with in-memory HTTP transport; never live evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx

from neurallm.experiments import (
    GitProvenance,
    build_plan,
    build_policy_runtimes,
    build_run_manifest,
    execute_plan,
    load_dataset,
    load_experiment_config,
)
from neurallm.experiments.config import ExperimentConfig, ProviderSelection
from neurallm.metrics import FINAL_ANSWER_METRIC_VERSIONS
from neurallm.providers import LlamaCppProvider
from neurallm.storage import SQLiteRunStore, TurnState
from tests.contract.test_llama_cpp_chat_template import _chat_config, _ChatHandler


def test_twenty_limit_responses_commit_with_answer_scores_and_zero_call_replay(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    loaded = load_experiment_config(
        root / "configs/experiments/model-backed-engineering-smoke.yaml"
    )
    dataset = load_dataset(loaded.dataset_path, expected_version=loaded.config.dataset.version)
    # Synthetic fixture only: every keyword is inside an unfinished reasoning block.
    content = "<think>burst lock expiry jitter capacity monitoring"
    handler = _ChatHandler(stop_type="limit", content=content)
    with LlamaCppProvider(_chat_config(), transport=httpx.MockTransport(handler)) as provider:
        selection = ProviderSelection(
            kind="llama_cpp",
            expected_identity=provider.provider_identity,
            expected_effective_configuration_json=provider.effective_configuration_json,
            config_path="unused-mock-provider.local.yaml",
        )
        config = ExperimentConfig.model_validate(
            {
                **loaded.config.model_dump(),
                "provider": selection,
                "metric_versions": FINAL_ANSWER_METRIC_VERSIONS,
            }
        )
        plan = build_plan(replace(loaded, config=config), dataset)
        assert config.policy_specs is not None
        runtimes = build_policy_runtimes(plan, config.policy_specs)
        manifest = build_run_manifest(
            plan,
            provider.provider_identity,
            runtimes,
            GitProvenance(source_commit="0" * 40, working_tree_clean=True),
        )
        database_path = tmp_path / "mock-smoke.sqlite3"
        result = execute_plan(plan, manifest, provider, runtimes, database_path)
        assert result.planned_turns == result.committed_turns == 20
        assert (
            result.dispatched_this_invocation == result.successful_responses_this_invocation == 20
        )
        assert result.uncertain_dispatches_this_invocation == 0
        calls_before_replay = len(handler.requests)
        replay = execute_plan(
            plan,
            manifest,
            provider,
            build_policy_runtimes(plan, config.policy_specs),
            database_path,
        )
        assert replay.provider_calls == replay.dispatched_this_invocation == 0
        assert len(handler.requests) == calls_before_replay
    assert len(handler.template_payloads) == 21  # One probe plus twenty actual prompts.
    assert len(handler.completion_payloads) == 20
    with SQLiteRunStore(database_path) as store:
        for turn in store.list_turns():
            assert turn.state is TurnState.COMMITTED
            assert turn.response is not None and turn.response.text == content
            assert turn.response.raw_metadata.generation_method == "llama_cpp_chat_template_http_v2"
            assert turn.metrics is not None
            assert turn.metrics.task_score.value == 0.0
            assert turn.metrics.response_length_tokens.value == 0
            assert turn.metrics.task_score.metric_version == "validator-final-answer-v2"

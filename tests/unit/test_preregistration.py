"""Provider-free confirmatory preregistration publication tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from neurallm.cli import main
from neurallm.control.specs import (
    BestStaticPolicySpec,
    HeuristicAdaptivePolicySpec,
    NeuralMatchedHistoryStateResetPolicySpec,
    NeuralPersistentPolicySpec,
    RandomMatchedPolicySpec,
)
from neurallm.domain.models import (
    ActionBounds,
    DecodingBounds,
    PromptFeatures,
    ProviderIdentity,
)
from neurallm.domain.serialization import canonical_json, canonical_json_bytes, canonical_sha256
from neurallm.evaluation.attribution import AttributionAnalysisSpec
from neurallm.evaluation.confirmatory import ConfirmatoryAnalysisSpec, RecoveryEventSpec
from neurallm.evaluation.models import DatasetPurpose, EvaluationSpec, MatchedUnitKey
from neurallm.evaluation.recovery import RecoveryAnalysisSpec, RecoveryMetricName
from neurallm.evaluation.scientific import EfficacyAnalysisSpec, LimitationDisposition
from neurallm.evaluation.selection import (
    StaticCandidateResult,
    StaticProfile,
    select_best_static,
)
from neurallm.experiments.config import (
    BaseDecodingProfile,
    DatasetReference,
    DevelopmentSelectionInput,
    ExperimentConfig,
    ProviderSelection,
)
from neurallm.experiments.dataset import (
    DatasetSeal,
    PromptCase,
    PromptDataset,
    PromptSequence,
)
from neurallm.experiments.preregistration import (
    load_preregistration_seal,
    publish_preregistration,
)
from neurallm.experiments.protocol import (
    ExperimentProtocol,
    RunTier,
    ScheduleSpec,
)
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.fake import (
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION


def _dataset(purpose: DatasetPurpose) -> PromptDataset:
    return PromptDataset(
        schema_version=1,
        dataset_id=f"preregistration-{purpose.value}",
        version=f"preregistration-{purpose.value}-v1",
        purpose=purpose,
        sequences=(
            PromptSequence(
                sequence_id="sequence-1",
                cases=tuple(
                    PromptCase(
                        case_id=f"case-{turn_index}",
                        prompt_family="preregistration_test",
                        prompt=f"Return test response {turn_index}.",
                        prompt_features=PromptFeatures({"constraint_count": 1.0}),
                        validator=ValidatorSpec(kind="non_empty"),
                    )
                    for turn_index in range(2)
                ),
            ),
        ),
    )


def _confirmatory_config(dataset: PromptDataset) -> ExperimentConfig:
    development = _dataset(DatasetPurpose.DEVELOPMENT)
    selection = select_best_static(
        (
            StaticCandidateResult(
                profile=StaticProfile(
                    profile_id="selected-static-v1",
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    presence_penalty=0.0,
                    max_tokens=64,
                ),
                unit_scores=(0.8,),
            ),
            StaticCandidateResult(
                profile=StaticProfile(
                    profile_id="unselected-static-v1",
                    temperature=0.5,
                    top_p=0.8,
                    top_k=20,
                    presence_penalty=0.0,
                    max_tokens=64,
                ),
                unit_scores=(0.7,),
            ),
        ),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=development.dataset_hash,
        development_unit_keys=(MatchedUnitKey(prompt_sequence_id="sequence-1", model_seed=11),),
    )
    winner = selection.winning_profile
    effective_provider_configuration = {
        "endpoint": "http://127.0.0.1:8080",
        "request_mode": "completion",
    }
    protocol = ExperimentProtocol(
        run_tier=RunTier.CONFIRMATORY,
        schedule=ScheduleSpec(
            sequence_count=1,
            turns_per_sequence=2,
            model_seed_count=1,
            controller_seed_count=1,
            policy_count=5,
            logical_generation_count=10,
        ),
    )
    evaluation = EvaluationSpec(
        focal_policy_id="neural_persistent",
        required_serious_comparator_ids=("best_static", "heuristic_adaptive"),
        negative_control_policy_ids=("random_matched",),
        bootstrap_seed=101,
        permutation_seed=102,
    )
    confirmatory_analysis = ConfirmatoryAnalysisSpec(
        efficacy=EfficacyAnalysisSpec(
            practical_effect_threshold=evaluation.practical_effect_threshold,
            bootstrap_resamples=evaluation.bootstrap_resamples,
            confidence_level=evaluation.confidence_level,
            bootstrap_seed=evaluation.bootstrap_seed,
            permutation_resamples=evaluation.permutation_resamples,
            permutation_seed=evaluation.permutation_seed,
        ),
        recovery=RecoveryAnalysisSpec(
            practical_thresholds={
                RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: 0.02,
                RecoveryMetricName.POST_STRESSOR_REPETITION_CHANGE: 0.02,
                RecoveryMetricName.TIME_TO_RETURN_TO_TARGET_BAND: 1.0,
            },
            bootstrap_seed=103,
        ),
        attribution=AttributionAnalysisSpec(
            bootstrap_seed=104,
            permutation_seed=105,
        ),
        recovery_events=(
            RecoveryEventSpec(
                prompt_sequence_id="sequence-1",
                stressor_turn_index=0,
                recovery_turn_indexes=(1,),
                minimum_task_score_target=0.8,
                maximum_repetition_ratio_target=0.2,
            ),
        ),
        optional_metric_dispositions={
            "semantic_similarity": LimitationDisposition.INCONCLUSIVE,
        },
    )
    return ExperimentConfig(
        schema_version=1,
        experiment_id="provider-free-preregistration-test",
        dataset=DatasetReference(
            path="dataset.yaml",
            version=dataset.version,
            purpose=DatasetPurpose.EVALUATION,
            expected_dataset_sha256=dataset.dataset_hash,
            seal=DatasetSeal(
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                dataset_sha256=dataset.dataset_hash,
            ),
        ),
        provider=ProviderSelection(
            kind="llama_cpp",
            config_path="llama-cpp.local.yaml",
            expected_identity=ProviderIdentity(
                provider_type="llama_cpp",
                implementation_version="llama-cpp-completion-http-v1",
                model_alias="preregistration-test-model",
                build_id="preregistration-test-build",
                provider_config_hash=canonical_sha256(effective_provider_configuration),
                model_path="C:/models/preregistration-test.gguf",
                model_sha256="b" * 64,
                chat_template_sha256="c" * 64,
            ),
            expected_effective_configuration_json=canonical_json(effective_provider_configuration),
        ),
        policy_specs=(
            BestStaticPolicySpec(),
            HeuristicAdaptivePolicySpec(),
            NeuralMatchedHistoryStateResetPolicySpec(),
            NeuralPersistentPolicySpec(),
            RandomMatchedPolicySpec(),
        ),
        protocol=protocol,
        evaluation=evaluation,
        confirmatory_analysis=confirmatory_analysis,
        development_selection_input=DevelopmentSelectionInput(
            dataset=DatasetReference(
                path="development.yaml",
                version=development.version,
                purpose=DatasetPurpose.DEVELOPMENT,
                expected_dataset_sha256=development.dataset_hash,
            )
        ),
        static_selection_record=selection,
        model_seeds=(11,),
        controller_seeds=(21,),
        base_decoding_profile_id=winner.profile_id,
        base_decoding_profile=BaseDecodingProfile(
            temperature=winner.temperature,
            top_p=winner.top_p,
            top_k=winner.top_k,
            presence_penalty=winner.presence_penalty,
            max_tokens=winner.max_tokens,
        ),
        action_bounds=ActionBounds(),
        decoding_bounds=DecodingBounds(),
        metric_versions=METRIC_VERSIONS,
        decision_rule_version=protocol.decision_rule_version,
        database_schema_version=CURRENT_SCHEMA_VERSION,
        artifact_root="run",
    )


def _write_confirmatory_inputs(
    root: Path,
    *,
    dataset_relative_path: Path = Path("dataset.yaml"),
    artifact_root: str = "run",
) -> Path:
    dataset = _dataset(DatasetPurpose.EVALUATION)
    dataset_path = root / dataset_relative_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        yaml.safe_dump(dataset.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    config_payload = _confirmatory_config(dataset).model_dump(mode="json")
    config_payload["dataset"]["path"] = dataset_relative_path.as_posix()
    config_payload["artifact_root"] = artifact_root
    config_path = root / "confirmatory.yaml"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_publication_is_deterministic_idempotent_and_path_independent(
    tmp_path: Path,
) -> None:
    first_config = _write_confirmatory_inputs(
        tmp_path / "first",
        dataset_relative_path=Path("evaluation.yaml"),
        artifact_root="runs/first",
    )
    second_config = _write_confirmatory_inputs(
        tmp_path / "second",
        dataset_relative_path=Path("inputs/copied-evaluation.yaml"),
        artifact_root="unrelated/artifacts",
    )
    first_output = tmp_path / "published" / "first.seal.json"
    second_output = tmp_path / "elsewhere" / "second.seal.json"

    first = publish_preregistration(first_config, first_output)
    second = publish_preregistration(second_config, second_output)
    repeated = publish_preregistration(first_config, first_output)

    assert first.created is True
    assert second.created is True
    assert repeated.created is False
    assert first.seal == second.seal == repeated.seal
    assert first_output.read_bytes() == canonical_json_bytes(first.seal)
    assert second_output.read_bytes() == canonical_json_bytes(second.seal)
    assert load_preregistration_seal(first_output) == first.seal
    assert first.config_path == first_config.resolve()
    assert first.dataset_path == (tmp_path / "first" / "evaluation.yaml").resolve()


def test_publication_never_overwrites_a_differing_existing_artifact(
    tmp_path: Path,
) -> None:
    config_path = _write_confirmatory_inputs(tmp_path / "inputs")
    output_path = tmp_path / "published" / "confirmatory.seal.json"
    output_path.parent.mkdir(parents=True)
    existing = b'{"different":true}'
    output_path.write_bytes(existing)

    with pytest.raises(FileExistsError, match="refusing to overwrite differing"):
        publish_preregistration(config_path, output_path)

    assert output_path.read_bytes() == existing


def test_publication_requires_confirmatory_unsealed_input(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    smoke = root / "configs" / "experiments" / "model-backed-engineering-smoke.yaml"
    with pytest.raises(ValueError, match="requires a confirmatory protocol"):
        publish_preregistration(smoke, tmp_path / "smoke.seal.json")

    config_path = _write_confirmatory_inputs(tmp_path / "confirmatory")
    fake_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fake_payload["provider"] = ProviderSelection(
        kind="fake",
        expected_identity=fake_provider_identity(),
        expected_effective_configuration_json=fake_provider_effective_configuration_json(),
    ).model_dump(mode="json")
    fake_config = config_path.with_name("fake-confirmatory.yaml")
    fake_config.write_text(yaml.safe_dump(fake_payload, sort_keys=False), encoding="utf-8")
    fake_output = tmp_path / "fake.seal.json"
    with pytest.raises(ValueError, match="explicit llama_cpp provider"):
        publish_preregistration(fake_config, fake_output)
    assert not fake_output.exists()

    publication = publish_preregistration(config_path, tmp_path / "first.seal.json")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["preregistration"] = publication.seal.model_dump(mode="json")
    sealed_config = config_path.with_name("already-sealed.yaml")
    sealed_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="requires an unsealed configuration"):
        publish_preregistration(sealed_config, tmp_path / "second.seal.json")


def test_cli_preregister_publishes_without_provider_or_network_paths(
    capsys: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config_path = _write_confirmatory_inputs(tmp_path / "inputs")
    output_path = tmp_path / "published" / "confirmatory.seal.json"

    def forbidden_provider_path(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preregistration entered a provider or network path")

    monkeypatch.setattr(
        "neurallm.experiments.workflow.construct_provider",
        forbidden_provider_path,
    )
    monkeypatch.setattr("neurallm.cli.preflight_llama_cpp", forbidden_provider_path)

    assert (
        main(
            [
                "preregister",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["command"] == "preregister"
    assert payload["created"] is True
    assert payload["provider_constructed"] is False
    assert payload["network_requested"] is False
    assert payload["output_path"] == str(output_path.resolve())
    assert len(payload["scientific_identity_sha256"]) == 64
    assert len(payload["preregistration_sha256"]) == 64
    assert (
        load_preregistration_seal(output_path).scientific_identity_sha256
        == (payload["scientific_identity_sha256"])
    )

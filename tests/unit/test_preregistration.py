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
)
from neurallm.domain.serialization import canonical_json_bytes
from neurallm.evaluation.attribution import AttributionAnalysisSpec
from neurallm.evaluation.confirmatory import ConfirmatoryAnalysisSpec, RecoveryEventSpec
from neurallm.evaluation.models import DatasetPurpose, EvaluationSpec
from neurallm.evaluation.recovery import RecoveryAnalysisSpec, RecoveryMetricName
from neurallm.evaluation.scientific import EfficacyAnalysisSpec, LimitationDisposition
from neurallm.evaluation.selection import StaticProfile
from neurallm.experiments.config import (
    BaseDecodingProfile,
    DatasetReference,
    DevelopmentSelectionInput,
    ExperimentConfig,
    ProviderSelection,
    load_experiment_config,
)
from neurallm.experiments.dataset import (
    DatasetSeal,
    PromptCase,
    PromptDataset,
    PromptSequence,
    load_dataset,
)
from neurallm.experiments.plan import build_plan
from neurallm.experiments.preregistration import (
    load_preregistration_seal,
    publish_preregistration,
)
from neurallm.experiments.protocol import (
    ExperimentProtocol,
    RunTier,
    ScheduleSpec,
)
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.metrics.deterministic import METRIC_VERSIONS
from neurallm.metrics.validators import ValidatorSpec
from neurallm.providers.fake import (
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.storage.migrations import CURRENT_SCHEMA_VERSION
from tests.integration.pilot_selection_helpers import build_test_static_selection_evidence


def _dataset(purpose: DatasetPurpose) -> PromptDataset:
    if purpose is DatasetPurpose.DEVELOPMENT:
        sequences = tuple(
            PromptSequence(
                sequence_id=f"pilot-sequence-{sequence_index:02d}",
                cases=tuple(
                    PromptCase(
                        case_id=f"pilot-sequence-{sequence_index:02d}-turn-{turn_index}",
                        prompt_family="pilot_selection_fixture",
                        prompt=(f"Return development response {sequence_index}-{turn_index}."),
                        prompt_features=PromptFeatures({"constraint_count": 1.0}),
                        validator=ValidatorSpec(kind="non_empty"),
                    )
                    for turn_index in range(4)
                ),
            )
            for sequence_index in range(1, 7)
        )
    else:
        sequences = (
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
        )
    return PromptDataset(
        schema_version=1,
        dataset_id=f"preregistration-{purpose.value}",
        version=f"preregistration-{purpose.value}-v1",
        purpose=purpose,
        sequences=sequences,
    )


def _confirmatory_config(dataset: PromptDataset) -> ExperimentConfig:
    development = _dataset(DatasetPurpose.DEVELOPMENT)
    selection_evidence = build_test_static_selection_evidence(
        development_dataset=development,
        winning_profile=StaticProfile(
            profile_id="static-balanced-v1",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            presence_penalty=0.0,
            max_tokens=192,
        ),
    )
    winner = selection_evidence.selection_record.winning_profile
    pilot_manifest = selection_evidence.candidates[0].source_run_manifest
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
            expected_identity=pilot_manifest.provider_identity,
            expected_effective_configuration_json=(
                pilot_manifest.provider_effective_configuration_json
            ),
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
        static_selection_evidence=selection_evidence,
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
    development = _dataset(DatasetPurpose.DEVELOPMENT)
    (root / "development.yaml").write_text(
        yaml.safe_dump(development.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    config = _confirmatory_config(dataset)
    selection_evidence = config.static_selection_evidence
    assert selection_evidence is not None
    selection_path = root / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(selection_evidence))
    config_payload = config.model_dump(mode="json", exclude_none=True)
    config_payload["dataset"]["path"] = dataset_relative_path.as_posix()
    config_payload["artifact_root"] = artifact_root
    config_payload["static_selection_evidence"] = {
        "path": "selection.json",
        "expected_sha256": selection_evidence.evidence_sha256,
    }
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
    assert (
        first.development_selection_dataset_path
        == (tmp_path / "first" / "development.yaml").resolve()
    )


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
    sealed_config_path = config_path.with_name("confirmatory.executable.local.yaml")
    with pytest.raises(FileExistsError, match="refusing to overwrite differing"):
        publish_preregistration(config_path, output_path, sealed_config_path)
    assert not sealed_config_path.exists()


def test_publication_materializes_an_idempotent_executable_sealed_config(
    tmp_path: Path,
) -> None:
    config_path = _write_confirmatory_inputs(tmp_path / "inputs")
    seal_path = tmp_path / "published" / "confirmatory.seal.json"
    sealed_config_path = config_path.with_name("confirmatory.executable.local.yaml")
    source_reference = load_yaml_mapping(config_path)["static_selection_evidence"]

    first = publish_preregistration(config_path, seal_path, sealed_config_path)
    repeated = publish_preregistration(config_path, seal_path, sealed_config_path)

    assert first.created is True
    assert first.sealed_config_created is True
    assert first.sealed_config_path == sealed_config_path.resolve()
    assert repeated.created is False
    assert repeated.sealed_config_created is False
    payload = load_yaml_mapping(sealed_config_path)
    assert payload["preregistration"] == first.seal.model_dump(mode="json")
    assert payload["static_selection_evidence"] == source_reference
    assert "static_selection_record" not in payload

    loaded_config = load_experiment_config(sealed_config_path)
    loaded_dataset = load_dataset(
        loaded_config.dataset_path,
        expected_version=loaded_config.config.dataset.version,
    )
    plan = build_plan(loaded_config, loaded_dataset)
    assert plan.preregistration == first.seal
    assert plan.scientific_identity_sha256 == first.scientific_identity_sha256


def test_publication_rejects_an_unsafe_or_conflicting_sealed_config_target(
    tmp_path: Path,
) -> None:
    config_path = _write_confirmatory_inputs(tmp_path / "inputs")
    seal_path = tmp_path / "published" / "confirmatory.seal.json"

    with pytest.raises(TypeError, match="sealed_config_output_path"):
        publish_preregistration(config_path, seal_path, "not-a-path")  # type: ignore[arg-type]

    shared_output = config_path.with_name("shared.local.yaml")
    with pytest.raises(ValueError, match="output paths must be distinct"):
        publish_preregistration(config_path, shared_output, shared_output)
    assert not shared_output.exists()

    with pytest.raises(ValueError, match="must be beside the source configuration"):
        publish_preregistration(
            config_path,
            seal_path,
            tmp_path / "elsewhere" / "confirmatory.local.yaml",
        )
    assert not seal_path.exists()

    sealed_config_path = config_path.with_name("confirmatory.local.yaml")
    sealed_config_path.write_text("different: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="sealed experiment configuration"):
        publish_preregistration(config_path, seal_path, sealed_config_path)
    assert not seal_path.exists()
    assert sealed_config_path.read_text(encoding="utf-8") == "different: true\n"


def test_publication_validates_the_declared_development_dataset_before_writing(
    tmp_path: Path,
) -> None:
    config_path = _write_confirmatory_inputs(tmp_path / "inputs")
    development_path = config_path.parent / "development.yaml"
    development_payload = load_yaml_mapping(development_path)
    development_payload["sequences"][0]["cases"][0]["prompt"] = "Drifted prompt."
    development_path.write_text(
        yaml.safe_dump(development_payload, sort_keys=False),
        encoding="utf-8",
    )
    seal_path = tmp_path / "published" / "confirmatory.seal.json"
    sealed_config_path = config_path.with_name("confirmatory.local.yaml")

    with pytest.raises(ValueError, match="dataset SHA-256"):
        publish_preregistration(config_path, seal_path, sealed_config_path)

    assert not seal_path.exists()
    assert not sealed_config_path.exists()


def test_publication_binds_pilot_request_hashes_to_the_development_prompts(
    tmp_path: Path,
) -> None:
    config_path = _write_confirmatory_inputs(tmp_path / "inputs")
    loaded = load_experiment_config(config_path)
    config = loaded.config
    evidence = config.static_selection_evidence
    assert evidence is not None
    assert config.policy_specs is not None
    development = _dataset(DatasetPurpose.DEVELOPMENT)
    unbound_evidence = build_test_static_selection_evidence(
        dataset_version=development.version,
        dataset_sha256=development.dataset_hash,
        sequence_ids=tuple(sequence.sequence_id for sequence in development.sequences),
        provider_identity=config.provider.expected_identity,
        provider_effective_configuration_json=(
            config.provider.expected_effective_configuration_json
        ),
        policy_specs=config.policy_specs,
        action_bounds=config.action_bounds,
        decoding_bounds=config.decoding_bounds,
        metric_versions=config.metric_versions,
        database_schema_version=config.database_schema_version,
        winning_profile=evidence.selection_record.winning_profile,
    )
    selection_path = config_path.parent / "selection.json"
    selection_path.write_bytes(canonical_json_bytes(unbound_evidence))
    config_payload = load_yaml_mapping(config_path)
    config_payload["static_selection_evidence"]["expected_sha256"] = (
        unbound_evidence.evidence_sha256
    )
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    seal_path = tmp_path / "published" / "confirmatory.seal.json"
    sealed_config_path = config_path.with_name("confirmatory.local.yaml")

    with pytest.raises(ValueError, match="request hash differs"):
        publish_preregistration(config_path, seal_path, sealed_config_path)

    assert not seal_path.exists()
    assert not sealed_config_path.exists()


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
    sealed_config_path = config_path.with_name("confirmatory.executable.local.yaml")

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
                "--sealed-config-output",
                str(sealed_config_path),
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
    assert payload["development_selection_dataset_path"] == str(
        (config_path.parent / "development.yaml").resolve()
    )
    assert payload["sealed_config_path"] == str(sealed_config_path.resolve())
    assert payload["sealed_config_created"] is True
    assert len(payload["scientific_identity_sha256"]) == 64
    assert len(payload["preregistration_sha256"]) == 64
    assert (
        load_preregistration_seal(output_path).scientific_identity_sha256
        == (payload["scientific_identity_sha256"])
    )
    assert load_experiment_config(sealed_config_path).config.preregistration is not None

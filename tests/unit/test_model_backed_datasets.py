"""Frozen identities and dimensions for the model-backed run datasets."""

from pathlib import Path

from neurallm.evaluation.models import DatasetPurpose
from neurallm.experiments.dataset import DatasetSeal, load_dataset, validate_dataset_identity
from neurallm.experiments.protocol import (
    ATTRIBUTION_POLICY_ID,
    EFFICACY_POLICY_IDS,
    MODEL_BACKED_POLICY_IDS,
    ExperimentProtocol,
    RunTier,
    ScheduleSpec,
)
from neurallm.experiments.yaml_loader import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[2]
ENGINEERING_DATASET = ROOT / "datasets" / "development" / ("model-backed-engineering-smoke-v1.yaml")
CONFIRMATORY_DATASET = ROOT / "datasets" / "evaluation" / ("model-backed-confirmatory-v1.yaml")
CONFIRMATORY_SEAL = ROOT / "datasets" / "evaluation" / ("model-backed-confirmatory-v1.seal.yaml")
ENGINEERING_DATASET_SHA256 = "14c382a04acbe9394474f05cf84d8389833058afc2dc6feda21a023d46e45ef3"
CONFIRMATORY_DATASET_SHA256 = "7cf2d3a9fa35735aadc9186438277d2b5f6b7beb9f96e9fc9bbeb400da2b5d72"


def test_engineering_smoke_dataset_has_frozen_two_by_two_identity() -> None:
    dataset = load_dataset(
        ENGINEERING_DATASET,
        expected_version="model-backed-engineering-smoke-v1",
    ).dataset

    validate_dataset_identity(
        dataset,
        expected_version="model-backed-engineering-smoke-v1",
        expected_purpose=DatasetPurpose.DEVELOPMENT,
        expected_sha256=ENGINEERING_DATASET_SHA256,
        seal=None,
    )
    assert dataset.dataset_id == "model-backed-engineering-smoke"
    assert dataset.purpose is DatasetPurpose.DEVELOPMENT
    assert dataset.dataset_hash == ENGINEERING_DATASET_SHA256
    assert len(dataset.sequences) == 2
    assert tuple(len(sequence.cases) for sequence in dataset.sequences) == (2, 2)


def test_confirmatory_dataset_has_frozen_twenty_four_by_four_sealed_identity() -> None:
    dataset = load_dataset(
        CONFIRMATORY_DATASET,
        expected_version="model-backed-confirmatory-v1",
    ).dataset
    seal = DatasetSeal.model_validate(load_yaml_mapping(CONFIRMATORY_SEAL))

    validate_dataset_identity(
        dataset,
        expected_version="model-backed-confirmatory-v1",
        expected_purpose=DatasetPurpose.EVALUATION,
        expected_sha256=CONFIRMATORY_DATASET_SHA256,
        seal=seal,
    )
    assert dataset.dataset_id == "model-backed-confirmatory"
    assert dataset.purpose is DatasetPurpose.EVALUATION
    assert dataset.dataset_hash == CONFIRMATORY_DATASET_SHA256
    assert len(dataset.sequences) == 24
    assert {len(sequence.cases) for sequence in dataset.sequences} == {4}
    assert seal.dataset_id == dataset.dataset_id
    assert seal.dataset_version == dataset.version
    assert seal.dataset_sha256 == dataset.dataset_hash


def test_model_backed_policy_roles_and_schedule_counts_are_exact() -> None:
    assert MODEL_BACKED_POLICY_IDS == (
        "best_static",
        "heuristic_adaptive",
        "neural_matched_history_state_reset",
        "neural_persistent",
        "random_matched",
    )
    assert EFFICACY_POLICY_IDS == (
        "best_static",
        "heuristic_adaptive",
        "neural_persistent",
        "random_matched",
    )
    assert set(MODEL_BACKED_POLICY_IDS) == {*EFFICACY_POLICY_IDS, ATTRIBUTION_POLICY_ID}
    assert ATTRIBUTION_POLICY_ID not in EFFICACY_POLICY_IDS

    schedules = (
        (RunTier.ENGINEERING_SMOKE, 2, 2, 1, 20),
        (RunTier.DEVELOPMENT_PILOT, 6, 4, 2, 240),
        (RunTier.CONFIRMATORY, 24, 4, 5, 2_400),
    )
    for tier, sequence_count, turn_count, model_seed_count, expected_count in schedules:
        protocol = ExperimentProtocol(
            run_tier=tier,
            schedule=ScheduleSpec(
                sequence_count=sequence_count,
                turns_per_sequence=turn_count,
                model_seed_count=model_seed_count,
                controller_seed_count=1,
                logical_generation_count=expected_count,
            ),
        )
        assert protocol.schedule.logical_generation_count == expected_count

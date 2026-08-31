"""Strict immutable Phase 3 evaluation model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neurallm.evaluation import (
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    Phase3Verdict,
    SequenceExpectation,
)


def test_evaluation_spec_defaults_are_preregistered_and_policy_ids_are_canonical() -> None:
    spec = EvaluationSpec(
        focal_policy_id="focal",
        required_serious_comparator_ids=("static", "heuristic"),
        negative_control_policy_ids=("random",),
        bootstrap_seed=7,
        permutation_seed=8,
    )

    assert spec.required_serious_comparator_ids == ("heuristic", "static")
    assert spec.bootstrap_resamples == 10_000
    assert spec.confidence_level == 0.95
    assert spec.permutation_resamples == 10_000
    assert spec.multiplicity_correction_version == "holm-v1"
    with pytest.raises(ValidationError):
        spec.bootstrap_seed = 9


def test_statistical_design_rejects_development_data_and_final_phase_vocabulary() -> None:
    with pytest.raises(ValidationError, match="cannot consume development data"):
        ExpectedEvaluationDesign(
            dataset_purpose=DatasetPurpose.DEVELOPMENT,
            dataset_sha256="a" * 64,
            provider_identity_id="b" * 64,
            sequences=(SequenceExpectation(prompt_sequence_id="sequence", turn_count=1),),
            model_seeds=(1,),
            controller_seeds=(2,),
            policy_ids=("focal", "static"),
        )

    with pytest.raises(ValidationError, match="dataset seal identity"):
        ExpectedEvaluationDesign(
            dataset_purpose=DatasetPurpose.EVALUATION,
            dataset_sha256="a" * 64,
            provider_identity_id="b" * 64,
            sequences=(SequenceExpectation(prompt_sequence_id="sequence", turn_count=1),),
            model_seeds=(1,),
            controller_seeds=(2,),
            policy_ids=("focal", "static"),
        )

    assert {verdict.value for verdict in Phase3Verdict} == {
        "superior",
        "inferior",
        "equivalent",
        "inconclusive",
        "invalid",
    }
    assert all("validated" not in verdict.value for verdict in Phase3Verdict)


@pytest.mark.parametrize(
    "updates",
    [
        {"required_serious_comparator_ids": []},
        {"required_serious_comparator_ids": ["static", "static"]},
        {"required_serious_comparator_ids": ["focal"]},
        {
            "required_serious_comparator_ids": ["static"],
            "negative_control_policy_ids": ["static"],
        },
        {"equivalence_margin": 0.03},
    ],
)
def test_evaluation_spec_rejects_ambiguous_policy_roles_and_thresholds(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "focal_policy_id": "focal",
        "required_serious_comparator_ids": ["static"],
        "bootstrap_seed": 7,
        "permutation_seed": 8,
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        EvaluationSpec.model_validate(values)


def test_expected_design_normalizes_serialized_axes_and_rejects_ambiguous_grids() -> None:
    design = ExpectedEvaluationDesign.model_validate(
        {
            "dataset_purpose": "synthetic",
            "dataset_sha256": "a" * 64,
            "provider_identity_id": "b" * 64,
            "sequences": {"sequence-b": 1, "sequence-a": 2},
            "model_seeds": [2, 1],
            "controller_seeds": [4, 3],
            "policy_ids": ["static", "focal"],
        }
    )

    assert tuple(sequence.prompt_sequence_id for sequence in design.sequences) == (
        "sequence-a",
        "sequence-b",
    )
    assert design.model_seeds == (1, 2)
    assert design.controller_seeds == (3, 4)
    assert design.policy_ids == ("focal", "static")

    base = design.model_dump()
    invalid_updates = (
        {"sequences": ()},
        {
            "sequences": (
                SequenceExpectation(prompt_sequence_id="same", turn_count=1),
                SequenceExpectation(prompt_sequence_id="same", turn_count=2),
            )
        },
        {"model_seeds": ()},
        {"controller_seeds": (1, 1)},
        {"policy_ids": ()},
        {"policy_ids": ("same", "same")},
        {"dataset_seal_sha256": "c" * 64},
    )
    for updates in invalid_updates:
        with pytest.raises(ValidationError):
            ExpectedEvaluationDesign.model_validate({**base, **updates})

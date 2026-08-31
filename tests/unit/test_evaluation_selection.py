"""Development-only static-profile selection contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neurallm.evaluation import (
    DatasetPurpose,
    MatchedUnitKey,
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    select_best_static,
)

DEVELOPMENT_HASH = "d" * 64


def unit_keys(count: int) -> tuple[MatchedUnitKey, ...]:
    return tuple(
        MatchedUnitKey(prompt_sequence_id=f"development-{index}", model_seed=101)
        for index in range(count)
    )


def profile(profile_id: str, temperature: float) -> StaticProfile:
    return StaticProfile(
        profile_id=profile_id,
        temperature=temperature,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.0,
        max_tokens=128,
    )


def test_static_selection_is_deterministic_frozen_and_uses_lexical_tie_break() -> None:
    candidate_b = StaticCandidateResult(
        profile=profile("b", 0.8),
        unit_scores=(0.6, 0.8),
    )
    candidate_a = StaticCandidateResult(
        profile=profile("a", 0.7),
        unit_scores=(0.7, 0.7),
    )

    selected = select_best_static(
        (candidate_b, candidate_a),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=DEVELOPMENT_HASH,
        development_unit_keys=unit_keys(2),
    )

    assert selected.dataset_purpose is DatasetPurpose.DEVELOPMENT
    assert tuple(candidate.profile.profile_id for candidate in selected.candidate_results) == (
        "a",
        "b",
    )
    assert selected.winning_profile.profile_id == "a"
    assert len(selected.selection_result_sha256) == 64
    assert type(selected).model_validate(selected.model_dump(mode="json")) == selected
    with pytest.raises(ValidationError):
        selected.winning_profile = candidate_b.profile


@pytest.mark.parametrize(
    "forbidden_purpose",
    [DatasetPurpose.EVALUATION, DatasetPurpose.SYNTHETIC],
)
def test_static_selection_rejects_nondevelopment_data_and_sealed_leakage(
    forbidden_purpose: DatasetPurpose,
) -> None:
    candidates = (
        StaticCandidateResult(profile=profile("a", 0.7), unit_scores=(0.6,)),
        StaticCandidateResult(profile=profile("b", 0.8), unit_scores=(0.7,)),
    )

    with pytest.raises(ValueError, match="development data only"):
        select_best_static(
            candidates,
            dataset_purpose=forbidden_purpose,
            dataset_sha256=DEVELOPMENT_HASH,
            development_unit_keys=unit_keys(1),
        )


def test_static_selection_evidence_rejects_incomplete_or_tampered_records() -> None:
    candidate_a = StaticCandidateResult(
        profile=profile("a", 0.7),
        unit_scores=(0.8, 0.7),
    )
    candidate_b = StaticCandidateResult(
        profile=profile("b", 0.8),
        unit_scores=(0.6, 0.5),
    )
    selected = select_best_static(
        (candidate_a, candidate_b),
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=DEVELOPMENT_HASH,
        development_unit_keys=unit_keys(2),
    )
    base = selected.model_dump()
    candidates = base["candidate_results"]
    development_unit_keys = base["development_unit_keys"]
    assert isinstance(candidates, tuple)
    assert isinstance(development_unit_keys, tuple)

    tampered_payloads = (
        {**base, "candidate_results": candidates[:1]},
        {**base, "candidate_results": tuple(reversed(candidates))},
        {
            **base,
            "candidate_results": (
                candidates[0],
                {**candidates[1], "unit_scores": (0.5,)},
            ),
        },
        {**base, "winning_profile": candidates[1]["profile"]},
        {**base, "selection_result_sha256": "0" * 64},
    )
    for payload in tampered_payloads:
        with pytest.raises(ValidationError):
            StaticSelectionRecord.model_validate(payload)

    with pytest.raises(ValidationError, match="requires development unit scores"):
        StaticCandidateResult(profile=profile("empty", 0.5), unit_scores=())
    with pytest.raises(ValueError, match="at least two"):
        select_best_static(
            (candidate_a,),
            dataset_purpose=DatasetPurpose.DEVELOPMENT,
            dataset_sha256=DEVELOPMENT_HASH,
            development_unit_keys=unit_keys(2),
        )


def test_static_selection_unit_keys_fail_closed() -> None:
    candidates = (
        StaticCandidateResult(profile=profile("a", 0.7), unit_scores=(0.8, 0.7)),
        StaticCandidateResult(profile=profile("b", 0.8), unit_scores=(0.6, 0.5)),
    )
    selected = select_best_static(
        candidates,
        dataset_purpose=DatasetPurpose.DEVELOPMENT,
        dataset_sha256=DEVELOPMENT_HASH,
        development_unit_keys=unit_keys(2),
    )
    base = selected.model_dump()
    frozen_keys = selected.development_unit_keys

    missing_field = dict(base)
    missing_field.pop("development_unit_keys")
    invalid_payloads = (
        (missing_field, "Field required"),
        (
            {**base, "development_unit_keys": (frozen_keys[0], frozen_keys[0])},
            "nonempty, sorted, and unique",
        ),
        (
            {**base, "development_unit_keys": tuple(reversed(frozen_keys))},
            "nonempty, sorted, and unique",
        ),
        (
            {**base, "development_unit_keys": frozen_keys[:1]},
            "score vector must align",
        ),
        (
            {
                **base,
                "candidate_results": (
                    base["candidate_results"][0],
                    {
                        **base["candidate_results"][1],
                        "unit_scores": (0.5,),
                    },
                ),
            },
            "score vector must align",
        ),
    )
    for payload, message in invalid_payloads:
        with pytest.raises(ValidationError, match=message):
            StaticSelectionRecord.model_validate(payload)


def test_static_selection_requires_one_fixed_generation_budget() -> None:
    fixed = profile("fixed", 0.7)
    shorter = profile("shorter", 0.8).model_copy(update={"max_tokens": 1})

    with pytest.raises(ValidationError, match="fixed max_tokens"):
        select_best_static(
            (
                StaticCandidateResult(profile=fixed, unit_scores=(0.5,)),
                StaticCandidateResult(profile=shorter, unit_scores=(0.9,)),
            ),
            dataset_purpose=DatasetPurpose.DEVELOPMENT,
            dataset_sha256=DEVELOPMENT_HASH,
            development_unit_keys=unit_keys(1),
        )

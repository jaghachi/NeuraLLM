"""Tests for versioned deterministic prompt datasets."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neurallm.experiments.dataset import PromptDataset, load_dataset


def dataset_payload(*, reverse: bool = False) -> dict[str, object]:
    sequences = [
        {
            "sequence_id": "sequence-b",
            "cases": [
                {
                    "case_id": "b-0",
                    "prompt_family": "constrained",
                    "prompt": "Return a non-empty answer.",
                    "prompt_features": {"difficulty": 0.2},
                    "validator": {"kind": "non_empty"},
                }
            ],
        },
        {
            "sequence_id": "sequence-a",
            "cases": [
                {
                    "case_id": "a-0",
                    "prompt_family": "constrained",
                    "prompt": "Include alpha and beta.",
                    "prompt_features": {"difficulty": 0.4},
                    "validator": {
                        "kind": "contains_all",
                        "required_terms": ["alpha", "beta"],
                    },
                },
                {
                    "case_id": "a-1",
                    "prompt_family": "json",
                    "prompt": "Return JSON with answer and reason.",
                    "validator": {
                        "kind": "json_object",
                        "required_json_keys": ["answer", "reason"],
                    },
                },
            ],
        },
    ]
    if reverse:
        sequences.reverse()
    return {
        "schema_version": 1,
        "dataset_id": "phase2-smoke",
        "version": "smoke-v1",
        "sequences": sequences,
    }


def test_dataset_hash_is_independent_of_sequence_iteration_order() -> None:
    first = PromptDataset.model_validate(dataset_payload())
    second = PromptDataset.model_validate(dataset_payload(reverse=True))

    assert first.dataset_hash == second.dataset_hash


def test_loader_enforces_expected_version(tmp_path: Path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(dataset_payload(), sort_keys=False), encoding="utf-8")

    loaded = load_dataset(path, expected_version="smoke-v1")

    assert loaded.dataset.version == "smoke-v1"
    with pytest.raises(ValueError, match="version mismatch"):
        load_dataset(path, expected_version="other-v1")


def test_duplicate_sequence_or_case_ids_fail_closed() -> None:
    sequence_duplicate = dataset_payload()
    sequence_duplicate["sequences"] = [
        sequence_duplicate["sequences"][0],  # type: ignore[index]
        sequence_duplicate["sequences"][0],  # type: ignore[index]
    ]
    with pytest.raises(ValidationError, match="sequence ids"):
        PromptDataset.model_validate(sequence_duplicate)

    case_duplicate = dataset_payload()
    cases = case_duplicate["sequences"][0]["cases"]  # type: ignore[index]
    cases.append(cases[0])
    with pytest.raises(ValidationError, match="case ids"):
        PromptDataset.model_validate(case_duplicate)

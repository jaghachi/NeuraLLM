"""Canonical Phase 3 analysis-contract identity shared by run and storage."""

from __future__ import annotations

from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import DatasetPurpose, EvaluationSpec, ExpectedEvaluationDesign
from neurallm.evaluation.selection import StaticSelectionRecord


def phase3_analysis_contract_sha256(
    *,
    experiment_plan_sha256: str,
    evaluation_spec: EvaluationSpec,
    evaluation_spec_sha256: str,
    static_selection_record: StaticSelectionRecord,
    static_selection_result_sha256: str,
    evaluation_design: ExpectedEvaluationDesign,
    dataset_sha256: str,
    dataset_purpose: DatasetPurpose,
    dataset_seal_sha256: str | None,
) -> str:
    """Hash the complete evaluator provenance expected before execution.

    The duplicated record/hash pairs are intentional: the contract binds both
    the canonical evidence and the published identities that downstream
    artifacts expose.
    """

    if evaluation_spec_sha256 != canonical_sha256(evaluation_spec):
        raise ValueError("evaluation spec hash does not match its canonical evidence")
    if static_selection_result_sha256 != static_selection_record.selection_result_sha256:
        raise ValueError("static selection hash does not match its canonical evidence")
    if (
        evaluation_design.dataset_sha256 != dataset_sha256
        or evaluation_design.dataset_purpose is not dataset_purpose
        or evaluation_design.dataset_seal_sha256 != dataset_seal_sha256
    ):
        raise ValueError("evaluation design disagrees with the analysis dataset identity")
    return canonical_sha256(
        {
            "schema_version": 1,
            "implementation_version": "phase3-analysis-contract-v1",
            "experiment_plan_sha256": experiment_plan_sha256,
            "evaluation_spec": evaluation_spec,
            "evaluation_spec_sha256": evaluation_spec_sha256,
            "static_selection_record": static_selection_record,
            "static_selection_result_sha256": static_selection_result_sha256,
            "evaluation_design": evaluation_design,
            "dataset_sha256": dataset_sha256,
            "dataset_purpose": dataset_purpose,
            "dataset_seal_sha256": dataset_seal_sha256,
        }
    )


__all__ = ["phase3_analysis_contract_sha256"]

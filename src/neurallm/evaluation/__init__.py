"""Pure, deterministic Phase 3 evaluation contracts and algorithms."""

from neurallm.evaluation.aggregation import (
    aggregate_matched_units,
    evaluation_input_sha256,
    expected_turn_keys,
    validate_exact_coverage,
)
from neurallm.evaluation.engine import evaluate_phase3
from neurallm.evaluation.models import (
    BootstrapResult,
    CoverageResult,
    DatasetPurpose,
    EvaluationSpec,
    ExpectedEvaluationDesign,
    GuardrailName,
    GuardrailResult,
    GuardrailStatus,
    HolmAdjustedPValue,
    MatchedUnitKey,
    PairwiseComparisonResult,
    PermutationTestResult,
    Phase3EvaluationResult,
    Phase3Verdict,
    SequenceExpectation,
    SequencePolicyOutcome,
    TurnEvaluationRecord,
    TurnRecordKey,
)
from neurallm.evaluation.selection import (
    StaticCandidateResult,
    StaticProfile,
    StaticSelectionRecord,
    select_best_static,
)
from neurallm.evaluation.statistics import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_permutation_test,
)

__all__ = [
    "BootstrapResult",
    "CoverageResult",
    "DatasetPurpose",
    "EvaluationSpec",
    "ExpectedEvaluationDesign",
    "GuardrailName",
    "GuardrailResult",
    "GuardrailStatus",
    "HolmAdjustedPValue",
    "MatchedUnitKey",
    "PairwiseComparisonResult",
    "PermutationTestResult",
    "Phase3EvaluationResult",
    "Phase3Verdict",
    "SequenceExpectation",
    "SequencePolicyOutcome",
    "StaticCandidateResult",
    "StaticProfile",
    "StaticSelectionRecord",
    "TurnEvaluationRecord",
    "TurnRecordKey",
    "aggregate_matched_units",
    "evaluate_phase3",
    "evaluation_input_sha256",
    "expected_turn_keys",
    "holm_adjust",
    "paired_bootstrap_ci",
    "paired_sign_flip_permutation_test",
    "select_best_static",
    "validate_exact_coverage",
]

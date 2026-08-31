"""Matched-history persistent-state output attribution statistics."""

from __future__ import annotations

from collections.abc import Sequence
from math import isclose
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from neurallm.domain.models import FiniteFloat, NonNegativeInt, PositiveInt, SqliteInt64
from neurallm.evaluation.models import BootstrapResult, PermutationTestResult
from neurallm.evaluation.scientific import (
    ATTRIBUTION_COMPARATOR_ID,
    FOCAL_POLICY_ID,
    ScientificEvidenceGate,
    ScientificEvidenceKind,
    ScientificEvidenceStatus,
    ScientificFrozenModel,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
)
from neurallm.evaluation.statistics import (
    paired_bootstrap_ci,
    paired_sign_flip_permutation_test,
)


class AttributionAnalysisSpec(ScientificFrozenModel):
    """Frozen signed-output threshold and deterministic statistical settings."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["persistent-state-output-attribution-v1"] = (
        "persistent-state-output-attribution-v1"
    )
    primary_metric: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    practical_effect_threshold: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    bootstrap_resamples: PositiveInt = 10_000
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, allow_inf_nan=False)
    bootstrap_seed: SqliteInt64
    permutation_resamples: PositiveInt = 10_000
    permutation_seed: SqliteInt64


class PersistentStateAttributionResult(ScientificFrozenModel):
    """Persistent-minus-reset output evidence for intervention turns only."""

    comparison_kind: Literal["persistent_state_attribution"] = "persistent_state_attribution"
    focal_policy_id: Literal["neural_persistent"] = FOCAL_POLICY_ID
    comparator_policy_id: Literal["neural_matched_history_state_reset"] = ATTRIBUTION_COMPARATOR_ID
    attribution_only: Literal[True] = True
    included_in_efficacy: Literal[False] = False
    included_in_holm_family: Literal[False] = False
    turn_zero_excluded_from_effect: Literal[True] = True
    primary_metric: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    unit_count: NonNegativeInt
    mean_difference: FiniteFloat | None = None
    bootstrap: BootstrapResult | None = None
    permutation: PermutationTestResult | None = None
    practical_effect_threshold: float = Field(ge=0.0, allow_inf_nan=False)
    behavioral_alias: bool = False
    causal_guardrails: tuple[ScientificGuardrailResult, ...]
    status: ScientificEvidenceStatus
    detail: str = Field(min_length=1)

    @field_validator("causal_guardrails", mode="before")
    @classmethod
    def _accept_guardrail_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("causal_guardrails")
    @classmethod
    def _canonical_guardrails(
        cls,
        values: tuple[ScientificGuardrailResult, ...],
    ) -> tuple[ScientificGuardrailResult, ...]:
        return tuple(sorted(values, key=lambda value: value.evidence_key))

    @model_validator(mode="after")
    def _require_causal_guardrails(self) -> Self:
        if not self.causal_guardrails and self.status is not ScientificEvidenceStatus.INVALID:
            raise ValueError("persistent-state attribution requires causal guardrails")
        return self

    @field_validator("status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceStatus(value)
        return value

    @model_validator(mode="after")
    def _validate_attribution_evidence(self) -> Self:
        keys = tuple(guardrail.evidence_key for guardrail in self.causal_guardrails)
        if len(keys) != len(set(keys)):
            raise ValueError("attribution guardrails must be unique by name and scope")
        statistics = (self.mean_difference, self.bootstrap, self.permutation)
        if self.status is ScientificEvidenceStatus.INVALID:
            if any(value is not None for value in statistics):
                raise ValueError("invalid attribution evidence must not contain statistics")
            return self
        if any(value is None for value in statistics) or self.unit_count < 1:
            raise ValueError("valid attribution evidence requires complete nonempty statistics")
        assert self.mean_difference is not None
        assert self.bootstrap is not None
        assert self.permutation is not None
        if (
            self.bootstrap.sample_size != self.unit_count
            or self.permutation.sample_size != self.unit_count
            or not isclose(self.mean_difference, self.bootstrap.estimate, abs_tol=1e-15)
            or not isclose(self.mean_difference, self.permutation.observed_mean, abs_tol=1e-15)
        ):
            raise ValueError("attribution statistics do not describe the same matched units")
        expected = _classify_attribution(
            self.bootstrap,
            self.permutation,
            practical_effect_threshold=self.practical_effect_threshold,
            confidence_level=self.bootstrap.confidence_level,
            behavioral_alias=self.behavioral_alias,
            causal_guardrails=self.causal_guardrails,
        )
        if self.status is not expected:
            raise ValueError("attribution status does not match its output and causal evidence")
        return self

    @property
    def decision_gate(self) -> ScientificEvidenceGate:
        """Return the attribution-only input to the final decision truth table."""

        return ScientificEvidenceGate(
            kind=ScientificEvidenceKind.PERSISTENT_STATE_ATTRIBUTION,
            status=self.status,
            detail=self.detail,
        )


def evaluate_persistent_state_attribution(
    persistent_minus_reset_differences: Sequence[float],
    *,
    spec: AttributionAnalysisSpec,
    causal_guardrails: Sequence[ScientificGuardrailResult],
    behavioral_alias: bool = False,
) -> PersistentStateAttributionResult:
    """Evaluate signed model-output attribution outside efficacy and Holm families."""

    if not isinstance(spec, AttributionAnalysisSpec):
        raise TypeError("spec must be an AttributionAnalysisSpec")
    guardrails = tuple(causal_guardrails)
    invalid = any(guardrail.status is ScientificGuardrailStatus.INVALID for guardrail in guardrails)
    values = tuple(persistent_minus_reset_differences)
    if invalid or not guardrails or not values:
        return PersistentStateAttributionResult(
            unit_count=len(values),
            practical_effect_threshold=spec.practical_effect_threshold,
            behavioral_alias=behavioral_alias,
            causal_guardrails=guardrails,
            status=ScientificEvidenceStatus.INVALID,
            detail=(
                "persistent-state attribution failed a causal integrity guardrail"
                if invalid
                else (
                    "persistent-state attribution has no declared causal guardrails"
                    if not guardrails
                    else "persistent-state attribution has no intervention-turn matched units"
                )
            ),
        )
    try:
        bootstrap = paired_bootstrap_ci(
            values,
            resamples=spec.bootstrap_resamples,
            confidence_level=spec.confidence_level,
            seed=spec.bootstrap_seed,
        )
        permutation = paired_sign_flip_permutation_test(
            values,
            resamples=spec.permutation_resamples,
            seed=spec.permutation_seed,
        )
    except (TypeError, ValueError):
        return PersistentStateAttributionResult(
            unit_count=len(values),
            practical_effect_threshold=spec.practical_effect_threshold,
            behavioral_alias=behavioral_alias,
            causal_guardrails=guardrails,
            status=ScientificEvidenceStatus.INVALID,
            detail="persistent-state attribution contains invalid matched differences",
        )
    status = _classify_attribution(
        bootstrap,
        permutation,
        practical_effect_threshold=spec.practical_effect_threshold,
        confidence_level=spec.confidence_level,
        behavioral_alias=behavioral_alias,
        causal_guardrails=guardrails,
    )
    details = {
        ScientificEvidenceStatus.PASS: (
            "persistent state has a beneficial model-output contribution above the frozen gate"
        ),
        ScientificEvidenceStatus.DECISIVE_NEGATIVE: (
            "persistent-state output contribution is equivalent, harmful, or below the frozen gate"
        ),
        ScientificEvidenceStatus.INCONCLUSIVE: (
            "persistent-state output contribution remains statistically unresolved"
        ),
    }
    return PersistentStateAttributionResult(
        unit_count=len(values),
        mean_difference=bootstrap.estimate,
        bootstrap=bootstrap,
        permutation=permutation,
        practical_effect_threshold=spec.practical_effect_threshold,
        behavioral_alias=behavioral_alias,
        causal_guardrails=guardrails,
        status=status,
        detail=details[status],
    )


def _classify_attribution(
    bootstrap: BootstrapResult,
    permutation: PermutationTestResult,
    *,
    practical_effect_threshold: float,
    confidence_level: float,
    behavioral_alias: bool,
    causal_guardrails: Sequence[ScientificGuardrailResult],
) -> ScientificEvidenceStatus:
    if any(
        guardrail.status is ScientificGuardrailStatus.INVALID for guardrail in causal_guardrails
    ):
        return ScientificEvidenceStatus.INVALID
    if behavioral_alias or any(
        guardrail.status is ScientificGuardrailStatus.FAIL for guardrail in causal_guardrails
    ):
        return ScientificEvidenceStatus.DECISIVE_NEGATIVE
    alpha = 1.0 - confidence_level
    if (
        bootstrap.estimate >= practical_effect_threshold
        and bootstrap.lower > 0.0
        and permutation.p_value <= alpha
    ):
        return ScientificEvidenceStatus.PASS
    if bootstrap.upper < practical_effect_threshold:
        return ScientificEvidenceStatus.DECISIVE_NEGATIVE
    return ScientificEvidenceStatus.INCONCLUSIVE


__all__ = [
    "AttributionAnalysisSpec",
    "PersistentStateAttributionResult",
    "evaluate_persistent_state_attribution",
]

"""Pure domain and decision rules for confirmatory scientific closeout.

This module deliberately has no dependency on providers, storage, reporting, or
the experiment runner.  It turns already-validated evaluation evidence into one
final scientific state while preserving the narrower Phase 3 result vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isclose
from types import MappingProxyType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neurallm.domain.models import (
    FiniteFloat,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    Sha256Hex,
    SqliteInt64,
    UnitInterval,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.models import BootstrapResult, HolmAdjustedPValue, PermutationTestResult
from neurallm.evaluation.statistics import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_permutation_test,
)

type EfficacyComparatorId = Literal[
    "best_static",
    "heuristic_adaptive",
    "random_matched",
]
FOCAL_POLICY_ID: Literal["neural_persistent"] = "neural_persistent"
SERIOUS_COMPARATOR_IDS: tuple[Literal["best_static"], Literal["heuristic_adaptive"]] = (
    "best_static",
    "heuristic_adaptive",
)
NEGATIVE_CONTROL_POLICY_ID: Literal["random_matched"] = "random_matched"
ATTRIBUTION_COMPARATOR_ID: Literal["neural_matched_history_state_reset"] = (
    "neural_matched_history_state_reset"
)
EFFICACY_COMPARATOR_IDS: tuple[EfficacyComparatorId, ...] = (
    *SERIOUS_COMPARATOR_IDS,
    NEGATIVE_CONTROL_POLICY_ID,
)

REQUIRED_SCIENTIFIC_GUARDRAILS = (
    "action_bound_compliance",
    "action_saturation_rate",
    "behavioral_alias_detection",
    "instruction_adherence_non_regression",
    "matched_condition_coverage",
    "metric_availability",
    "provider_identity_stability",
    "response_length_confound",
    "turn_zero_equivalence",
)


class ScientificFrozenModel(BaseModel):
    """Strict immutable base for final scientific evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class ExperimentTier(StrEnum):
    """Run tiers with distinct claim permissions."""

    ENGINEERING_SMOKE = "engineering_smoke"
    DEVELOPMENT_PILOT = "development_pilot"
    CONFIRMATORY = "confirmatory"


class ScientificDecisionState(StrEnum):
    """The complete and exact final scientific outcome vocabulary."""

    VALIDATED_POSITIVE = "VALIDATED_POSITIVE"
    VALIDATED_NEGATIVE = "VALIDATED_NEGATIVE"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID_RUN = "INVALID_RUN"


class ScientificEvidenceStatus(StrEnum):
    """Decision-facing status of one required evidence family."""

    PASS = "pass"
    DECISIVE_NEGATIVE = "decisive_negative"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


class ScientificGuardrailStatus(StrEnum):
    """Scientific guardrail result with integrity distinct from substance."""

    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


class ScientificEvidenceKind(StrEnum):
    """Non-overlapping analysis populations entering the final decision."""

    RECOVERY = "recovery"
    PERSISTENT_STATE_ATTRIBUTION = "persistent_state_attribution"


class ComparatorRole(StrEnum):
    """Role of an independently operating efficacy comparator."""

    SERIOUS = "serious"
    NEGATIVE_CONTROL = "negative_control"


class LimitationKind(StrEnum):
    """Predeclared limitation classes relevant to decision eligibility."""

    OPTIONAL_METRIC_UNAVAILABLE = "optional_metric_unavailable"
    SUBGROUP_CONFLICT = "subgroup_conflict"
    OTHER = "other"


class LimitationDisposition(StrEnum):
    """Whether a limitation is disclosure-only or decision-blocking."""

    DISCLOSURE_ONLY = "disclosure_only"
    INCONCLUSIVE = "inconclusive"


class ScientificReasonCode(StrEnum):
    """Stable machine-readable reasons for a final decision."""

    ALL_POSITIVE_GATES_PASSED = "all_positive_gates_passed"
    ATTRIBUTION_FAILED = "attribution_failed"
    ATTRIBUTION_INVALID = "attribution_invalid"
    ATTRIBUTION_UNRESOLVED = "attribution_unresolved"
    EFFICACY_INVALID = "efficacy_invalid"
    EFFICACY_UNRESOLVED = "efficacy_unresolved"
    GUARDRAIL_FAILED = "guardrail_failed"
    GUARDRAIL_INVALID = "guardrail_invalid"
    LIMITATION_REQUIRES_INCONCLUSIVE = "limitation_requires_inconclusive"
    MISSING_REQUIRED_GUARDRAIL = "missing_required_guardrail"
    NEGATIVE_CONTROL_SANITY_FAILED = "negative_control_sanity_failed"
    OPTIONAL_METRIC_UNAVAILABLE = "optional_metric_unavailable"
    RECOVERY_FAILED = "recovery_failed"
    RECOVERY_INVALID = "recovery_invalid"
    RECOVERY_UNRESOLVED = "recovery_unresolved"
    REQUIRED_COMPARATOR_FAILED = "required_comparator_failed"
    SUBGROUP_CONFLICT = "subgroup_conflict"


class GuardrailCleanTaskScore(ScientificFrozenModel):
    """Raw task score together with its explicit, non-blended gate.

    A failed or invalid gate makes ``gated_value`` unavailable.  The raw score
    remains present for audit and is never replaced with zero.
    """

    metric_name: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    raw_task_score: UnitInterval
    gate_status: ScientificGuardrailStatus
    gate_names: tuple[NonEmptyString, ...]

    @field_validator("gate_status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificGuardrailStatus(value)
        return value

    @field_validator("gate_names", mode="before")
    @classmethod
    def _accept_gate_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("gate_names")
    @classmethod
    def _canonical_gate_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(sorted(set(values)))
        if not canonical or canonical != tuple(sorted(values)) or len(canonical) != len(values):
            raise ValueError("gate_names must be nonempty, sorted, and unique")
        return canonical

    @property
    def gated_value(self) -> float | None:
        """Return the unmodified task score only when every declared gate passes."""

        if self.gate_status is ScientificGuardrailStatus.PASS:
            return self.raw_task_score
        return None


def guardrail_clean_task_difference(
    focal: GuardrailCleanTaskScore,
    comparator: GuardrailCleanTaskScore,
) -> float | None:
    """Return the raw focal-minus-comparator score only when both gates pass.

    ``None`` is preserved when either side is not clean; this helper never drops
    a unit, substitutes zero, or blends guardrail values into the task score.
    """

    if not isinstance(focal, GuardrailCleanTaskScore) or not isinstance(
        comparator,
        GuardrailCleanTaskScore,
    ):
        raise TypeError("guardrail-clean differences require typed task scores")
    if focal.gated_value is None or comparator.gated_value is None:
        return None
    return focal.raw_task_score - comparator.raw_task_score


class ScientificGuardrailResult(ScientificFrozenModel):
    """One explicit integrity or substantive guardrail result."""

    name: NonEmptyString
    status: ScientificGuardrailStatus
    scope: NonEmptyString
    detail: NonEmptyString
    observed_value: FiniteFloat | None = None
    threshold: FiniteFloat | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificGuardrailStatus(value)
        return value

    @property
    def evidence_key(self) -> tuple[str, str]:
        """Return the canonical key allowing one named guardrail per scope."""

        return self.name, self.scope


class ScientificLimitation(ScientificFrozenModel):
    """One disclosed limitation and its preregistered decision effect."""

    kind: LimitationKind
    code: NonEmptyString
    detail: NonEmptyString
    disposition: LimitationDisposition

    @field_validator("kind", mode="before")
    @classmethod
    def _accept_serialized_kind(cls, value: object) -> object:
        if isinstance(value, str):
            return LimitationKind(value)
        return value

    @field_validator("disposition", mode="before")
    @classmethod
    def _accept_serialized_disposition(cls, value: object) -> object:
        if isinstance(value, str):
            return LimitationDisposition(value)
        return value


class ScientificEvidenceGate(ScientificFrozenModel):
    """Compact decision input emitted by a full evidence result."""

    kind: ScientificEvidenceKind
    status: ScientificEvidenceStatus
    detail: NonEmptyString

    @field_validator("kind", mode="before")
    @classmethod
    def _accept_serialized_kind(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceKind(value)
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceStatus(value)
        return value


class EfficacyAnalysisSpec(ScientificFrozenModel):
    """Frozen statistics and exact roles for end-to-end efficacy."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["confirmatory-efficacy-v1"] = "confirmatory-efficacy-v1"
    primary_metric: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    focal_policy_id: Literal["neural_persistent"] = FOCAL_POLICY_ID
    serious_comparator_ids: tuple[Literal["best_static"], Literal["heuristic_adaptive"]] = (
        SERIOUS_COMPARATOR_IDS
    )
    negative_control_policy_id: Literal["random_matched"] = NEGATIVE_CONTROL_POLICY_ID
    practical_effect_threshold: float = Field(default=0.02, gt=0.0, allow_inf_nan=False)
    bootstrap_resamples: PositiveInt = 10_000
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, allow_inf_nan=False)
    bootstrap_seed: SqliteInt64
    permutation_resamples: PositiveInt = 10_000
    permutation_seed: SqliteInt64

    @field_validator("serious_comparator_ids", mode="before")
    @classmethod
    def _accept_serious_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _require_exact_roles(self) -> Self:
        if self.serious_comparator_ids != SERIOUS_COMPARATOR_IDS:
            raise ValueError(
                "serious comparators must be exactly best_static and heuristic_adaptive"
            )
        return self


class EfficacyComparisonResult(ScientificFrozenModel):
    """One independent-history neural efficacy comparison."""

    comparison_kind: Literal["efficacy"] = "efficacy"
    primary_metric: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    focal_policy_id: Literal["neural_persistent"] = FOCAL_POLICY_ID
    comparator_policy_id: EfficacyComparatorId
    comparator_role: ComparatorRole
    included_in_holm_family: bool
    unit_count: NonNegativeInt
    mean_difference: FiniteFloat | None = None
    bootstrap: BootstrapResult | None = None
    permutation: PermutationTestResult | None = None
    holm: HolmAdjustedPValue | None = None
    practical_effect_threshold: float = Field(gt=0.0, allow_inf_nan=False)
    behavioral_alias: bool = False
    guardrails: tuple[ScientificGuardrailResult, ...] = ()
    status: ScientificEvidenceStatus
    detail: NonEmptyString

    @field_validator("comparator_role", mode="before")
    @classmethod
    def _accept_serialized_role(cls, value: object) -> object:
        if isinstance(value, str):
            return ComparatorRole(value)
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceStatus(value)
        return value

    @field_validator("guardrails", mode="before")
    @classmethod
    def _accept_guardrail_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("guardrails")
    @classmethod
    def _canonical_guardrails(
        cls,
        values: tuple[ScientificGuardrailResult, ...],
    ) -> tuple[ScientificGuardrailResult, ...]:
        return tuple(sorted(values, key=lambda value: value.evidence_key))

    @model_validator(mode="after")
    def _validate_role_statistics_and_status(self) -> Self:
        serious = self.comparator_policy_id in SERIOUS_COMPARATOR_IDS
        expected_role = ComparatorRole.SERIOUS if serious else ComparatorRole.NEGATIVE_CONTROL
        if self.comparator_role is not expected_role:
            raise ValueError("comparator role does not match the fixed efficacy policy role")
        if self.included_in_holm_family != serious:
            raise ValueError("Holm-family membership must include only serious comparators")
        keys = tuple(guardrail.evidence_key for guardrail in self.guardrails)
        if len(keys) != len(set(keys)):
            raise ValueError("comparison guardrails must be unique by name and scope")

        statistics = (self.mean_difference, self.bootstrap, self.permutation)
        if self.status is ScientificEvidenceStatus.INVALID:
            if any(value is not None for value in statistics) or self.holm is not None:
                raise ValueError(
                    "invalid efficacy evidence must not contain inferential statistics"
                )
            return self
        if any(value is None for value in statistics) or self.unit_count < 1:
            raise ValueError("valid efficacy evidence requires complete nonempty statistics")
        assert self.mean_difference is not None
        assert self.bootstrap is not None
        assert self.permutation is not None
        if (
            self.bootstrap.sample_size != self.unit_count
            or self.permutation.sample_size != self.unit_count
            or not isclose(self.mean_difference, self.bootstrap.estimate, abs_tol=1e-15)
            or not isclose(self.mean_difference, self.permutation.observed_mean, abs_tol=1e-15)
        ):
            raise ValueError("efficacy statistics do not describe the same matched units")
        if serious:
            if self.holm is None or self.holm.comparator_policy_id != self.comparator_policy_id:
                raise ValueError("serious efficacy comparison requires its Holm result")
            if self.holm.family_size != len(SERIOUS_COMPARATOR_IDS) or not isclose(
                self.holm.raw_p_value,
                self.permutation.p_value,
                abs_tol=1e-15,
            ):
                raise ValueError("serious efficacy Holm evidence must match its raw test")
        elif self.holm is not None:
            raise ValueError("negative control must be excluded from Holm correction")
        expected_status = _classify_efficacy(
            bootstrap=self.bootstrap,
            permutation=self.permutation,
            holm=self.holm,
            practical_effect_threshold=self.practical_effect_threshold,
            confidence_level=self.bootstrap.confidence_level,
            behavioral_alias=self.behavioral_alias,
            guardrails=self.guardrails,
        )
        if self.status is not expected_status:
            raise ValueError(
                "efficacy status does not match its statistical and guardrail evidence"
            )
        return self


class ScientificDecisionInput(ScientificFrozenModel):
    """Complete decision-facing evidence for one confirmatory analysis."""

    tier: ExperimentTier
    efficacy_comparisons: tuple[EfficacyComparisonResult, ...]
    recovery: ScientificEvidenceGate
    attribution: ScientificEvidenceGate
    guardrails: tuple[ScientificGuardrailResult, ...]
    limitations: tuple[ScientificLimitation, ...] = ()
    required_guardrail_names: tuple[NonEmptyString, ...] = REQUIRED_SCIENTIFIC_GUARDRAILS

    @field_validator("tier", mode="before")
    @classmethod
    def _accept_serialized_tier(cls, value: object) -> object:
        if isinstance(value, str):
            return ExperimentTier(value)
        return value

    @field_validator(
        "efficacy_comparisons",
        "guardrails",
        "limitations",
        "required_guardrail_names",
        mode="before",
    )
    @classmethod
    def _accept_sequence_lists(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("required_guardrail_names")
    @classmethod
    def _canonical_required_guardrails(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != REQUIRED_SCIENTIFIC_GUARDRAILS:
            raise ValueError("required guardrail names must match the frozen scientific contract")
        return values

    @field_validator("guardrails")
    @classmethod
    def _canonical_guardrails(
        cls,
        values: tuple[ScientificGuardrailResult, ...],
    ) -> tuple[ScientificGuardrailResult, ...]:
        return tuple(sorted(values, key=lambda value: value.evidence_key))

    @field_validator("limitations")
    @classmethod
    def _canonical_limitations(
        cls,
        values: tuple[ScientificLimitation, ...],
    ) -> tuple[ScientificLimitation, ...]:
        return tuple(sorted(values, key=lambda value: (value.kind.value, value.code)))

    @model_validator(mode="after")
    def _validate_evidence_roles(self) -> Self:
        comparison_ids = tuple(
            comparison.comparator_policy_id for comparison in self.efficacy_comparisons
        )
        if comparison_ids != EFFICACY_COMPARATOR_IDS:
            raise ValueError("efficacy comparisons must contain the exact canonical comparator set")
        unit_counts = {comparison.unit_count for comparison in self.efficacy_comparisons}
        if (
            not any(
                comparison.status is ScientificEvidenceStatus.INVALID
                for comparison in self.efficacy_comparisons
            )
            and len(unit_counts) != 1
        ):
            raise ValueError("efficacy comparisons must share exact matched-unit coverage")
        if self.recovery.kind is not ScientificEvidenceKind.RECOVERY:
            raise ValueError("recovery gate has the wrong evidence kind")
        if self.attribution.kind is not ScientificEvidenceKind.PERSISTENT_STATE_ATTRIBUTION:
            raise ValueError("attribution gate has the wrong evidence kind")
        guardrail_keys = tuple(guardrail.evidence_key for guardrail in self.guardrails)
        if len(guardrail_keys) != len(set(guardrail_keys)):
            raise ValueError("scientific guardrails must be unique by name and scope")
        limitation_keys = tuple(
            (limitation.kind, limitation.code) for limitation in self.limitations
        )
        if len(limitation_keys) != len(set(limitation_keys)):
            raise ValueError("scientific limitations must be unique by kind and code")
        return self


class ScientificDecisionRecord(ScientificFrozenModel):
    """Canonical final state and the evidence reasons that selected it."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["scientific-decision-v1"] = "scientific-decision-v1"
    claim_scope: Literal["confirmatory-model-backed-scientific-decision"] = (
        "confirmatory-model-backed-scientific-decision"
    )
    tier: Literal[ExperimentTier.CONFIRMATORY] = ExperimentTier.CONFIRMATORY
    decision: ScientificDecisionState
    reason_codes: tuple[ScientificReasonCode, ...]
    decision_input_sha256: Sha256Hex
    limitations: tuple[ScientificLimitation, ...] = ()

    @field_validator("decision", mode="before")
    @classmethod
    def _accept_serialized_decision(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificDecisionState(value)
        return value

    @field_validator("tier", mode="before")
    @classmethod
    def _accept_serialized_tier(cls, value: object) -> object:
        if isinstance(value, str):
            return ExperimentTier(value)
        return value

    @field_validator("limitations", mode="before")
    @classmethod
    def _accept_limitation_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("limitations")
    @classmethod
    def _canonical_limitations(
        cls,
        values: tuple[ScientificLimitation, ...],
    ) -> tuple[ScientificLimitation, ...]:
        return tuple(sorted(values, key=lambda value: (value.kind.value, value.code)))

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _accept_serialized_reason_codes(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(
                ScientificReasonCode(item) if isinstance(item, str) else item for item in value
            )
        return value

    @field_validator("reason_codes")
    @classmethod
    def _canonical_reason_codes(
        cls,
        values: tuple[ScientificReasonCode, ...],
    ) -> tuple[ScientificReasonCode, ...]:
        canonical = tuple(sorted(set(values), key=lambda value: value.value))
        if not canonical:
            raise ValueError("a scientific decision requires at least one reason code")
        if values != canonical:
            raise ValueError("scientific reason codes must be sorted and unique")
        return values

    @model_validator(mode="after")
    def _validate_reason_semantics(self) -> Self:
        allowed = {
            ScientificDecisionState.VALIDATED_POSITIVE: {
                ScientificReasonCode.ALL_POSITIVE_GATES_PASSED,
            },
            ScientificDecisionState.VALIDATED_NEGATIVE: {
                ScientificReasonCode.ATTRIBUTION_FAILED,
                ScientificReasonCode.GUARDRAIL_FAILED,
                ScientificReasonCode.NEGATIVE_CONTROL_SANITY_FAILED,
                ScientificReasonCode.RECOVERY_FAILED,
                ScientificReasonCode.REQUIRED_COMPARATOR_FAILED,
            },
            ScientificDecisionState.INCONCLUSIVE: {
                ScientificReasonCode.ATTRIBUTION_UNRESOLVED,
                ScientificReasonCode.EFFICACY_UNRESOLVED,
                ScientificReasonCode.LIMITATION_REQUIRES_INCONCLUSIVE,
                ScientificReasonCode.OPTIONAL_METRIC_UNAVAILABLE,
                ScientificReasonCode.RECOVERY_UNRESOLVED,
                ScientificReasonCode.SUBGROUP_CONFLICT,
            },
            ScientificDecisionState.INVALID_RUN: {
                ScientificReasonCode.ATTRIBUTION_INVALID,
                ScientificReasonCode.EFFICACY_INVALID,
                ScientificReasonCode.GUARDRAIL_INVALID,
                ScientificReasonCode.MISSING_REQUIRED_GUARDRAIL,
                ScientificReasonCode.RECOVERY_INVALID,
            },
        }
        reason_set = set(self.reason_codes)
        if not reason_set.issubset(allowed[self.decision]):
            raise ValueError("scientific reason codes do not match the declared decision")
        if (
            self.decision is ScientificDecisionState.VALIDATED_POSITIVE
            and reason_set != allowed[self.decision]
        ):
            raise ValueError("a positive decision requires the exact all-gates-passed reason")
        return self


class FinalDecisionIneligibleError(ValueError):
    """Raised when a smoke or pilot attempts to emit a final scientific state."""


def _classify_efficacy(
    *,
    bootstrap: BootstrapResult,
    permutation: PermutationTestResult,
    holm: HolmAdjustedPValue | None,
    practical_effect_threshold: float,
    confidence_level: float,
    behavioral_alias: bool,
    guardrails: Sequence[ScientificGuardrailResult],
) -> ScientificEvidenceStatus:
    if any(guardrail.status is ScientificGuardrailStatus.INVALID for guardrail in guardrails):
        return ScientificEvidenceStatus.INVALID
    if behavioral_alias or any(
        guardrail.status is ScientificGuardrailStatus.FAIL for guardrail in guardrails
    ):
        return ScientificEvidenceStatus.DECISIVE_NEGATIVE
    p_value = holm.adjusted_p_value if holm is not None else permutation.p_value
    alpha = 1.0 - confidence_level
    if (
        bootstrap.estimate >= practical_effect_threshold
        and bootstrap.lower > 0.0
        and p_value <= alpha
    ):
        return ScientificEvidenceStatus.PASS
    if bootstrap.upper < practical_effect_threshold:
        return ScientificEvidenceStatus.DECISIVE_NEGATIVE
    return ScientificEvidenceStatus.INCONCLUSIVE


def _guardrails_for(
    comparator_policy_id: str,
    guardrails_by_comparator: Mapping[str, Sequence[ScientificGuardrailResult]],
) -> tuple[ScientificGuardrailResult, ...]:
    return tuple(guardrails_by_comparator.get(comparator_policy_id, ()))


def _invalid_efficacy_results(
    values: Mapping[EfficacyComparatorId, Sequence[float]],
    *,
    spec: EfficacyAnalysisSpec,
    guardrails: Mapping[str, Sequence[ScientificGuardrailResult]],
    aliases: Mapping[str, bool],
    detail: str,
) -> tuple[EfficacyComparisonResult, ...]:
    return tuple(
        EfficacyComparisonResult(
            comparator_policy_id=comparator_id,
            comparator_role=(
                ComparatorRole.SERIOUS
                if comparator_id in SERIOUS_COMPARATOR_IDS
                else ComparatorRole.NEGATIVE_CONTROL
            ),
            included_in_holm_family=comparator_id in SERIOUS_COMPARATOR_IDS,
            unit_count=len(values[comparator_id]),
            practical_effect_threshold=spec.practical_effect_threshold,
            behavioral_alias=aliases.get(comparator_id, False),
            guardrails=_guardrails_for(comparator_id, guardrails),
            status=ScientificEvidenceStatus.INVALID,
            detail=detail,
        )
        for comparator_id in EFFICACY_COMPARATOR_IDS
    )


def evaluate_efficacy_comparisons(
    paired_differences: Mapping[str, Sequence[float]],
    *,
    spec: EfficacyAnalysisSpec,
    guardrails_by_comparator: Mapping[str, Sequence[ScientificGuardrailResult]] | None = None,
    behavioral_alias_by_comparator: Mapping[str, bool] | None = None,
) -> tuple[EfficacyComparisonResult, ...]:
    """Evaluate the exact three efficacy comparisons under one frozen contract.

    The matched-reset policy is unrepresentable here.  It has its own typed
    attribution result and never enters the efficacy or Holm families.
    """

    if not isinstance(spec, EfficacyAnalysisSpec):
        raise TypeError("spec must be an EfficacyAnalysisSpec")
    if set(paired_differences) != set(EFFICACY_COMPARATOR_IDS):
        raise ValueError("paired differences must contain exactly the efficacy comparators")
    guardrails = guardrails_by_comparator or MappingProxyType({})
    aliases = behavioral_alias_by_comparator or MappingProxyType({})
    if not set(guardrails).issubset(EFFICACY_COMPARATOR_IDS):
        raise ValueError("guardrails target a non-efficacy comparator")
    if not set(aliases).issubset(EFFICACY_COMPARATOR_IDS):
        raise ValueError("behavioral aliases target a non-efficacy comparator")
    if any(not isinstance(value, bool) for value in aliases.values()):
        raise TypeError("behavioral alias values must be booleans")
    values = {
        comparator_id: tuple(paired_differences[comparator_id])
        for comparator_id in EFFICACY_COMPARATOR_IDS
    }
    unit_counts = {len(differences) for differences in values.values()}
    if len(unit_counts) != 1 or unit_counts == {0}:
        return _invalid_efficacy_results(
            values,
            spec=spec,
            guardrails=guardrails,
            aliases=aliases,
            detail="efficacy comparisons lack equal nonempty matched-unit coverage",
        )

    any_invalid = any(
        guardrail.status is ScientificGuardrailStatus.INVALID
        for comparator_id in EFFICACY_COMPARATOR_IDS
        for guardrail in _guardrails_for(comparator_id, guardrails)
    )
    if any_invalid:
        return _invalid_efficacy_results(
            values,
            spec=spec,
            guardrails=guardrails,
            aliases=aliases,
            detail="efficacy statistics are suppressed by an invalid integrity guardrail",
        )

    drafts: dict[str, tuple[BootstrapResult, PermutationTestResult]] = {}
    try:
        for comparator_id in EFFICACY_COMPARATOR_IDS:
            differences = values[comparator_id]
            drafts[comparator_id] = (
                paired_bootstrap_ci(
                    differences,
                    resamples=spec.bootstrap_resamples,
                    confidence_level=spec.confidence_level,
                    seed=spec.bootstrap_seed,
                ),
                paired_sign_flip_permutation_test(
                    differences,
                    resamples=spec.permutation_resamples,
                    seed=spec.permutation_seed,
                ),
            )
    except (TypeError, ValueError):
        return _invalid_efficacy_results(
            values,
            spec=spec,
            guardrails=guardrails,
            aliases=aliases,
            detail="efficacy evidence contains nonfinite or otherwise invalid differences",
        )
    holm_by_comparator = {
        result.comparator_policy_id: result
        for result in holm_adjust(
            {
                comparator_id: drafts[comparator_id][1].p_value
                for comparator_id in SERIOUS_COMPARATOR_IDS
            }
        )
    }
    results: list[EfficacyComparisonResult] = []
    for comparator_id in EFFICACY_COMPARATOR_IDS:
        bootstrap, permutation = drafts[comparator_id]
        comparison_guardrails = _guardrails_for(comparator_id, guardrails)
        holm = holm_by_comparator.get(comparator_id)
        results.append(
            EfficacyComparisonResult(
                comparator_policy_id=comparator_id,
                comparator_role=(
                    ComparatorRole.SERIOUS
                    if comparator_id in SERIOUS_COMPARATOR_IDS
                    else ComparatorRole.NEGATIVE_CONTROL
                ),
                included_in_holm_family=comparator_id in SERIOUS_COMPARATOR_IDS,
                unit_count=len(values[comparator_id]),
                mean_difference=bootstrap.estimate,
                bootstrap=bootstrap,
                permutation=permutation,
                holm=holm,
                practical_effect_threshold=spec.practical_effect_threshold,
                behavioral_alias=aliases.get(comparator_id, False),
                guardrails=comparison_guardrails,
                status=_classify_efficacy(
                    bootstrap=bootstrap,
                    permutation=permutation,
                    holm=holm,
                    practical_effect_threshold=spec.practical_effect_threshold,
                    confidence_level=spec.confidence_level,
                    behavioral_alias=aliases.get(comparator_id, False),
                    guardrails=comparison_guardrails,
                ),
                detail=(
                    "efficacy comparison is classified by frozen practical and statistical gates"
                ),
            )
        )
    return tuple(results)


def decide_scientific_outcome(evidence: ScientificDecisionInput) -> ScientificDecisionRecord:
    """Apply the frozen precedence: invalid, negative, inconclusive, positive."""

    if not isinstance(evidence, ScientificDecisionInput):
        raise TypeError("evidence must be a ScientificDecisionInput")
    if evidence.tier is not ExperimentTier.CONFIRMATORY:
        raise FinalDecisionIneligibleError(
            "engineering smoke and development pilot runs cannot emit a final decision"
        )

    observed_guardrail_names = {guardrail.name for guardrail in evidence.guardrails}
    missing_guardrails = set(evidence.required_guardrail_names) - observed_guardrail_names
    invalid_reasons: set[ScientificReasonCode] = set()
    if missing_guardrails:
        invalid_reasons.add(ScientificReasonCode.MISSING_REQUIRED_GUARDRAIL)
    if any(
        guardrail.status is ScientificGuardrailStatus.INVALID for guardrail in evidence.guardrails
    ):
        invalid_reasons.add(ScientificReasonCode.GUARDRAIL_INVALID)
    if any(
        comparison.status is ScientificEvidenceStatus.INVALID
        for comparison in evidence.efficacy_comparisons
    ):
        invalid_reasons.add(ScientificReasonCode.EFFICACY_INVALID)
    if evidence.recovery.status is ScientificEvidenceStatus.INVALID:
        invalid_reasons.add(ScientificReasonCode.RECOVERY_INVALID)
    if evidence.attribution.status is ScientificEvidenceStatus.INVALID:
        invalid_reasons.add(ScientificReasonCode.ATTRIBUTION_INVALID)
    if invalid_reasons:
        return _decision_record(
            evidence,
            ScientificDecisionState.INVALID_RUN,
            invalid_reasons,
        )

    negative_reasons: set[ScientificReasonCode] = set()
    if any(guardrail.status is ScientificGuardrailStatus.FAIL for guardrail in evidence.guardrails):
        negative_reasons.add(ScientificReasonCode.GUARDRAIL_FAILED)
    serious = tuple(
        comparison
        for comparison in evidence.efficacy_comparisons
        if comparison.comparator_role is ComparatorRole.SERIOUS
    )
    if any(
        comparison.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE for comparison in serious
    ):
        negative_reasons.add(ScientificReasonCode.REQUIRED_COMPARATOR_FAILED)
    negative_control = next(
        comparison
        for comparison in evidence.efficacy_comparisons
        if comparison.comparator_role is ComparatorRole.NEGATIVE_CONTROL
    )
    if negative_control.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE:
        negative_reasons.add(ScientificReasonCode.NEGATIVE_CONTROL_SANITY_FAILED)
    if evidence.recovery.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE:
        negative_reasons.add(ScientificReasonCode.RECOVERY_FAILED)
    if evidence.attribution.status is ScientificEvidenceStatus.DECISIVE_NEGATIVE:
        negative_reasons.add(ScientificReasonCode.ATTRIBUTION_FAILED)
    if negative_reasons:
        return _decision_record(
            evidence,
            ScientificDecisionState.VALIDATED_NEGATIVE,
            negative_reasons,
        )

    inconclusive_reasons: set[ScientificReasonCode] = set()
    if any(comparison.status is ScientificEvidenceStatus.INCONCLUSIVE for comparison in serious):
        inconclusive_reasons.add(ScientificReasonCode.EFFICACY_UNRESOLVED)
    if evidence.recovery.status is ScientificEvidenceStatus.INCONCLUSIVE:
        inconclusive_reasons.add(ScientificReasonCode.RECOVERY_UNRESOLVED)
    if evidence.attribution.status is ScientificEvidenceStatus.INCONCLUSIVE:
        inconclusive_reasons.add(ScientificReasonCode.ATTRIBUTION_UNRESOLVED)
    for limitation in evidence.limitations:
        if limitation.disposition is not LimitationDisposition.INCONCLUSIVE:
            continue
        inconclusive_reasons.add(ScientificReasonCode.LIMITATION_REQUIRES_INCONCLUSIVE)
        if limitation.kind is LimitationKind.SUBGROUP_CONFLICT:
            inconclusive_reasons.add(ScientificReasonCode.SUBGROUP_CONFLICT)
        elif limitation.kind is LimitationKind.OPTIONAL_METRIC_UNAVAILABLE:
            inconclusive_reasons.add(ScientificReasonCode.OPTIONAL_METRIC_UNAVAILABLE)
    if inconclusive_reasons:
        return _decision_record(
            evidence,
            ScientificDecisionState.INCONCLUSIVE,
            inconclusive_reasons,
        )

    if not all(comparison.status is ScientificEvidenceStatus.PASS for comparison in serious):
        raise ValueError("confirmatory evidence reached an unclassified serious-comparator state")
    if (
        evidence.recovery.status is not ScientificEvidenceStatus.PASS
        or evidence.attribution.status is not ScientificEvidenceStatus.PASS
        or any(
            guardrail.status is not ScientificGuardrailStatus.PASS
            for guardrail in evidence.guardrails
        )
    ):
        raise ValueError("confirmatory evidence reached an unclassified required gate state")
    return _decision_record(
        evidence,
        ScientificDecisionState.VALIDATED_POSITIVE,
        {ScientificReasonCode.ALL_POSITIVE_GATES_PASSED},
    )


def _decision_record(
    evidence: ScientificDecisionInput,
    decision: ScientificDecisionState,
    reason_codes: set[ScientificReasonCode],
) -> ScientificDecisionRecord:
    return ScientificDecisionRecord(
        decision=decision,
        reason_codes=tuple(sorted(reason_codes, key=lambda value: value.value)),
        decision_input_sha256=canonical_sha256(evidence),
        limitations=evidence.limitations,
    )


__all__ = [
    "ATTRIBUTION_COMPARATOR_ID",
    "EFFICACY_COMPARATOR_IDS",
    "FOCAL_POLICY_ID",
    "NEGATIVE_CONTROL_POLICY_ID",
    "REQUIRED_SCIENTIFIC_GUARDRAILS",
    "SERIOUS_COMPARATOR_IDS",
    "ComparatorRole",
    "EfficacyAnalysisSpec",
    "EfficacyComparisonResult",
    "ExperimentTier",
    "FinalDecisionIneligibleError",
    "GuardrailCleanTaskScore",
    "LimitationDisposition",
    "LimitationKind",
    "ScientificDecisionInput",
    "ScientificDecisionRecord",
    "ScientificDecisionState",
    "ScientificEvidenceGate",
    "ScientificEvidenceKind",
    "ScientificEvidenceStatus",
    "ScientificFrozenModel",
    "ScientificGuardrailResult",
    "ScientificGuardrailStatus",
    "ScientificLimitation",
    "ScientificReasonCode",
    "decide_scientific_outcome",
    "evaluate_efficacy_comparisons",
    "guardrail_clean_task_difference",
]

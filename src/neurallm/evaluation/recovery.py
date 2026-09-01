"""Output-based recovery endpoints for preregistered stressor transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isclose, isfinite
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from neurallm.domain.models import FiniteFloat, NonNegativeInt, PositiveInt, SqliteInt64
from neurallm.evaluation.models import BootstrapResult
from neurallm.evaluation.scientific import (
    DEFAULT_VALIDATED_NEGATIVE_MULTIPLICITY,
    SERIOUS_COMPARATOR_IDS,
    NegativeSideEvidence,
    ScientificEvidenceGate,
    ScientificEvidenceKind,
    ScientificEvidenceStatus,
    ScientificFrozenModel,
    ValidatedNegativeMultiplicitySpec,
    negative_side_evidence,
)
from neurallm.evaluation.statistics import paired_bootstrap_ci


class RecoveryMetricName(StrEnum):
    """The three required output-based recovery endpoints."""

    POST_STRESSOR_TASK_SCORE_CHANGE = "post_stressor_task_score_change"
    POST_STRESSOR_REPETITION_CHANGE = "post_stressor_repetition_change"
    TIME_TO_RETURN_TO_TARGET_BAND = "time_to_return_to_target_band"


RECOVERY_METRIC_NAMES = tuple(RecoveryMetricName)


class RecoveryAnalysisSpec(ScientificFrozenModel):
    """Frozen practical thresholds and deterministic bootstrap settings."""

    schema_version: Literal[1] = 1
    implementation_version: Literal["output-recovery-v2"] = "output-recovery-v2"
    serious_comparator_ids: tuple[
        Literal["best_static"],
        Literal["heuristic_adaptive"],
    ] = SERIOUS_COMPARATOR_IDS
    comparator_reduction_version: Literal["per-unit-minimum-serious-comparator-margin-v1"] = (
        "per-unit-minimum-serious-comparator-margin-v1"
    )
    no_return_handling_version: Literal["right-censored-window-plus-one-v1"] = (
        "right-censored-window-plus-one-v1"
    )
    practical_thresholds: Mapping[RecoveryMetricName, FiniteFloat]
    bootstrap_resamples: PositiveInt = 10_000
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, allow_inf_nan=False)
    bootstrap_seed: SqliteInt64

    @field_validator("serious_comparator_ids", mode="before")
    @classmethod
    def _accept_comparator_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("serious_comparator_ids")
    @classmethod
    def _require_exact_comparators(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if values != SERIOUS_COMPARATOR_IDS:
            raise ValueError("recovery requires the exact serious comparator set")
        return values

    @field_validator("practical_thresholds", mode="before")
    @classmethod
    def _accept_serialized_thresholds(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                RecoveryMetricName(key) if isinstance(key, str) else key: item
                for key, item in value.items()
            }
        return value

    @field_validator("practical_thresholds")
    @classmethod
    def _freeze_exact_thresholds(
        cls,
        values: Mapping[RecoveryMetricName, float],
    ) -> Mapping[RecoveryMetricName, float]:
        if set(values) != set(RECOVERY_METRIC_NAMES):
            raise ValueError("recovery thresholds must cover exactly the three required metrics")
        if any(value < 0.0 for value in values.values()):
            raise ValueError("recovery practical thresholds must be nonnegative")
        return MappingProxyType(dict(sorted(values.items(), key=lambda item: item[0].value)))

    @field_serializer("practical_thresholds")
    def _serialize_thresholds(
        self,
        values: Mapping[RecoveryMetricName, float],
    ) -> dict[str, float]:
        return {key.value: value for key, value in values.items()}


class RecoveryMetricResult(ScientificFrozenModel):
    """Matched-unit evidence for one recovery endpoint.

    Every difference is oriented so a positive value favors the neural policy.
    For time-to-band this means comparator turns minus neural turns.
    """

    metric_name: RecoveryMetricName
    improvement_direction: Literal["positive_favors_neural"] = "positive_favors_neural"
    unit_count: PositiveInt
    estimate: FiniteFloat
    bootstrap: BootstrapResult
    negative_side_evidence: NegativeSideEvidence
    practical_effect_threshold: float = Field(ge=0.0, allow_inf_nan=False)
    status: ScientificEvidenceStatus

    @field_validator("metric_name", mode="before")
    @classmethod
    def _accept_serialized_metric(cls, value: object) -> object:
        if isinstance(value, str):
            return RecoveryMetricName(value)
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceStatus(value)
        return value

    @model_validator(mode="after")
    def _validate_statistics_and_status(self) -> Self:
        if (
            self.bootstrap.sample_size != self.unit_count
            or self.negative_side_evidence.bootstrap.sample_size != self.unit_count
            or not isclose(self.estimate, self.bootstrap.estimate, abs_tol=1e-15)
            or not isclose(
                self.estimate,
                self.negative_side_evidence.bootstrap.estimate,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("recovery result statistics do not describe the same matched units")
        if (
            self.negative_side_evidence.gate_id != f"recovery:{self.metric_name.value}"
            or not isclose(
                self.negative_side_evidence.practical_effect_threshold,
                self.practical_effect_threshold,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("recovery negative-side evidence targets the wrong frozen gate")
        expected = _classify_recovery_metric(
            self.bootstrap,
            self.negative_side_evidence,
            self.practical_effect_threshold,
        )
        if self.status is not expected:
            raise ValueError("recovery metric status does not match its confidence evidence")
        return self


class RecoveryEvaluationResult(ScientificFrozenModel):
    """Complete required recovery evidence or one explicit invalid result."""

    evidence_kind: Literal[ScientificEvidenceKind.RECOVERY] = ScientificEvidenceKind.RECOVERY
    implementation_version: Literal["output-recovery-v2"] = "output-recovery-v2"
    metric_results: tuple[RecoveryMetricResult, ...] = ()
    right_censored_focal_units: NonNegativeInt = 0
    right_censored_comparator_units: NonNegativeInt = 0
    status: ScientificEvidenceStatus
    detail: str = Field(min_length=1)

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def _accept_serialized_kind(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceKind(value)
        return value

    @field_validator("metric_results", mode="before")
    @classmethod
    def _accept_result_list(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _accept_serialized_status(cls, value: object) -> object:
        if isinstance(value, str):
            return ScientificEvidenceStatus(value)
        return value

    @model_validator(mode="after")
    def _validate_result_family(self) -> Self:
        if self.status is ScientificEvidenceStatus.INVALID:
            if self.metric_results:
                raise ValueError("invalid recovery evidence must not contain partial statistics")
            return self
        names = tuple(result.metric_name for result in self.metric_results)
        if names != RECOVERY_METRIC_NAMES:
            raise ValueError("recovery result requires the exact canonical metric set")
        statuses = {result.status for result in self.metric_results}
        if ScientificEvidenceStatus.INVALID in statuses:
            raise ValueError("invalid recovery metrics require an invalid family result")
        if self.right_censored_focal_units > 0:
            expected = ScientificEvidenceStatus.DECISIVE_NEGATIVE
        elif ScientificEvidenceStatus.DECISIVE_NEGATIVE in statuses:
            expected = ScientificEvidenceStatus.DECISIVE_NEGATIVE
        elif ScientificEvidenceStatus.INCONCLUSIVE in statuses:
            expected = ScientificEvidenceStatus.INCONCLUSIVE
        else:
            expected = ScientificEvidenceStatus.PASS
        if self.status is not expected:
            raise ValueError("recovery family status does not match its metric results")
        return self

    @property
    def decision_gate(self) -> ScientificEvidenceGate:
        """Return the compact input consumed by the final decision truth table."""

        return ScientificEvidenceGate(
            kind=ScientificEvidenceKind.RECOVERY,
            status=self.status,
            detail=self.detail,
        )


def post_stressor_task_score_change(stressor_score: float, recovery_score: float) -> float:
    """Return recovery minus stressor task score; positive means improvement."""

    _require_finite(stressor_score, "stressor_score")
    _require_finite(recovery_score, "recovery_score")
    return recovery_score - stressor_score


def post_stressor_repetition_change(
    stressor_repetition: float,
    recovery_repetition: float,
) -> float:
    """Return stressor minus recovery repetition; positive means improvement."""

    _require_finite(stressor_repetition, "stressor_repetition")
    _require_finite(recovery_repetition, "recovery_repetition")
    return stressor_repetition - recovery_repetition


def time_to_return_to_target_band(
    task_scores: Sequence[float],
    repetition_ratios: Sequence[float],
    *,
    minimum_task_score: float,
    maximum_repetition_ratio: float,
) -> int | None:
    """Return the one-based first recovery turn in-band, or explicit non-recovery."""

    if len(task_scores) != len(repetition_ratios):
        raise ValueError("task and repetition trajectories must have equal lengths")
    _require_finite(minimum_task_score, "minimum_task_score")
    _require_finite(maximum_repetition_ratio, "maximum_repetition_ratio")
    if not 0.0 <= minimum_task_score <= 1.0:
        raise ValueError("minimum_task_score must be in [0, 1]")
    if not 0.0 <= maximum_repetition_ratio <= 1.0:
        raise ValueError("maximum_repetition_ratio must be in [0, 1]")
    for index, (task_score, repetition_ratio) in enumerate(
        zip(task_scores, repetition_ratios, strict=True),
        start=1,
    ):
        _require_finite(task_score, "task_score")
        _require_finite(repetition_ratio, "repetition_ratio")
        if not 0.0 <= task_score <= 1.0 or not 0.0 <= repetition_ratio <= 1.0:
            raise ValueError("recovery trajectory values must be in [0, 1]")
        if task_score >= minimum_task_score and repetition_ratio <= maximum_repetition_ratio:
            return index
    return None


def evaluate_recovery(
    paired_improvements: Mapping[RecoveryMetricName | str, Sequence[float]],
    *,
    spec: RecoveryAnalysisSpec,
    negative_multiplicity: ValidatedNegativeMultiplicitySpec = (
        DEFAULT_VALIDATED_NEGATIVE_MULTIPLICITY
    ),
    right_censored_focal_units: int = 0,
    right_censored_comparator_units: int = 0,
) -> RecoveryEvaluationResult:
    """Evaluate all three preregistered recovery endpoints without imputation."""

    if not isinstance(spec, RecoveryAnalysisSpec):
        raise TypeError("spec must be a RecoveryAnalysisSpec")
    if not isinstance(negative_multiplicity, ValidatedNegativeMultiplicitySpec):
        raise TypeError("negative_multiplicity must be a ValidatedNegativeMultiplicitySpec")
    if (
        not isinstance(right_censored_focal_units, int)
        or isinstance(right_censored_focal_units, bool)
        or not isinstance(right_censored_comparator_units, int)
        or isinstance(right_censored_comparator_units, bool)
        or right_censored_focal_units < 0
        or right_censored_comparator_units < 0
    ):
        raise ValueError("right-censored recovery counts must be nonnegative integers")
    try:
        values = {
            RecoveryMetricName(key) if isinstance(key, str) else key: tuple(differences)
            for key, differences in paired_improvements.items()
        }
    except (TypeError, ValueError):
        return RecoveryEvaluationResult(
            status=ScientificEvidenceStatus.INVALID,
            detail="recovery evidence contains an unknown endpoint",
        )
    if set(values) != set(RECOVERY_METRIC_NAMES):
        return RecoveryEvaluationResult(
            status=ScientificEvidenceStatus.INVALID,
            detail="recovery evidence does not contain the exact required endpoint set",
        )
    counts = {len(differences) for differences in values.values()}
    if len(counts) != 1 or counts == {0}:
        return RecoveryEvaluationResult(
            status=ScientificEvidenceStatus.INVALID,
            detail="recovery evidence lacks equal nonempty matched-unit coverage",
        )
    try:
        results = tuple(
            _recovery_metric_result(
                metric_name,
                values[metric_name],
                spec,
                negative_multiplicity,
            )
            for metric_name in RECOVERY_METRIC_NAMES
        )
    except (TypeError, ValueError):
        return RecoveryEvaluationResult(
            status=ScientificEvidenceStatus.INVALID,
            detail="recovery evidence contains nonfinite or otherwise invalid differences",
        )
    statuses = {result.status for result in results}
    if right_censored_focal_units > 0:
        status = ScientificEvidenceStatus.DECISIVE_NEGATIVE
        detail = "one or more focal units did not return to the preregistered target band"
    elif ScientificEvidenceStatus.DECISIVE_NEGATIVE in statuses:
        status = ScientificEvidenceStatus.DECISIVE_NEGATIVE
        detail = "one or more required recovery endpoints decisively fail"
    elif ScientificEvidenceStatus.INCONCLUSIVE in statuses:
        status = ScientificEvidenceStatus.INCONCLUSIVE
        detail = "one or more required recovery endpoints remain unresolved"
    else:
        status = ScientificEvidenceStatus.PASS
        detail = "all required recovery endpoints meet their frozen improvement gates"
    return RecoveryEvaluationResult(
        metric_results=results,
        right_censored_focal_units=right_censored_focal_units,
        right_censored_comparator_units=right_censored_comparator_units,
        status=status,
        detail=detail,
    )


def _recovery_metric_result(
    metric_name: RecoveryMetricName,
    differences: tuple[float, ...],
    spec: RecoveryAnalysisSpec,
    negative_multiplicity: ValidatedNegativeMultiplicitySpec,
) -> RecoveryMetricResult:
    bootstrap = paired_bootstrap_ci(
        differences,
        resamples=spec.bootstrap_resamples,
        confidence_level=spec.confidence_level,
        seed=spec.bootstrap_seed,
    )
    threshold = spec.practical_thresholds[metric_name]
    negative_evidence = negative_side_evidence(
        differences,
        gate_id=f"recovery:{metric_name.value}",
        practical_effect_threshold=threshold,
        resamples=spec.bootstrap_resamples,
        seed=spec.bootstrap_seed,
        multiplicity=negative_multiplicity,
    )
    return RecoveryMetricResult(
        metric_name=metric_name,
        unit_count=len(differences),
        estimate=bootstrap.estimate,
        bootstrap=bootstrap,
        negative_side_evidence=negative_evidence,
        practical_effect_threshold=threshold,
        status=_classify_recovery_metric(bootstrap, negative_evidence, threshold),
    )


def _classify_recovery_metric(
    bootstrap: BootstrapResult,
    negative_evidence: NegativeSideEvidence,
    practical_effect_threshold: float,
) -> ScientificEvidenceStatus:
    if bootstrap.estimate >= practical_effect_threshold and bootstrap.lower > 0.0:
        return ScientificEvidenceStatus.PASS
    if negative_evidence.decisive_negative:
        return ScientificEvidenceStatus.DECISIVE_NEGATIVE
    return ScientificEvidenceStatus.INCONCLUSIVE


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "RECOVERY_METRIC_NAMES",
    "RecoveryAnalysisSpec",
    "RecoveryEvaluationResult",
    "RecoveryMetricName",
    "RecoveryMetricResult",
    "evaluate_recovery",
    "post_stressor_repetition_change",
    "post_stressor_task_score_change",
    "time_to_return_to_target_band",
]

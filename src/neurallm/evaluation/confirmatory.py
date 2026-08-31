"""Frozen confirmatory analysis contract and complete result envelope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from neurallm.domain.models import (
    FiniteFloat,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    Sha256Hex,
    UnitInterval,
)
from neurallm.domain.serialization import canonical_sha256
from neurallm.evaluation.attribution import (
    AttributionAnalysisSpec,
    PersistentStateAttributionResult,
    evaluate_persistent_state_attribution,
)
from neurallm.evaluation.models import (
    BootstrapResult,
    CoverageResult,
    MatchedUnitKey,
    PermutationTestResult,
)
from neurallm.evaluation.recovery import (
    RECOVERY_METRIC_NAMES,
    RecoveryAnalysisSpec,
    RecoveryEvaluationResult,
    RecoveryMetricName,
    evaluate_recovery,
)
from neurallm.evaluation.scientific import (
    VALIDATED_NEGATIVE_MULTIPLICITY_SHA256,
    EfficacyAnalysisSpec,
    EfficacyComparisonResult,
    ExperimentTier,
    GuardrailCleanTaskScore,
    LimitationDisposition,
    LimitationKind,
    ScientificDecisionInput,
    ScientificDecisionRecord,
    ScientificEvidenceStatus,
    ScientificFrozenModel,
    ScientificGuardrailResult,
    ScientificGuardrailStatus,
    ScientificLimitation,
    ValidatedNegativeMultiplicitySpec,
    decide_scientific_outcome,
    evaluate_efficacy_comparisons,
)
from neurallm.evaluation.statistics import paired_bootstrap_ci


class RecoveryEventSpec(ScientificFrozenModel):
    """One preregistered stressor and its ordered eligible recovery turns."""

    prompt_sequence_id: NonEmptyString
    stressor_turn_index: NonNegativeInt
    recovery_turn_indexes: tuple[NonNegativeInt, ...]
    minimum_task_score_target: UnitInterval
    maximum_repetition_ratio_target: UnitInterval

    @field_validator("recovery_turn_indexes", mode="before")
    @classmethod
    def _accept_yaml_turns(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_recovery_window(self) -> Self:
        if not self.recovery_turn_indexes:
            raise ValueError("a recovery event requires at least one eligible recovery turn")
        if self.recovery_turn_indexes != tuple(sorted(set(self.recovery_turn_indexes))):
            raise ValueError("recovery turn indexes must be sorted and unique")
        if any(index <= self.stressor_turn_index for index in self.recovery_turn_indexes):
            raise ValueError("recovery turns must occur strictly after the stressor")
        return self


class ConfirmatoryAnalysisSpec(ScientificFrozenModel):
    """Complete post-pilot contract frozen before confirmatory execution."""

    schema_version: Literal[2] = 2
    implementation_version: Literal["confirmatory-analysis-v2"] = "confirmatory-analysis-v2"
    primary_endpoint: Literal["guardrail_clean_task_score"] = "guardrail_clean_task_score"
    efficacy: EfficacyAnalysisSpec
    recovery: RecoveryAnalysisSpec
    attribution: AttributionAnalysisSpec
    validated_negative_multiplicity: ValidatedNegativeMultiplicitySpec = (
        ValidatedNegativeMultiplicitySpec()
    )
    recovery_events: tuple[RecoveryEventSpec, ...]
    optional_metric_dispositions: Mapping[NonEmptyString, LimitationDisposition]
    subgroup_fields: tuple[NonEmptyString, ...] = ("prompt_family",)
    subgroup_conflict_rule: Literal["opposite-resolved-serious-effect-v1"] = (
        "opposite-resolved-serious-effect-v1"
    )

    @field_validator("recovery_events", "subgroup_fields", mode="before")
    @classmethod
    def _accept_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("recovery_events")
    @classmethod
    def _sort_unique_events(
        cls,
        values: tuple[RecoveryEventSpec, ...],
    ) -> tuple[RecoveryEventSpec, ...]:
        if not values:
            raise ValueError("confirmatory analysis requires preregistered recovery events")
        identifiers = tuple(event.prompt_sequence_id for event in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recovery events must target unique prompt sequences")
        return tuple(sorted(values, key=lambda event: event.prompt_sequence_id))

    @field_validator("optional_metric_dispositions", mode="before")
    @classmethod
    def _accept_serialized_optional_dispositions(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                name: (
                    LimitationDisposition(disposition)
                    if isinstance(disposition, str)
                    else disposition
                )
                for name, disposition in value.items()
            }
        return value

    @field_validator("optional_metric_dispositions")
    @classmethod
    def _freeze_optional_dispositions(
        cls,
        values: Mapping[str, LimitationDisposition],
    ) -> Mapping[str, LimitationDisposition]:
        if not values:
            raise ValueError("optional metric missingness dispositions must be explicit")
        return MappingProxyType(dict(sorted(values.items())))

    @field_serializer("optional_metric_dispositions")
    def _serialize_optional_dispositions(
        self,
        values: Mapping[str, LimitationDisposition],
    ) -> dict[str, str]:
        return {name: disposition.value for name, disposition in values.items()}

    @field_validator("subgroup_fields")
    @classmethod
    def _sort_unique_subgroups(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("subgroup fields must be nonempty, sorted, and unique")
        return values


class ScientificUnitOutcome(ScientificFrozenModel):
    """Auditable raw primary and recovery metrics for one efficacy unit."""

    unit_key: MatchedUnitKey
    prompt_family: NonEmptyString
    policy_id: Literal[
        "best_static",
        "heuristic_adaptive",
        "neural_persistent",
        "random_matched",
    ]
    guardrail_clean_task_score: GuardrailCleanTaskScore
    instruction_adherence: UnitInterval
    repetition_ratio: UnitInterval
    response_length_tokens: float = Field(ge=0.0, allow_inf_nan=False)


class SubgroupEffectResult(ScientificFrozenModel):
    """Persisted prompt-family sensitivity evidence for one serious comparator."""

    field_name: Literal["prompt_family"] = "prompt_family"
    field_value: NonEmptyString
    comparator_policy_id: Literal["best_static", "heuristic_adaptive"]
    unit_count: PositiveInt
    bootstrap: BootstrapResult
    practical_effect_threshold: float = Field(gt=0.0, allow_inf_nan=False)
    direction: Literal["beneficial", "harmful", "unresolved"]

    @model_validator(mode="after")
    def _validate_direction(self) -> Self:
        if (
            self.bootstrap.estimate >= self.practical_effect_threshold
            and self.bootstrap.lower > 0.0
        ):
            expected = "beneficial"
        elif self.bootstrap.upper < 0.0:
            expected = "harmful"
        else:
            expected = "unresolved"
        if self.direction != expected:
            raise ValueError("subgroup direction does not match its bootstrap evidence")
        if self.bootstrap.sample_size != self.unit_count:
            raise ValueError("subgroup bootstrap does not match its unit count")
        return self


class RecoveryUnitOutcome(ScientificFrozenModel):
    """One preregistered recovery-event by model-seed raw margin record."""

    unit_key: MatchedUnitKey
    post_stressor_task_score_change: FiniteFloat
    post_stressor_repetition_change: FiniteFloat
    time_to_return_to_target_band: FiniteFloat
    focal_right_censored: bool
    comparator_right_censored_count: int = Field(ge=0, le=2)


class AttributionUnitOutcome(ScientificFrozenModel):
    """One intervention-only persistent-minus-reset matched-unit difference."""

    unit_key: MatchedUnitKey
    persistent_minus_reset_task_score: FiniteFloat


class ConfirmatoryEvaluationResult(ScientificFrozenModel):
    """One hash-bound confirmatory result and all decision evidence families."""

    schema_version: Literal[2] = 2
    implementation_version: Literal["confirmatory-evaluation-v2"] = "confirmatory-evaluation-v2"
    claim_scope: Literal["confirmatory-model-backed-scientific-decision"] = (
        "confirmatory-model-backed-scientific-decision"
    )
    analysis_contract_sha256: Sha256Hex
    confirmatory_analysis_spec: ConfirmatoryAnalysisSpec
    confirmatory_analysis_spec_sha256: Sha256Hex
    prompt_family_by_sequence: Mapping[NonEmptyString, NonEmptyString]
    prompt_family_design_sha256: Sha256Hex
    validated_negative_multiplicity_sha256: Sha256Hex = VALIDATED_NEGATIVE_MULTIPLICITY_SHA256
    causal_mechanism_validated: Literal[True] = True
    claim_eligible: bool
    run_manifest_sha256: Sha256Hex | None
    run_finalization_sha256: Sha256Hex | None
    input_sha256: Sha256Hex
    result_sha256: Sha256Hex
    coverage: CoverageResult
    optional_metric_availability: Mapping[
        NonEmptyString,
        tuple[NonNegativeInt, PositiveInt],
    ]
    unit_outcomes: tuple[ScientificUnitOutcome, ...]
    recovery_unit_outcomes: tuple[RecoveryUnitOutcome, ...]
    attribution_unit_outcomes: tuple[AttributionUnitOutcome, ...]
    efficacy_comparisons: tuple[EfficacyComparisonResult, ...]
    recovery: RecoveryEvaluationResult
    attribution: PersistentStateAttributionResult
    subgroup_effects: tuple[SubgroupEffectResult, ...]
    guardrails: tuple[ScientificGuardrailResult, ...]
    limitations: tuple[ScientificLimitation, ...]
    decision: ScientificDecisionRecord
    statistics_call_count: NonNegativeInt

    @field_validator(
        "unit_outcomes",
        "recovery_unit_outcomes",
        "attribution_unit_outcomes",
        "efficacy_comparisons",
        "subgroup_effects",
        "guardrails",
        "limitations",
        mode="before",
    )
    @classmethod
    def _accept_serialized_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("subgroup_effects")
    @classmethod
    def _canonical_subgroup_effects(
        cls,
        values: tuple[SubgroupEffectResult, ...],
    ) -> tuple[SubgroupEffectResult, ...]:
        return tuple(
            sorted(
                values,
                key=lambda value: (value.comparator_policy_id, value.field_value),
            )
        )

    @field_validator("recovery_unit_outcomes")
    @classmethod
    def _canonical_recovery_unit_evidence(
        cls,
        values: tuple[RecoveryUnitOutcome, ...],
    ) -> tuple[RecoveryUnitOutcome, ...]:
        ordered = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.unit_key.prompt_sequence_id,
                    value.unit_key.model_seed,
                ),
            )
        )
        keys = tuple(
            (value.unit_key.prompt_sequence_id, value.unit_key.model_seed) for value in ordered
        )
        if len(keys) != len(set(keys)):
            raise ValueError("raw confirmatory unit evidence contains duplicate keys")
        return ordered

    @field_validator("attribution_unit_outcomes")
    @classmethod
    def _canonical_attribution_unit_evidence(
        cls,
        values: tuple[AttributionUnitOutcome, ...],
    ) -> tuple[AttributionUnitOutcome, ...]:
        ordered = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.unit_key.prompt_sequence_id,
                    value.unit_key.model_seed,
                ),
            )
        )
        keys = tuple(
            (value.unit_key.prompt_sequence_id, value.unit_key.model_seed) for value in ordered
        )
        if len(keys) != len(set(keys)):
            raise ValueError("raw confirmatory unit evidence contains duplicate keys")
        return ordered

    @field_validator("prompt_family_by_sequence")
    @classmethod
    def _freeze_prompt_family_design(
        cls,
        values: Mapping[str, str],
    ) -> Mapping[str, str]:
        if not values:
            raise ValueError("confirmatory result requires a prompt-family design")
        return MappingProxyType(dict(sorted(values.items())))

    @field_serializer("prompt_family_by_sequence")
    def _serialize_prompt_family_design(self, values: Mapping[str, str]) -> dict[str, str]:
        return dict(values)

    @field_validator("optional_metric_availability", mode="before")
    @classmethod
    def _accept_serialized_optional_availability(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                metric_name: tuple(counts) if isinstance(counts, list) else counts
                for metric_name, counts in value.items()
            }
        return value

    @field_validator("optional_metric_availability")
    @classmethod
    def _freeze_optional_availability(
        cls,
        values: Mapping[str, tuple[int, int]],
    ) -> Mapping[str, tuple[int, int]]:
        if not values:
            raise ValueError("confirmatory result requires optional metric availability evidence")
        for available, total in values.values():
            if available > total:
                raise ValueError("optional metric availability counts are invalid")
        return MappingProxyType(dict(sorted(values.items())))

    @field_serializer("optional_metric_availability")
    def _serialize_optional_availability(
        self,
        values: Mapping[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]]:
        return dict(values)

    @model_validator(mode="after")
    def _validate_complete_result(self) -> Self:
        spec = self.confirmatory_analysis_spec
        if self.confirmatory_analysis_spec_sha256 != canonical_sha256(spec):
            raise ValueError("confirmatory analysis spec hash does not match its evidence")
        if self.prompt_family_design_sha256 != canonical_sha256(self.prompt_family_by_sequence):
            raise ValueError("prompt-family design hash does not match its evidence")
        if self.validated_negative_multiplicity_sha256 != VALIDATED_NEGATIVE_MULTIPLICITY_SHA256:
            raise ValueError("confirmatory result does not bind the frozen negative family")
        if self.validated_negative_multiplicity_sha256 != canonical_sha256(
            spec.validated_negative_multiplicity
        ):
            raise ValueError("confirmatory result multiplicity does not match its analysis spec")
        has_closed_run_bindings = (
            self.run_manifest_sha256 is not None and self.run_finalization_sha256 is not None
        )
        if self.claim_eligible != has_closed_run_bindings:
            raise ValueError(
                "claim eligibility requires both closed-run manifest and finalization bindings"
            )
        if not self.coverage.exact:
            raise ValueError("confirmatory result requires exact condition coverage")
        if set(self.optional_metric_availability) != set(spec.optional_metric_dispositions):
            raise ValueError("optional metric availability must match the frozen disposition set")
        if any(
            total != self.coverage.observed_count
            for _, total in self.optional_metric_availability.values()
        ):
            raise ValueError("optional metric availability totals must match exact coverage")
        comparator_ids = tuple(
            comparison.comparator_policy_id for comparison in self.efficacy_comparisons
        )
        if comparator_ids != ("best_static", "heuristic_adaptive", "random_matched"):
            raise ValueError("confirmatory result requires exactly three efficacy comparisons")
        if not self.unit_outcomes:
            raise ValueError("confirmatory result requires auditable unit outcomes")
        _validate_statistical_contract(self, spec)
        evidence_hashes = {
            comparison.negative_side_evidence.multiplicity_spec_sha256
            for comparison in self.efficacy_comparisons
            if comparison.negative_side_evidence is not None
        }
        evidence_hashes.update(
            result.negative_side_evidence.multiplicity_spec_sha256
            for result in self.recovery.metric_results
        )
        if self.attribution.negative_side_evidence is not None:
            evidence_hashes.add(self.attribution.negative_side_evidence.multiplicity_spec_sha256)
        if evidence_hashes and evidence_hashes != {self.validated_negative_multiplicity_sha256}:
            raise ValueError("confirmatory negative-side evidence does not share one frozen family")
        decision_input = ScientificDecisionInput(
            tier=ExperimentTier.CONFIRMATORY,
            efficacy_comparisons=self.efficacy_comparisons,
            recovery=self.recovery.decision_gate,
            attribution=self.attribution.decision_gate,
            guardrails=self.guardrails,
            limitations=self.limitations,
        )
        if self.decision.decision_input_sha256 != canonical_sha256(decision_input):
            raise ValueError("scientific decision does not hash the enclosed evidence")
        if self.decision != decide_scientific_outcome(decision_input):
            raise ValueError("scientific decision does not match the enclosed evidence")
        expected_statistics = confirmatory_statistics_call_count(
            self.efficacy_comparisons,
            self.recovery,
            self.attribution,
            self.subgroup_effects,
        )
        if self.statistics_call_count != expected_statistics:
            raise ValueError("statistics call count does not match the persisted evidence")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"result_sha256"}))
        if self.result_sha256 != expected_hash:
            raise ValueError("confirmatory result hash does not match its canonical evidence")
        return self


def _bootstrap_matches(
    bootstrap: BootstrapResult,
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> bool:
    return (
        bootstrap.resamples == resamples
        and bootstrap.seed == seed
        and bootstrap.confidence_level == confidence_level
    )


def _permutation_metadata_matches(permutation: PermutationTestResult) -> bool:
    exact = permutation.sample_size <= 20 and 2**permutation.sample_size <= (
        permutation.requested_resamples
    )
    performed = 2**permutation.sample_size if exact else permutation.requested_resamples
    return permutation.exact is exact and permutation.performed_permutations == performed


def _validate_statistical_contract(
    result: ConfirmatoryEvaluationResult,
    spec: ConfirmatoryAnalysisSpec,
) -> None:
    expected_guardrail_keys = {
        (name, "efficacy:global")
        for name in (
            "action_bound_compliance",
            "matched_condition_coverage",
            "metric_availability",
            "provider_identity_stability",
            "turn_zero_equivalence",
        )
    }
    expected_guardrail_keys.add(("action_saturation_rate", "efficacy:policy:neural_persistent"))
    for comparator_id in ("best_static", "heuristic_adaptive", "random_matched"):
        expected_guardrail_keys.update(
            {
                (name, f"efficacy:pair:neural_persistent:{comparator_id}")
                for name in (
                    "behavioral_alias_detection",
                    "instruction_adherence_non_regression",
                    "response_length_confound",
                )
            }
        )
    expected_guardrail_keys.update(
        {
            ("causal_mechanism_validation", "attribution:causal"),
            ("intervention_turn_only_attribution", "attribution:causal"),
        }
    )
    expected_guardrail_keys.update(
        {
            (
                name,
                "attribution:pair:neural_persistent:neural_matched_history_state_reset",
            )
            for name in (
                "behavioral_alias_detection",
                "instruction_adherence_non_regression",
                "response_length_confound",
            )
        }
    )
    actual_guardrail_keys = {(guardrail.name, guardrail.scope) for guardrail in result.guardrails}
    if len(result.guardrails) != 20 or actual_guardrail_keys != expected_guardrail_keys:
        raise ValueError("confirmatory guardrails do not cover the exact frozen scope set")

    for comparison in result.efficacy_comparisons:
        if comparison.bootstrap is None:
            continue
        assert comparison.negative_side_evidence is not None
        assert comparison.permutation is not None
        if not _bootstrap_matches(
            comparison.bootstrap,
            resamples=spec.efficacy.bootstrap_resamples,
            seed=spec.efficacy.bootstrap_seed,
            confidence_level=spec.efficacy.confidence_level,
        ) or not _bootstrap_matches(
            comparison.negative_side_evidence.bootstrap,
            resamples=spec.efficacy.bootstrap_resamples,
            seed=spec.efficacy.bootstrap_seed,
            confidence_level=(
                spec.validated_negative_multiplicity.adjusted_two_sided_confidence_level
            ),
        ):
            raise ValueError("efficacy bootstrap evidence does not match the analysis spec")
        if (
            comparison.permutation.requested_resamples != spec.efficacy.permutation_resamples
            or comparison.permutation.seed != spec.efficacy.permutation_seed
            or not _permutation_metadata_matches(comparison.permutation)
            or comparison.practical_effect_threshold != spec.efficacy.practical_effect_threshold
        ):
            raise ValueError("efficacy evidence parameters do not match the analysis spec")

    for metric in result.recovery.metric_results:
        if not _bootstrap_matches(
            metric.bootstrap,
            resamples=spec.recovery.bootstrap_resamples,
            seed=spec.recovery.bootstrap_seed,
            confidence_level=spec.recovery.confidence_level,
        ) or not _bootstrap_matches(
            metric.negative_side_evidence.bootstrap,
            resamples=spec.recovery.bootstrap_resamples,
            seed=spec.recovery.bootstrap_seed,
            confidence_level=(
                spec.validated_negative_multiplicity.adjusted_two_sided_confidence_level
            ),
        ):
            raise ValueError("recovery bootstrap evidence does not match the analysis spec")
        if (
            metric.practical_effect_threshold
            != spec.recovery.practical_thresholds[metric.metric_name]
        ):
            raise ValueError("recovery threshold does not match the analysis spec")

    attribution = result.attribution
    if attribution.bootstrap is not None:
        assert attribution.negative_side_evidence is not None
        assert attribution.permutation is not None
        if not _bootstrap_matches(
            attribution.bootstrap,
            resamples=spec.attribution.bootstrap_resamples,
            seed=spec.attribution.bootstrap_seed,
            confidence_level=spec.attribution.confidence_level,
        ) or not _bootstrap_matches(
            attribution.negative_side_evidence.bootstrap,
            resamples=spec.attribution.bootstrap_resamples,
            seed=spec.attribution.bootstrap_seed,
            confidence_level=(
                spec.validated_negative_multiplicity.adjusted_two_sided_confidence_level
            ),
        ):
            raise ValueError("attribution bootstrap evidence does not match the analysis spec")
        if (
            attribution.permutation.requested_resamples != spec.attribution.permutation_resamples
            or attribution.permutation.seed != spec.attribution.permutation_seed
            or not _permutation_metadata_matches(attribution.permutation)
            or attribution.practical_effect_threshold != spec.attribution.practical_effect_threshold
        ):
            raise ValueError("attribution evidence parameters do not match the analysis spec")

    prompt_family_design = dict(result.prompt_family_by_sequence)
    outcomes_by_policy: dict[str, dict[tuple[str, int], ScientificUnitOutcome]] = {}
    for outcome in result.unit_outcomes:
        sequence_id = outcome.unit_key.prompt_sequence_id
        if prompt_family_design.get(sequence_id) != outcome.prompt_family:
            raise ValueError("unit outcomes disagree with the frozen prompt-family design")
        key = (sequence_id, outcome.unit_key.model_seed)
        policy_outcomes = outcomes_by_policy.setdefault(outcome.policy_id, {})
        if key in policy_outcomes:
            raise ValueError("unit outcomes contain a duplicate policy/unit key")
        policy_outcomes[key] = outcome
    if {outcome.unit_key.prompt_sequence_id for outcome in result.unit_outcomes} != set(
        prompt_family_design
    ):
        raise ValueError("unit outcomes do not cover the exact prompt-family design")
    expected_policies = {
        "best_static",
        "heuristic_adaptive",
        "neural_persistent",
        "random_matched",
    }
    if set(outcomes_by_policy) != expected_policies:
        raise ValueError("unit outcomes do not cover the exact efficacy policy set")
    focal = outcomes_by_policy["neural_persistent"]
    if any(values.keys() != focal.keys() for values in outcomes_by_policy.values()):
        raise ValueError("unit outcomes do not share exact matched-unit keys")
    recovery_sequences = {event.prompt_sequence_id for event in spec.recovery_events}
    if not recovery_sequences.issubset(result.prompt_family_by_sequence):
        raise ValueError("recovery events reference a sequence outside the prompt-family design")
    recovery_keys = {
        (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed)
        for outcome in result.recovery_unit_outcomes
    }
    expected_recovery_keys = {
        (sequence_id, model_seed)
        for sequence_id in recovery_sequences
        for model_seed in {key[1] for key in focal}
    }
    if recovery_keys != expected_recovery_keys:
        raise ValueError("recovery unit outcomes do not cover the exact event/seed keys")
    attribution_keys = {
        (outcome.unit_key.prompt_sequence_id, outcome.unit_key.model_seed)
        for outcome in result.attribution_unit_outcomes
    }
    if attribution_keys != set(focal):
        raise ValueError("attribution unit outcomes do not cover the exact focal matched-unit keys")

    for outcome in result.unit_outcomes:
        policy_id = outcome.policy_id
        applicable_guardrails = tuple(
            guardrail
            for guardrail in result.guardrails
            if guardrail.scope == "efficacy:global"
            or guardrail.scope == f"efficacy:policy:{policy_id}"
            or (
                guardrail.scope.startswith("efficacy:pair:")
                and policy_id in guardrail.scope.split(":")[2:]
            )
        )
        expected_gate_names = tuple(sorted({guardrail.name for guardrail in applicable_guardrails}))
        if any(
            guardrail.status is ScientificGuardrailStatus.INVALID
            for guardrail in applicable_guardrails
        ):
            expected_gate_status = ScientificGuardrailStatus.INVALID
        elif any(
            guardrail.status is ScientificGuardrailStatus.FAIL
            for guardrail in applicable_guardrails
        ):
            expected_gate_status = ScientificGuardrailStatus.FAIL
        else:
            expected_gate_status = ScientificGuardrailStatus.PASS
        score = outcome.guardrail_clean_task_score
        if (
            not expected_gate_names
            or score.gate_names != expected_gate_names
            or score.gate_status is not expected_gate_status
        ):
            raise ValueError("unit outcome gates do not match the enclosed guardrail evidence")

    guardrails_by_comparator: dict[str, tuple[ScientificGuardrailResult, ...]] = {}
    alias_by_comparator: dict[str, bool] = {}
    differences_by_comparator: dict[str, tuple[float, ...]] = {}
    for comparator_id in (
        "best_static",
        "heuristic_adaptive",
        "random_matched",
    ):
        comparator = outcomes_by_policy[comparator_id]
        differences_by_comparator[comparator_id] = tuple(
            focal[key].guardrail_clean_task_score.raw_task_score
            - comparator[key].guardrail_clean_task_score.raw_task_score
            for key in sorted(focal)
        )
        guardrails_by_comparator[comparator_id] = tuple(
            guardrail
            for guardrail in result.guardrails
            if guardrail.scope == "efficacy:global"
            or guardrail.scope == "efficacy:policy:neural_persistent"
            or guardrail.scope == f"efficacy:pair:neural_persistent:{comparator_id}"
        )
        alias_by_comparator[comparator_id] = any(
            guardrail.name == "behavioral_alias_detection"
            and guardrail.status is ScientificGuardrailStatus.FAIL
            for guardrail in guardrails_by_comparator[comparator_id]
        )
    expected_efficacy = evaluate_efficacy_comparisons(
        differences_by_comparator,
        spec=spec.efficacy,
        negative_multiplicity=spec.validated_negative_multiplicity,
        guardrails_by_comparator=guardrails_by_comparator,
        behavioral_alias_by_comparator=alias_by_comparator,
    )
    if result.efficacy_comparisons != expected_efficacy:
        raise ValueError("efficacy evidence does not reconstruct from the raw unit outcomes")

    recovery_values: Mapping[RecoveryMetricName | str, Sequence[float]] = {
        RecoveryMetricName.POST_STRESSOR_TASK_SCORE_CHANGE: tuple(
            outcome.post_stressor_task_score_change for outcome in result.recovery_unit_outcomes
        ),
        RecoveryMetricName.POST_STRESSOR_REPETITION_CHANGE: tuple(
            outcome.post_stressor_repetition_change for outcome in result.recovery_unit_outcomes
        ),
        RecoveryMetricName.TIME_TO_RETURN_TO_TARGET_BAND: tuple(
            outcome.time_to_return_to_target_band for outcome in result.recovery_unit_outcomes
        ),
    }
    assert tuple(recovery_values) == RECOVERY_METRIC_NAMES
    integrity_invalid = any(
        guardrail.status is ScientificGuardrailStatus.INVALID
        for guardrail in result.guardrails
        if guardrail.scope == "efficacy:global"
    )
    expected_focal_censored = sum(
        int(outcome.focal_right_censored) for outcome in result.recovery_unit_outcomes
    )
    expected_comparator_censored = sum(
        outcome.comparator_right_censored_count for outcome in result.recovery_unit_outcomes
    )
    if (
        result.recovery.right_censored_focal_units != expected_focal_censored
        or result.recovery.right_censored_comparator_units != expected_comparator_censored
    ):
        raise ValueError("recovery censoring counts do not match raw unit outcomes")
    if integrity_invalid:
        if (
            result.recovery.status is not ScientificEvidenceStatus.INVALID
            or result.recovery.metric_results
        ):
            raise ValueError("invalid global integrity must suppress recovery statistics")
    else:
        expected_recovery = evaluate_recovery(
            recovery_values,
            spec=spec.recovery,
            negative_multiplicity=spec.validated_negative_multiplicity,
            right_censored_focal_units=expected_focal_censored,
            right_censored_comparator_units=expected_comparator_censored,
        )
        if result.recovery != expected_recovery:
            raise ValueError("recovery evidence does not reconstruct from its raw unit outcomes")

    attribution_guardrails = tuple(
        guardrail for guardrail in result.guardrails if guardrail.scope.startswith("attribution:")
    )
    attribution_alias = any(
        guardrail.name == "behavioral_alias_detection"
        and guardrail.status is ScientificGuardrailStatus.FAIL
        for guardrail in attribution_guardrails
    )
    if integrity_invalid:
        if (
            result.attribution.status is not ScientificEvidenceStatus.INVALID
            or result.attribution.bootstrap is not None
            or result.attribution.negative_side_evidence is not None
            or result.attribution.permutation is not None
        ):
            raise ValueError("invalid global integrity must suppress attribution statistics")
    else:
        expected_attribution = evaluate_persistent_state_attribution(
            tuple(
                outcome.persistent_minus_reset_task_score
                for outcome in result.attribution_unit_outcomes
            ),
            spec=spec.attribution,
            negative_multiplicity=spec.validated_negative_multiplicity,
            causal_guardrails=attribution_guardrails,
            behavioral_alias=attribution_alias,
        )
        if result.attribution != expected_attribution:
            raise ValueError("attribution evidence does not reconstruct from its raw unit outcomes")

    expected_effects: dict[tuple[str, str], tuple[float, ...]] = {}
    for comparator_id in () if integrity_invalid else spec.efficacy.serious_comparator_ids:
        grouped: dict[str, list[float]] = {}
        comparator = outcomes_by_policy[comparator_id]
        for key in sorted(focal):
            family = prompt_family_design[key[0]]
            grouped.setdefault(family, []).append(
                focal[key].guardrail_clean_task_score.raw_task_score
                - comparator[key].guardrail_clean_task_score.raw_task_score
            )
        if len(grouped) >= 2:
            expected_effects.update(
                {
                    (comparator_id, family): tuple(differences)
                    for family, differences in grouped.items()
                }
            )

    effects_by_key: dict[tuple[str, str], SubgroupEffectResult] = {
        (effect.comparator_policy_id, effect.field_value): effect
        for effect in result.subgroup_effects
    }
    if len(effects_by_key) != len(result.subgroup_effects):
        raise ValueError("subgroup effects must be unique by comparator and field value")
    if set(effects_by_key) != set(expected_effects):
        raise ValueError("subgroup effects do not cover the exact prompt-family design")
    for effect_key, differences in expected_effects.items():
        effect = effects_by_key[effect_key]
        expected_bootstrap = paired_bootstrap_ci(
            differences,
            resamples=spec.efficacy.bootstrap_resamples,
            confidence_level=spec.efficacy.confidence_level,
            seed=spec.efficacy.bootstrap_seed,
        )
        if (
            effect.unit_count != len(differences)
            or effect.bootstrap != expected_bootstrap
            or effect.practical_effect_threshold != spec.efficacy.practical_effect_threshold
        ):
            raise ValueError("subgroup evidence does not match the analysis spec and unit design")
    expected_limitations: list[ScientificLimitation] = []
    for metric_name, disposition in spec.optional_metric_dispositions.items():
        available, total = result.optional_metric_availability[metric_name]
        if available != total:
            expected_limitations.append(
                ScientificLimitation(
                    kind=LimitationKind.OPTIONAL_METRIC_UNAVAILABLE,
                    code=f"optional_metric_unavailable_{metric_name}",
                    detail=f"{metric_name} available for {available} of {total} committed turns",
                    disposition=disposition,
                )
            )
    focal_censored = sum(
        int(outcome.focal_right_censored) for outcome in result.recovery_unit_outcomes
    )
    comparator_censored = sum(
        outcome.comparator_right_censored_count for outcome in result.recovery_unit_outcomes
    )
    if focal_censored or comparator_censored:
        expected_limitations.append(
            ScientificLimitation(
                kind=LimitationKind.OTHER,
                code="recovery_right_censoring",
                detail=(
                    f"right-censored window+1 units: focal={focal_censored}, "
                    f"serious_comparator={comparator_censored}"
                ),
                disposition=LimitationDisposition.DISCLOSURE_ONLY,
            )
        )
    for comparator_id in spec.efficacy.serious_comparator_ids:
        directions = {
            effect.direction
            for effect in result.subgroup_effects
            if effect.comparator_policy_id == comparator_id and effect.direction != "unresolved"
        }
        if directions == {"beneficial", "harmful"}:
            expected_limitations.append(
                ScientificLimitation(
                    kind=LimitationKind.SUBGROUP_CONFLICT,
                    code=f"prompt_family_conflict_{comparator_id}",
                    detail=(
                        "prompt_family subgroups contain resolved beneficial and harmful "
                        f"effects against {comparator_id}"
                    ),
                    disposition=LimitationDisposition.INCONCLUSIVE,
                )
            )
    if result.limitations != tuple(expected_limitations):
        raise ValueError("limitations do not reconstruct from the persisted raw evidence")


def confirmatory_statistics_call_count(
    efficacy: tuple[EfficacyComparisonResult, ...],
    recovery: RecoveryEvaluationResult,
    attribution: PersistentStateAttributionResult,
    subgroup_effects: tuple[SubgroupEffectResult, ...],
) -> int:
    """Derive the exact number of persisted statistical computations."""

    efficacy_calls = sum(
        int(comparison.bootstrap is not None)
        + int(comparison.negative_side_evidence is not None)
        + int(comparison.permutation is not None)
        for comparison in efficacy
    )
    recovery_calls = sum(
        1 + int(metric.negative_side_evidence is not None) for metric in recovery.metric_results
    )
    attribution_calls = (
        int(attribution.bootstrap is not None)
        + int(attribution.negative_side_evidence is not None)
        + int(attribution.permutation is not None)
    )
    return efficacy_calls + recovery_calls + attribution_calls + len(subgroup_effects)


def confirmatory_result_sha256(payload: Mapping[str, object]) -> str:
    """Hash a complete result payload before validated construction."""

    return canonical_sha256(payload)


__all__ = [
    "ConfirmatoryAnalysisSpec",
    "ConfirmatoryEvaluationResult",
    "RecoveryEventSpec",
    "ScientificUnitOutcome",
    "SubgroupEffectResult",
    "confirmatory_statistics_call_count",
    "confirmatory_result_sha256",
]

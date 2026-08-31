"""Provider-free preparation and explicitly authorized experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurallm.domain.serialization import canonical_json, canonical_sha256
from neurallm.evaluation.models import DatasetPurpose
from neurallm.experiments.analysis import analyze_closed_run
from neurallm.experiments.config import (
    DatasetReference,
    LoadedExperimentConfig,
    load_experiment_config,
)
from neurallm.experiments.dataset import (
    LoadedDataset,
    load_dataset,
    validate_dataset_identity,
)
from neurallm.experiments.plan import ExperimentPlan, build_plan
from neurallm.experiments.protocol import RunTier
from neurallm.experiments.runner import (
    ExecutionSummary,
    GitProvenance,
    PolicyRuntime,
    build_fixed_policy_runtimes,
    build_policy_runtimes,
    build_run_manifest,
    execute_plan,
    read_git_provenance,
)
from neurallm.experiments.scientific_analysis import analyze_closed_confirmatory_run
from neurallm.experiments.yaml_loader import load_yaml_mapping
from neurallm.providers.fake import (
    FakeProvider,
    fake_provider_effective_configuration_json,
    fake_provider_identity,
)
from neurallm.providers.llama_cpp import (
    LlamaCppEffectiveConfiguration,
    LlamaCppProvider,
    LlamaCppProviderConfig,
    llama_cpp_provider_identity,
)
from neurallm.reporting import (
    CLOSED_RUN_ARTIFACTS,
    SQLITE_RECOVERY_SIDECARS,
    ArtifactExportSummary,
    export_closed_run,
)
from neurallm.storage import ScientificAnalysisManifest, SQLiteRunStore, StoreInvariantError

ExecutableProvider = FakeProvider | LlamaCppProvider


class LiveProviderAuthorizationError(RuntimeError):
    """Raised before live provider construction when execution lacks authorization."""


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    """All provider-free validation, planning, policy, and provenance results."""

    loaded_config: LoadedExperimentConfig
    loaded_dataset: LoadedDataset
    plan: ExperimentPlan
    provenance: GitProvenance
    policy_runtimes: dict[str, PolicyRuntime]
    development_selection_dataset: LoadedDataset | None = None

    @property
    def artifact_identity_sha256(self) -> str:
        """Hash the manifest expected if execution identity inspection agrees."""

        manifest = build_run_manifest(
            self.plan,
            self.plan.provider_identity,
            self.policy_runtimes,
            self.provenance,
        )
        return canonical_sha256(manifest)


@dataclass(frozen=True, slots=True)
class WorkflowExecutionSummary:
    """Execution checkpoints plus the exact compact exported surface."""

    execution: ExecutionSummary
    artifacts: ArtifactExportSummary


def _load_declared_dataset(path: Path, reference: DatasetReference) -> LoadedDataset:
    """Load and validate one dataset against its complete declared identity."""

    loaded_dataset = load_dataset(path, expected_version=reference.version)
    validate_dataset_identity(
        loaded_dataset.dataset,
        expected_version=reference.version,
        expected_purpose=reference.purpose,
        expected_sha256=reference.expected_dataset_sha256,
        seal=reference.seal,
    )
    return loaded_dataset


def _load_development_selection_dataset(
    loaded_config: LoadedExperimentConfig,
) -> LoadedDataset | None:
    selection_input = loaded_config.config.development_selection_input
    if selection_input is None:
        if loaded_config.development_selection_dataset_path is not None:
            raise ValueError("unexpected development-selection dataset path")
        return None
    if selection_input.dataset.purpose is not DatasetPurpose.DEVELOPMENT:
        raise ValueError("development-selection input must have development purpose")
    selection_path = loaded_config.development_selection_dataset_path
    if selection_path is None:
        raise ValueError("development-selection input requires an explicit dataset path")
    return _load_declared_dataset(selection_path, selection_input.dataset)


def prepare_experiment(
    config_path: Path,
    *,
    provenance: GitProvenance | None = None,
) -> PreparedExperiment:
    """Validate and plan without constructing any generation provider."""

    loaded_config = load_experiment_config(config_path)
    _validate_declared_provider_configuration(loaded_config)
    loaded_dataset = _load_declared_dataset(
        loaded_config.dataset_path,
        loaded_config.config.dataset,
    )
    development_selection_dataset = _load_development_selection_dataset(loaded_config)
    plan = build_plan(loaded_config, loaded_dataset)
    policy_specs = loaded_config.config.policy_specs
    policy_runtimes = dict(
        build_fixed_policy_runtimes(plan)
        if policy_specs is None
        else build_policy_runtimes(plan, policy_specs)
    )
    resolved_provenance = provenance or read_git_provenance(loaded_config.source_path)
    return PreparedExperiment(
        loaded_config=loaded_config,
        loaded_dataset=loaded_dataset,
        plan=plan,
        provenance=resolved_provenance,
        policy_runtimes=policy_runtimes,
        development_selection_dataset=development_selection_dataset,
    )


def _validate_declared_provider_configuration(
    loaded_config: LoadedExperimentConfig,
) -> None:
    """Validate provider files and preflight evidence without constructing a provider."""

    selection = loaded_config.config.provider
    if selection.kind == "fake":
        if selection.expected_identity != fake_provider_identity():
            raise ValueError("fake provider identity is not the built-in contract")
        if (
            selection.expected_effective_configuration_json
            != fake_provider_effective_configuration_json()
        ):
            raise ValueError("fake provider effective configuration is not the built-in contract")
        return

    provider_path = loaded_config.provider_config_path
    if provider_path is None:
        raise ValueError("llama_cpp validation requires an explicit provider config path")
    if any(not 0 <= seed < 0xFFFFFFFF for seed in loaded_config.config.model_seeds):
        raise ValueError("llama_cpp model seeds must be in range 0..4294967294")
    provider_config = LlamaCppProviderConfig.model_validate(
        load_yaml_mapping(provider_path.expanduser().resolve(strict=True))
    )
    effective = LlamaCppEffectiveConfiguration.model_validate_json(
        selection.expected_effective_configuration_json
    )
    if canonical_json(effective) != selection.expected_effective_configuration_json:
        raise ValueError("expected effective configuration is not normalized preflight evidence")
    if effective.client_config != provider_config:
        raise ValueError("provider config file differs from expected effective configuration")
    if selection.expected_identity != llama_cpp_provider_identity(effective):
        raise ValueError("expected provider identity disagrees with effective configuration")


def construct_provider(loaded_config: LoadedExperimentConfig) -> ExecutableProvider:
    """Construct exactly the explicitly selected provider for execute mode."""

    selection = loaded_config.config.provider
    if selection.kind == "fake":
        provider: ExecutableProvider = FakeProvider()
    else:
        provider_path = loaded_config.provider_config_path
        if provider_path is None:
            raise ValueError("llama_cpp execution requires an explicit provider config path")
        provider_config = LlamaCppProviderConfig.model_validate(
            load_yaml_mapping(provider_path.expanduser().resolve(strict=True))
        )
        provider = LlamaCppProvider(provider_config)
    if provider.provider_identity != selection.expected_identity:
        if isinstance(provider, LlamaCppProvider):
            provider.close()
        raise ValueError("constructed provider identity does not match expected_identity")
    if provider.effective_configuration_json != selection.expected_effective_configuration_json:
        if isinstance(provider, LlamaCppProvider):
            provider.close()
        raise ValueError("constructed provider effective configuration does not match preflight")
    return provider


def execute_prepared(
    prepared: PreparedExperiment,
    *,
    allow_live_provider: bool = False,
) -> WorkflowExecutionSummary:
    """Execute one prepared experiment and publish only its compact artifact set."""

    protocol = prepared.plan.protocol
    if protocol is not None and protocol.run_tier is RunTier.CONFIRMATORY:
        if prepared.loaded_config.config.provider.kind != "llama_cpp":
            raise ValueError("confirmatory execution requires the live llama_cpp provider")
        if not prepared.provenance.working_tree_clean:
            raise ValueError("confirmatory execution requires a clean source worktree")
    if (
        prepared.loaded_config.config.provider.kind == "llama_cpp"
        and allow_live_provider is not True
    ):
        raise LiveProviderAuthorizationError(
            "llama_cpp workflow execution requires allow_live_provider=True; "
            "no output directory or provider was constructed"
        )
    output_directory = prepared.loaded_config.artifact_root
    output_directory.mkdir(parents=True, exist_ok=True)
    allowed_before_recovery = CLOSED_RUN_ARTIFACTS | SQLITE_RECOVERY_SIDECARS
    unexpected = sorted(
        item.name for item in output_directory.iterdir() if item.name not in allowed_before_recovery
    )
    if unexpected:
        raise ValueError(f"run directory contains unexpected artifacts: {unexpected!r}")

    provider = construct_provider(prepared.loaded_config)
    try:
        manifest = build_run_manifest(
            prepared.plan,
            provider.provider_identity,
            prepared.policy_runtimes,
            prepared.provenance,
        )
        execution = execute_plan(
            prepared.plan,
            manifest,
            provider,
            prepared.policy_runtimes,
            output_directory / "run.sqlite3",
        )
    finally:
        if isinstance(provider, LlamaCppProvider):
            provider.close()
    database_path = output_directory / "run.sqlite3"
    if protocol is not None and protocol.run_tier is RunTier.CONFIRMATORY:
        if not isinstance(provider, LlamaCppProvider):
            raise StoreInvariantError(
                "confirmatory claim finalization requires the digest-bound llama_cpp provider"
            )
        provider.verify_model_artifact()
        result, context = analyze_closed_confirmatory_run(prepared.plan, database_path)
        spec = prepared.plan.confirmatory_analysis
        preregistration = prepared.plan.preregistration
        dataset_seal = prepared.plan.dataset_seal
        dataset_purpose = prepared.plan.dataset_purpose
        if (
            not context.claim_eligible
            or not context.causal_mechanism_validated
            or context.run_manifest_sha256 is None
            or context.run_finalization_sha256 is None
            or spec is None
            or preregistration is None
            or dataset_seal is None
            or dataset_purpose is None
        ):
            raise StoreInvariantError(
                "confirmatory execution did not produce claim-eligible frozen evidence"
            )
        with SQLiteRunStore(database_path) as store:
            run_manifest = store.get_manifest()
            run_finalization = store.get_finalization()
            if run_manifest is None or run_finalization is None:
                raise StoreInvariantError(
                    "confirmatory analysis requires a manifest-bound finalized run"
                )
            analysis_manifest = ScientificAnalysisManifest(
                claim_eligible=True,
                causal_mechanism_validated=True,
                run_manifest_sha256=context.run_manifest_sha256,
                run_finalization_sha256=context.run_finalization_sha256,
                scientific_result_sha256=run_finalization.scientific_result_sha256,
                scientific_identity_sha256=prepared.plan.scientific_identity_sha256,
                preregistration_sha256=preregistration.seal_sha256,
                confirmatory_analysis_contract_sha256=(context.analysis_contract_sha256),
                confirmatory_analysis_spec=spec,
                confirmatory_analysis_spec_sha256=canonical_sha256(spec),
                prompt_family_by_sequence=result.prompt_family_by_sequence,
                prompt_family_design_sha256=result.prompt_family_design_sha256,
                dataset_sha256=prepared.plan.dataset_hash,
                dataset_purpose=dataset_purpose,
                dataset_seal_sha256=dataset_seal.seal_sha256,
                evaluation_input_sha256=context.evaluation_input_sha256,
            )
            store.persist_scientific_analysis(analysis_manifest, result, context=context)
            stored = store.get_scientific_analysis()
            if stored is None or stored.result != result:
                raise StoreInvariantError(
                    "confirmatory scientific analysis was not durably persisted"
                )
            store.verify_integrity()
            store.compact()
    elif prepared.plan.evaluation is not None:
        selection = prepared.loaded_config.config.static_selection_record
        if selection is None:
            raise ValueError("Phase 3 execution lacks frozen static-selection evidence")
        analyze_closed_run(
            prepared.plan,
            selection,
            database_path,
        )
    artifacts = export_closed_run(output_directory)
    return WorkflowExecutionSummary(execution=execution, artifacts=artifacts)


__all__ = [
    "ExecutableProvider",
    "LiveProviderAuthorizationError",
    "PreparedExperiment",
    "WorkflowExecutionSummary",
    "construct_provider",
    "execute_prepared",
    "prepare_experiment",
]

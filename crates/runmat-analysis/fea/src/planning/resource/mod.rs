//! Checked FEA resource estimation, worker admission, and usage reconciliation.

mod accuracy;
mod acoustic;
mod electrical_static;
mod fluid;
pub(crate) mod fluid_transient;
pub use contract::FluidTransientBlockResourceDemand;
pub use contract::{FluidFieldRecoveryResourceDemand, FluidSteadySolveResourceDemand};
pub use fluid::{
    build_fluid_field_recovery_resource_plan, build_fluid_field_recovery_resource_plan_on,
    build_fluid_steady_resource_plan, build_fluid_steady_resource_plan_with,
    derive_fluid_steady_resource_projection, fluid_field_recovery_shape,
    fluid_steady_result_payload_bytes_upper_bound, validate_fluid_steady_prepared_facts,
    FluidSteadyResourceProjection, PreparedFluidSteadyResourceFacts,
};
pub use fluid_transient::{
    build_fluid_transient_block_resource_plan, derive_fluid_transient_block_resource_projection,
    prepare_fluid_transient_resources, FluidTransientBlockResourceProjection,
    PreparedFluidTransientResources,
};
mod electrical_recovery;
mod magnetic_recovery;
pub use electrical_recovery::{
    build_electrical_field_recovery_resource_plan, electrical_field_recovery_partition_count,
    electrical_field_recovery_shape, electrical_recovery_source_validation_workspace,
};
mod statistical_recovery;
pub use statistical_recovery::{
    build_statistical_field_recovery_resource_plan, statistical_field_recovery_shape,
};
mod magnetic_static;
pub(crate) use magnetic_static::build_magnetostatic_resource_plan_with_system;
mod maxwell_harmonic;
mod maxwell_recovery;
mod acoustic_response;
pub use acoustic_response::{
    acoustic_response_result_payload_bytes_upper_bound, acoustic_response_sample_count,
    build_acoustic_response_resource_plan, PreparedAcousticResponseResourceFacts,
};
pub use contract::AcousticResponseResourceDemand;
mod acoustic_recovery;
pub use acoustic::{
    acoustic_harmonic_result_payload_bytes_upper_bound, build_acoustic_resource_plan,
    PreparedAcousticResourceFacts,
};
pub use acoustic_recovery::{
    acoustic_field_recovery_partition_count, acoustic_field_recovery_shape,
    acoustic_recovery_source_validation_workspace, acoustic_response_field_recovery_shape,
    acoustic_response_recovery_source_validation_workspace,
    build_acoustic_field_recovery_resource_plan,
    build_acoustic_response_field_recovery_resource_plan,
};
pub use contract::AcousticFieldRecoveryResourceDemand;
pub use contract::AcousticHarmonicResourceDemand;
pub use contract::ElectricalStaticResourceDemand;
pub use contract::MaxwellFieldRecoveryResourceDemand;
pub use contract::{ElectricalFieldRecoveryResourceDemand, MagneticFieldRecoveryResourceDemand};
pub use contract::{MagneticGmresResourceEnvelope, MagnetostaticResourceDemand};
pub use contract::{MaxwellDirectFactorEnvelope, MaxwellHarmonicResourceDemand};
pub use electrical_static::{
    build_electrical_static_resource_plan, electrical_static_result_payload_bytes_upper_bound,
    electrical_static_solver_admission, PreparedElectricalStaticResourceFacts,
};
pub use magnetic_recovery::{
    build_magnetic_field_recovery_resource_plan, magnetic_field_recovery_partition_count,
    magnetic_field_recovery_shape, magnetic_recovery_source_validation_workspace,
};
pub use magnetic_static::{
    build_magnetostatic_resource_plan, magnetostatic_result_payload_bytes_upper_bound,
    PreparedMagnetostaticResourceFacts,
};
pub use maxwell_harmonic::{
    build_maxwell_harmonic_resource_plan, maxwell_harmonic_result_payload_bytes_upper_bound,
    PreparedMaxwellHarmonicResourceFacts,
};
pub use maxwell_recovery::{
    build_maxwell_field_recovery_resource_plan, maxwell_field_recovery_partition_count,
    maxwell_field_recovery_shape, maxwell_recovery_source_validation_workspace,
};
mod result_publication;
mod state_publication;
pub use contract::ResultPublicationResourceDemand;
pub use result_publication::build_result_publication_resource_plan;
mod static_profiles;
pub use contract::SemanticStatePublicationResourceDemand;
pub use contract::StructuralLinearPipelineResourceDemand;
pub use contract::{
    NonlinearHistoryLimits, NonlinearSolidBlockTransportDemand,
    PreparedNonlinearSolidBlockResourceFacts, StructuralNonlinearPipelineResourceDemand,
    StructuralNonlinearRecoveryResourceDemand,
};
mod nonlinear;
mod numeric_profile;
pub(crate) use nonlinear::{
    build_nonlinear_recovery_resource_plan, build_nonlinear_resource_plan_with_output_limit,
};
pub use nonlinear::{
    build_nonlinear_solid_block_resource_plan, nonlinear_solid_block_output_construction_bytes,
    nonlinear_solid_block_payload_bytes_upper_bound,
};
pub use state_publication::build_fluid_state_publication_resource_plan;
pub use state_publication::build_state_publication_resource_plan;
pub use state_publication::build_state_publication_resource_plan_with_ceiling;
pub use static_profiles::{
    profiled_canonical_output_resource_facts, projected_canonical_output_resource_facts,
    structural_state_profiles_output_resource_facts,
};
mod admission;
pub use accuracy::fluid_accuracy_source_workspace;
pub use accuracy::{
    build_accuracy_resource_plan, build_accuracy_resource_plan_with_source_workspace,
};
mod assembly_sizing;
mod buckling;
mod connector_buffers;
mod contract;
mod estimate;
mod excitation_workspace;
mod facts;
mod field_metadata;
mod initial_fields;
pub(crate) use initial_fields::require_borrowed_initial_field_ceiling;
pub use initial_fields::{
    initial_field_reservation_bytes, initial_field_source_ceiling, require_initial_field_ceiling,
};
mod frequency;
mod local_execution;
mod modal;
mod observation_history_finalization;
mod post_process;
pub use contract::ObservationHistoryFinalizationResourceDemand;
pub use observation_history_finalization::{
    build_observation_history_finalization_resource_plan,
    observation_history_finalization_payload_bytes_upper_bound,
    observation_history_series_reduction_payload_bytes_upper_bound,
};
mod general_solve;
mod preflight;
mod random_response;
mod reconciliation;
mod response_spectrum;
mod scoped_execution;
mod solve_policy;
pub use general_solve::{GeneralSolveResourceControls, GeneralSparseSolveResourceDemand};
mod structural_facts;
pub use structural_facts::structural_resource_derivation_work_upper_bound;
mod symbolic;
mod thermal;
mod thermal_backend_evidence;
pub use thermal::{
    build_thermal_resource_plan, build_thermal_resource_plan_from, PreparedThermalResourceFacts,
};
mod thermal_field_construction;
mod thermal_history;
mod thermal_manifest_catalog;
mod thermal_manifest_metadata;
mod thermal_manifest_retained;
pub use thermal_backend_evidence::ThermalBackendEvidenceResourceFacts;
pub use thermal_field_construction::ThermalFieldConstructionResourceFacts;
pub use thermal_manifest_catalog::ThermalManifestCatalogResourceFacts;
pub use thermal_manifest_metadata::ThermalManifestMetadataResourceFacts;
pub use thermal_manifest_retained::ThermalManifestRetainedResourceFacts;
mod thermal_metadata;
pub use thermal_metadata::ThermalMetadataResourceFacts;
mod thermal_observations;
pub use thermal_observations::ThermalObservationResourceFacts;
mod thermal_operator;
#[cfg(test)]
pub(crate) use thermal_operator::tests as thermal_operator_tests;
pub use thermal_operator::{ThermalOperatorResourceFacts, ThermalSolvePath};
mod thermal_preassembly;
mod thermal_realization;
pub use thermal_realization::ThermalRealizationResourceFacts;
mod thermal_sources;
pub use thermal_preassembly::ThermalPreassemblyResourceFacts;
mod thermal_state;
pub use thermal_sources::ThermalSourceEvidenceResourceFacts;
mod thermal_workspace;
pub use thermal_history::ThermalHistoryResourceFacts;
pub use thermal_workspace::ThermalWorkspaceResourceFacts;
mod transient_block;
mod transient_recovery;
mod transient_result;
mod transient_scope;
mod usage;

pub use admission::{
    admit_resource_plan, allocation_matches_plan, reconcile_resource_usage,
    record_resource_graph_usage, record_resource_usage, validate_resource_usage_evidence,
};
pub use buckling::{
    build_structural_buckling_resource_plan, structural_buckling_receipt_payload_bytes_upper_bound,
    structural_buckling_result_payload_bytes_upper_bound, PreparedStructuralBucklingResourceFacts,
};
pub use contract::{
    AcceleratorRequirement, AccuracyEvaluationResourceDemand, AccuracyResourceDemand,
    AcousticFrequencyAssemblyResourceDemand, AssemblyResourcePolicy, ByteCount,
    CheckpointArtifactPolicy, ConstraintResourceFacts, Count, ExcitationWorkspaceResourceFacts,
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, FeaResourcePlanIdentity,
    FeaResourcePlanningInput, FieldRecoveryResourceFacts, ModalSolveSelection, NumericalScalar,
    ObservationAssemblyResourceDemand, ObservationHistoryCheckpointResourceDemand,
    ObservationPartitionPayloadLayout, ObservationPartitionResourceDemand,
    ObservationReductionResourceDemand, ObservationResourceFacts, PartitioningPolicy,
    PhaseResourcePolicy, PreconditionerStorage, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, PreparedPhaseResourceFacts,
    PreparedStructuralModalResourceFacts, ResourceAdmission, ResourceBreakdown,
    ResourcePlanningError, SolveResourcePolicy, SolverAdmissionResourcePolicy, SparseResourceFacts,
    StaticSolvePath, StaticSolveSelection, StructuralBucklingResourceDemand,
    StructuralFrequencyOperatorResourceDemand, StructuralFrequencyResultAssemblyResourceDemand,
    StructuralModalResourceDemand, StructuralModalResponseSpectrumAssemblyResourceDemand,
    StructuralModalResponseSpectrumDirectionResourceDemand, StructuralOperatorResourceDemand,
    StructuralRandomResponseAssemblyResourceDemand, StructuralRandomResponseIntervalResourceDemand,
    StructuralTransientBlockResourceDemand, StructuralTransientFieldRecoveryResourceDemand,
    StructuralTransientResultAssemblyResourceDemand, StructuralTransientScopeResourceDemand,
    SubspaceSolvePath, ThermalFieldSourceResourceFacts, ThermalSolveResourceDemand,
    TopologyResourceFacts, WorkerResourcePolicy, RESIDENT_MEMORY_RESOURCE,
    RESOURCE_ESTIMATION_MODEL_REVISION, RESOURCE_PLAN_MAX_ENCODED_BYTES,
};
pub use estimate::build_resource_plan;
pub use frequency::{
    build_structural_frequency_entry_resource_plan,
    build_structural_frequency_entry_resource_plan_from_compilation,
    build_structural_frequency_recovery_resource_plan,
    build_structural_frequency_recovery_resource_plan_from_compilation,
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_recovery_payload_bytes_upper_bound,
    StructuralFrequencyEntryResourceProjection, StructuralFrequencyRecoveryResourceInput,
};
pub use local_execution::{
    LocalExecutionPlan, LocalExecutionPlanError, LocalThreadCount, LocalWorkChunk,
    CPU_MILLICORES_PER_LOCAL_THREAD, MAX_LOCAL_EXECUTION_THREADS,
};
pub use modal::{
    build_structural_modal_resource_plan, build_structural_modal_resource_plan_from_compilation,
    build_structural_modal_resource_plan_selecting, recorded_modal_solve_path,
    structural_modal_receipt_payload_bytes_upper_bound,
    structural_modal_result_output_resource_facts,
    structural_modal_result_payload_bytes_upper_bound,
};
mod acoustic_frequency_assembly;
pub use acoustic_frequency_assembly::build_acoustic_frequency_assembly_resource_plan;
pub use post_process::{
    build_observation_assembly_resource_plan, build_observation_history_checkpoint_resource_plan,
    build_observation_partition_resource_plan, build_observation_reduction_resource_plan,
    build_structural_frequency_result_assembly_resource_plan,
    observation_artifact_payload_bytes_upper_bound,
    observation_history_checkpoint_payload_bytes_upper_bound,
    observation_partition_payload_bytes_upper_bound,
    observation_reduction_payload_bytes_upper_bound,
    structural_frequency_receipt_payload_bytes_upper_bound,
    structural_frequency_result_payload_bytes_upper_bound, ObservationPartitionResourcePlanning,
    ObservationPartitionSourceWorkspace,
};
#[cfg(test)]
pub(crate) use preflight::build_static_resource_plan;
pub use preflight::recorded_static_solve_path;
pub(crate) use preflight::{build_static_resource_plan_selecting, StaticResourcePreflight};
pub use preflight::{
    build_structural_resource_plan, build_structural_resource_plan_from_compilation,
    require_executable_static_backend,
};
pub use random_response::{
    structural_random_response_assembly_payload_bytes_upper_bound,
    structural_random_response_interval_payload_bytes_upper_bound,
    structural_random_response_receipt_payload_bytes_upper_bound,
    StructuralRandomResponseResourceProjection,
};
pub use response_spectrum::{
    structural_modal_response_spectrum_assembly_payload_bytes_upper_bound,
    structural_modal_response_spectrum_direction_payload_bytes_upper_bound,
    structural_modal_response_spectrum_receipt_payload_bytes_upper_bound,
    PreparedStructuralResponseSpectrumAssemblyOutputs,
    StructuralModalResponseSpectrumResourceProjection,
};
pub(crate) use scoped_execution::local_chunk_active;
pub use scoped_execution::{
    execute_scoped_chunks, execute_scoped_slices_mut, ScopedLocalExecutionError,
};
pub(crate) use structural_facts::verify_assembled_structural_storage;
pub use symbolic::StructuralSymbolicOperators;
pub use transient_block::{
    build_structural_transient_block_resource_plan,
    structural_transient_block_payload_bytes_upper_bound,
};
pub use transient_recovery::build_structural_transient_recovery_resource_plan;
pub use transient_result::{
    build_structural_transient_result_assembly_resource_plan,
    structural_transient_receipt_payload_bytes_upper_bound,
    structural_transient_result_payload_bytes_upper_bound,
};
pub(crate) use transient_scope::build_structural_transient_scope_resource_plan;
#[cfg(test)]
pub(crate) use transient_scope::structural_transient_scope_payload_bytes_upper_bound;
pub use usage::{
    AnalysisResourceUsage, ResourceOverrun, ResourceUsageAttempt, ResourceUsageAttemptOverrun,
    ResourceUsageEvidence, ResourceUsageEvidenceIdentity,
};

#[cfg(test)]
mod frequency_tests;
#[cfg(test)]
mod modal_tests;
#[cfg(test)]
mod tests;

mod structural_recovery;
pub use contract::StructuralFieldRecoveryResourceDemand;
pub use structural_recovery::{
    build_structural_field_recovery_resource_plan, structural_field_recovery_shape,
    StructuralFieldRecoveryPlanning,
};
mod nonlinear_recovery;
mod recovered_field_shape;
pub use contract::NonlinearFieldRecoveryResourceDemand;
pub use nonlinear_recovery::{
    build_nonlinear_field_recovery_resource_plan, nonlinear_field_recovery_shape,
    nonlinear_recovery_workspace,
};

mod constraint_compilation;
pub use constraint_compilation::constraint_compilation_available_bytes;

#[cfg(test)]
mod compiler_test_support;

/// Persistent CPU worker stack reservation shared with the execution provider.
pub const CPU_POOL_STACK_BYTES_PER_THREAD: u64 = runmat_cpu_execution::WORKER_STACK_BYTES;

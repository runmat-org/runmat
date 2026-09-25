//! Checked FEA resource estimation, worker admission, and usage reconciliation.

mod admission;
mod assembly_sizing;
mod buckling;
mod connector_buffers;
mod contract;
mod estimate;
mod facts;
mod frequency;
mod local_execution;
mod modal;
mod post_process;
mod preflight;
mod random_response;
mod reconciliation;
mod response_spectrum;
mod scoped_execution;
mod solve_policy;
mod structural_facts;
mod symbolic;
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
    AcceleratorRequirement, AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount,
    CheckpointArtifactPolicy, ConstraintResourceFacts, Count, FeaResourceDemand, FeaResourcePhase,
    FeaResourcePlan, FeaResourcePlanIdentity, FeaResourcePlanningInput, FieldRecoveryResourceFacts,
    NumericalScalar, ObservationAssemblyResourceDemand, ObservationHistoryCheckpointResourceDemand,
    ObservationPartitionPayloadLayout, ObservationPartitionResourceDemand,
    ObservationReductionResourceDemand, ObservationResourceFacts, PartitioningPolicy,
    PhaseResourcePolicy, PreconditionerStorage, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, PreparedPhaseResourceFacts,
    PreparedStructuralModalResourceFacts, ResourceAdmission, ResourceBreakdown,
    ResourcePlanningError, SolveResourcePolicy, SolverAdmissionResourcePolicy, SparseResourceFacts,
    StructuralBucklingResourceDemand, StructuralFrequencyOperatorResourceDemand,
    StructuralFrequencyResultAssemblyResourceDemand, StructuralModalResourceDemand,
    StructuralModalResponseSpectrumAssemblyResourceDemand,
    StructuralModalResponseSpectrumDirectionResourceDemand, StructuralOperatorResourceDemand,
    StructuralRandomResponseAssemblyResourceDemand, StructuralRandomResponseIntervalResourceDemand,
    StructuralTransientBlockResourceDemand, StructuralTransientFieldRecoveryResourceDemand,
    StructuralTransientResultAssemblyResourceDemand, StructuralTransientScopeResourceDemand,
    TopologyResourceFacts, WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
    RESOURCE_PLAN_MAX_ENCODED_BYTES,
};
pub use estimate::build_resource_plan;
pub use frequency::{
    build_structural_frequency_entry_resource_plan,
    build_structural_frequency_recovery_resource_plan,
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_recovery_payload_bytes_upper_bound,
    StructuralFrequencyEntryResourceProjection, StructuralFrequencyRecoveryResourceInput,
};
pub use local_execution::{
    LocalExecutionPlan, LocalExecutionPlanError, LocalThreadCount, LocalWorkChunk,
    CPU_MILLICORES_PER_LOCAL_THREAD, MAX_LOCAL_EXECUTION_THREADS,
};
pub use modal::{
    build_structural_modal_resource_plan, structural_modal_receipt_payload_bytes_upper_bound,
    structural_modal_result_payload_bytes_upper_bound,
};
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
};
pub use preflight::build_structural_resource_plan;
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
    StructuralModalResponseSpectrumResourceProjection,
};
pub use scoped_execution::{
    execute_scoped_chunks, execute_scoped_slices_mut, ScopedLocalExecutionError,
};
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

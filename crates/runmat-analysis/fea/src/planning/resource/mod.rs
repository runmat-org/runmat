//! Checked FEA resource estimation, worker admission, and usage reconciliation.

mod admission;
mod contract;
mod estimate;
mod facts;
mod frequency;
mod preflight;
mod structural_facts;
mod symbolic;

pub use admission::{
    admit_resource_plan, allocation_matches_plan, reconcile_resource_usage, record_resource_usage,
    validate_resource_usage_evidence,
};
pub use contract::{
    AcceleratorRequirement, AccuracyResourceDemand, AnalysisResourceUsage, AssemblyResourcePolicy,
    ByteCount, CheckpointArtifactPolicy, ConstraintResourceFacts, Count, FeaResourcePhase,
    FeaResourcePlan, FeaResourcePlanIdentity, FeaResourcePlanningInput, NumericalScalar,
    ObservationResourceFacts, PartitioningPolicy, PreconditionerStorage, RecoveryResourcePolicy,
    ResourceAdmission, ResourceBreakdown, ResourceOverrun, ResourcePlanningError,
    ResourceUsageAttempt, ResourceUsageEvidence, ResourceUsageEvidenceIdentity,
    SolveResourcePolicy, SolverAdmissionResourcePolicy, SparseResourceFacts, TopologyResourceFacts,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};
pub use estimate::build_resource_plan;
pub use frequency::{
    build_structural_frequency_entry_resource_plan, StructuralFrequencyEntryResourceProjection,
};
pub use preflight::build_structural_resource_plan;
pub use symbolic::StructuralSymbolicOperators;

#[cfg(test)]
mod frequency_tests;
#[cfg(test)]
mod tests;

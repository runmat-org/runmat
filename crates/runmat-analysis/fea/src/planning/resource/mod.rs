//! Checked FEA resource estimation, worker admission, and usage reconciliation.

mod admission;
mod contract;
mod estimate;
mod facts;
mod preflight;
mod symbolic;

pub use admission::{admit_resource_plan, reconcile_resource_usage};
pub use contract::{
    AcceleratorRequirement, AccuracyResourceDemand, AnalysisResourceUsage, AssemblyResourcePolicy,
    ByteCount, CheckpointArtifactPolicy, ConstraintResourceFacts, Count, FeaResourcePlan,
    FeaResourcePlanIdentity, FeaResourcePlanningInput, NumericalScalar, ObservationResourceFacts,
    PartitioningPolicy, PreconditionerStorage, RecoveryResourcePolicy, ResourceAdmission,
    ResourceBreakdown, ResourceOverrun, ResourcePlanningError, SolveResourcePolicy,
    SparseResourceFacts, TopologyResourceFacts, WorkerResourcePolicy,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};
pub use estimate::build_resource_plan;
pub use preflight::build_structural_resource_plan;
pub use symbolic::StructuralSymbolicOperators;

#[cfg(test)]
mod tests;

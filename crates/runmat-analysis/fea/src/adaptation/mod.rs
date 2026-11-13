mod solved_source;
pub use solved_source::{
    build_structural_adaptation_iteration_from_solved_results, StructuralAdaptationSolvedIteration,
    StructuralAdaptationSolvedIterationInput, StructuralAdaptationSolvedSource,
    StructuralAdaptationSolvedSourceInput, StructuralAdaptationSourceError,
    StructuralAdaptationSourceLimits,
};
mod field_transfer;
mod iteration;
pub(crate) mod structural_recovery;

pub use field_transfer::{
    measure_solver_field_transfer_error, transfer_solver_field, SolverFieldTransferError,
    SolverFieldTransferErrorEvidence, SolverFieldTransferEvidence, SolverFieldTransferMethod,
    SolverFieldTransferResult,
};
pub use iteration::{
    build_structural_adaptation_iteration, StructuralAdaptationConstraintLineage,
    StructuralAdaptationConvergenceDecision, StructuralAdaptationDecisionStatus,
    StructuralAdaptationIteration, StructuralAdaptationIterationError,
    StructuralAdaptationIterationInput, StructuralAdaptationSolverResult, StructuralTargetQuantity,
    STRUCTURAL_ADAPTATION_ITERATION_SCHEMA_VERSION,
};
pub use structural_recovery::{
    estimate_structural_recovery_error, StructuralRecoveryEstimate,
    StructuralRecoveryEstimatorError, StructuralRecoveryEstimatorOptions,
    StructuralRecoveryIndicator, StructuralRecoveryNorm, StructuralRecoveryStatistics,
};

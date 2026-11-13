//! Shared geometry-control failure categories at meshing execution boundaries.
use runmat_geometry_core::GeometryEvaluationErrorKind;
use runmat_meshing_core::MeshingFailureCategory;

pub(crate) fn category(kind: GeometryEvaluationErrorKind) -> MeshingFailureCategory {
    match kind {
        GeometryEvaluationErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
        GeometryEvaluationErrorKind::TimeBudgetExceeded => {
            MeshingFailureCategory::TimeBudgetExceeded
        }
        GeometryEvaluationErrorKind::AllocationBudgetExceeded => {
            MeshingFailureCategory::MemoryBudgetExceeded
        }
        GeometryEvaluationErrorKind::SearchWorkBudgetExceeded
        | GeometryEvaluationErrorKind::BudgetExceeded => {
            MeshingFailureCategory::SearchWorkBudgetExceeded
        }
        GeometryEvaluationErrorKind::IterationBudgetExceeded => {
            MeshingFailureCategory::IterationBudgetExceeded
        }
        GeometryEvaluationErrorKind::UnknownEvaluator
        | GeometryEvaluationErrorKind::ParameterOutsideDomain
        | GeometryEvaluationErrorKind::InconsistentGeometry => {
            MeshingFailureCategory::InvalidGeometry
        }
        GeometryEvaluationErrorKind::ProjectionDidNotConverge
        | GeometryEvaluationErrorKind::ClassificationDidNotConverge
        | GeometryEvaluationErrorKind::KernelUnavailable
        | GeometryEvaluationErrorKind::KernelFailure
        | GeometryEvaluationErrorKind::InvalidResult => MeshingFailureCategory::NumericalFailure,
    }
}

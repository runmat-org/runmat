//! Numerical preparation receives the existing authored ceiling; placement stays outside.
use crate::{
    constraints::{CanonicalStructuralConstraintPlan, ConstraintPlanError},
    contracts::FeaRunError,
    planning::resource::ResourcePlanningError,
};
use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;

pub(super) fn compile_inventory<'a>(
    model: &'a AnalysisModel,
    mesh: Option<&'a SolverMeshArtifact>,
    step: &ProcedureStep,
) -> Result<crate::constraints::ConstraintCompilation<'a>, FeaRunError> {
    CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        model,
        mesh,
        step.accuracy.resources.resident_byte_budget(),
    )
    .map_err(|error| match error {
        ConstraintPlanError::WorkspaceLimit {
            required, maximum, ..
        } => FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling {
            resource: "resident memory bytes",
            required,
            maximum,
        }),
        ConstraintPlanError::Cancelled => FeaRunError::Cancelled,
        ConstraintPlanError::Accounting(cause) => FeaRunError::Usage(cause),
        other => FeaRunError::Assembly(other.to_string()),
    })
}

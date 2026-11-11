//! Composed numerical preparation quotes for static execution owners.
use runmat_analysis_core::AnalysisModel;
use runmat_meshing_core::SolverMeshArtifact;
use runmat_simulation_usage::OperationCount;

use crate::{
    assembly::explicit_connectors::connector_plan_projection_work_upper_bound,
    constraints::structural_constraint_normal_projection_work_upper_bound, contracts::FeaRunError,
    planning::resource::structural_resource_derivation_work_upper_bound,
    results::observation::structural_observation_derivation_work_upper_bound,
    topology::AnalysisProjectionError,
};

pub(super) fn resource_derivation_work_upper_bound(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<OperationCount, FeaRunError> {
    structural_resource_derivation_work_upper_bound(model, mesh).map_err(FeaRunError::from)
}

pub(super) fn preparation_work_upper_bound(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<OperationCount, FeaRunError> {
    let observation =
        structural_observation_derivation_work_upper_bound(mesh).map_err(FeaRunError::from)?;
    let connectors =
        connector_plan_projection_work_upper_bound(model, mesh).map_err(|error| match error {
            AnalysisProjectionError::Accounting(cause) => FeaRunError::Usage(cause),
            AnalysisProjectionError::Cancelled => FeaRunError::Cancelled,
            other => FeaRunError::Assembly(other.to_string()),
        })?;
    let constraints = structural_constraint_normal_projection_work_upper_bound(model, mesh)
        .map_err(|error| {
            FeaRunError::from(crate::assembly::LinearAssemblyError::ConstraintPlan(error))
        })?;
    // Plain observation preparation compiles a connector plan; numerical
    // preassembly compiles it before resolving observations. Both then derive
    // resource facts, which include a separate symbolic connector plan.
    observation
        .checked_add(resource_derivation_work_upper_bound(model, mesh)?)
        .map_err(|cause| FeaRunError::Usage(cause.into()))?
        .checked_add(connectors)
        .map_err(|cause| FeaRunError::Usage(cause.into()))?
        .checked_add(constraints)
        .map_err(|cause| FeaRunError::Usage(cause.into()))
}

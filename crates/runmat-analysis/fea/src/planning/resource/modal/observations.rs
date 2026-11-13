//! Re-derive original modal observation plans before trusting transported resource facts.
use super::*;
use crate::{
    post::fields::structural_observable_field_definitions,
    results::observation::{resolve_structural_observation_plan, ObservationLimits},
};

pub(super) fn resolve_modal_observations(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> Result<ResolvedObservationPlan, ResourcePlanningError> {
    let fields = structural_observable_field_definitions()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    resolve_structural_observation_plan(
        workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        workload.model(),
        mesh,
        constraints,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

//! Whole-task original-energy reservation; the numerical proof keeps its original cap.
use super::*;

pub(super) fn original_energy_source_bytes(
    mesh: Option<&SolverMeshArtifact>,
    reduction: &crate::constraints::AffineConstraintReduction,
    prestressed: bool,
    model: &runmat_analysis_core::AnalysisModel,
) -> Result<u64, ResourcePlanningError> {
    let Some(mesh) = mesh.filter(|_| {
        !prestressed && crate::assembly::solver_solid::SolidEnergyPreparation::supports_model(model)
    }) else {
        return Ok(0);
    };
    if crate::constraints::HomogeneousTranslationSelection::from_reduction(
        reduction,
        mesh.topology.nodes.len(),
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
    .is_none()
    {
        return Ok(0);
    }
    crate::assembly::solver_solid::SolidEnergyPreparation::resident_bytes(mesh)
        .map_err(|error| ResourcePlanningError::Invalid(format!("{error:?}")))
}

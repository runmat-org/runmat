//! Original capture and projected-proof inventory; generic sparse factor caps stay unchanged.
use super::*;

pub(super) fn original_energy_source_bytes(
    mesh: Option<&SolverMeshArtifact>,
    reduction: &crate::constraints::AffineConstraintReduction,
    prestressed: bool,
    model: &runmat_analysis_core::AnalysisModel,
    matrix: CoercivityWorkEnvelope,
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
    let capture = crate::assembly::solver_solid::SolidEnergyPreparation::resident_bytes(mesh)
        .map_err(|error| ResourcePlanningError::Invalid(format!("{error:?}")))?;
    let proof = crate::algebra::anchored_energy::SourceEnergyDemand::derive(
        matrix,
        mesh.topology.nodes.len(),
        checked_bytes(
            count(mesh.topology.nodes.len(), "energy projection vertices")?,
            std::mem::size_of::<Option<usize>>() as u64,
            "energy projection map",
        )?,
        mesh.topology
            .volume_elements
            .iter()
            .map(|element| element.node_ids.len()),
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    Ok(capture.max(proof.additional_resident_bytes()))
}

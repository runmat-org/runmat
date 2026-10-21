//! Preserve physical assembly and preload semantics while capturing a local proof source once.
use super::*;
use crate::assembly::{
    assemble_structural_modal_system, assemble_structural_perturbation_system,
    assemble_structural_system_borrowed, StructuralFormulationRequirement,
};

pub(super) fn assemble(
    model: &AnalysisModel,
    excitations: &crate::excitation::RealizedExcitations,
    mesh: Option<&SolverMeshArtifact>,
    prestressed: bool,
    capture: &mut SolidEnergyPreparation,
) -> Result<crate::assembly::AssemblySummary, FeaRunError> {
    if prestressed {
        // A geometric tangent is not certified by the elastic local source.
        assemble_structural_perturbation_system(
            model,
            excitations,
            mesh.cloned(),
            StructuralFormulationRequirement::SyntheticOperator,
        )
    } else if SolidEnergyPreparation::supports_model(model) {
        assemble_structural_modal_system(model, excitations, mesh, capture)
    } else {
        assemble_structural_system_borrowed(
            model,
            excitations,
            mesh,
            StructuralFormulationRequirement::SyntheticOperator,
        )
    }
    .map_err(|error| FeaRunError::Assembly(error.to_string()))
}

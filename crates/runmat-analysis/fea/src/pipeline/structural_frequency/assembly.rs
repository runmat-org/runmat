use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::{
        complex_structural::{
            assemble_complex_structural_perturbation_system, assemble_complex_structural_system,
            ComplexStructuralAssemblySummary,
        },
        StructuralFormulationRequirement,
    },
    execution::{
        AnalysisExecutionError, AnalysisExecutionResult, CanonicalSemanticState,
        StructuralFrequencyWorkload,
    },
};

pub(super) fn assemble_entry(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    frequency_hz: f64,
    prestress_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<ComplexStructuralAssemblySummary> {
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let linearization = match (workload.prestress_transition(), prestress_state) {
        (None, None) => None,
        (Some(transition), Some(state)) => Some(
            super::super::prestress::structural_linearization_from_state(
                transition,
                state,
                topology_digest,
            )
            .map_err(AnalysisExecutionError::Invalid)?,
        ),
        _ => {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency workload and prestress-state presence differ".into(),
            ));
        }
    };
    let assemble = if linearization.is_some() {
        assemble_complex_structural_perturbation_system
    } else {
        assemble_complex_structural_system
    };
    let mut assembly = assemble(
        workload.model(),
        workload.procedure_step()?,
        frequency_hz,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    if let Some(linearization) = linearization {
        assembly.real =
            super::super::prestress::prestressed_structural_summary(&assembly.real, linearization)
                .map_err(AnalysisExecutionError::Invalid)?;
    }
    Ok(assembly)
}

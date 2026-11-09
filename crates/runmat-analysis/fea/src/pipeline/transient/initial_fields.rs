//! Validate closed initial-field inputs before physical dynamics assembly.
use crate::{
    contracts::FeaRunError, execution::InitialFieldBinding,
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
};
use runmat_analysis_core::{AnalysisModel, ProcedureStep, StructuralModel};
use runmat_meshing_core::SolverMeshArtifact;

pub(super) fn validate(
    model: &AnalysisModel,
    step: &ProcedureStep,
    mesh: Option<&SolverMeshArtifact>,
    fields: &[InitialFieldBinding],
) -> Result<(), FeaRunError> {
    crate::planning::resource::require_initial_field_ceiling(
        fields,
        runmat_canonical_codec::CanonicalLimits::platform_content(65_536, 64),
        step.accuracy.resources.resident_byte_budget(),
    )?;
    crate::execution::validate_initial_field_declarations(model, step, fields)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    if fields.is_empty() {
        return Ok(());
    }
    let empty = StructuralModel {
        nodes: Vec::new(),
        elements: Vec::new(),
        beam_sections: Vec::new(),
        shell_sections: Vec::new(),
    };
    let structural = model.structural.as_ref().unwrap_or(&empty);
    let topology = match mesh {
        Some(mesh) => CanonicalStructuralNodeSpace::derive(mesh, structural),
        None => CanonicalStructuralNodeSpace::derive_explicit(structural),
    }
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let digest = match mesh {
        Some(mesh) if structural.nodes.is_empty() => {
            runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        }
        _ => topology.identity.discretization_digest(),
    };
    crate::execution::validate_initial_field_selection(
        model,
        step,
        digest,
        topology.nodes.len(),
        fields,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))
}

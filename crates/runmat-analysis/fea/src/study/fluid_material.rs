//! Exact Cell-role material authority for the initial homogeneous laminar envelope.
//! This is material admission, not a qualified solve or a native dispatch adapter.
use crate::properties::{compile_property_map, CompiledPropertyMap, PropertyCompilationError};
use runmat_analysis_core::{
    AnalysisModel, CfdDomain, FluidMaterialError, PhysicsFamily, TurbulenceIntensity,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FluidMaterialCompilationError {
    #[error("requires explicit standalone Fluid procedure intent")]
    UnsupportedIntent,
    #[error("invalid canonical solver mesh: {0}")]
    InvalidMesh(String),
    #[error("invalid retained material map: {0}")]
    InvalidPropertyMap(String),
    #[error("compiled material authority differs from original model and mesh")]
    SourceSubstitution,
    #[error("requires nonempty exact Cell-only volume material inventory")]
    NonCellInventory,
    #[error("canonical element {element_id} has no Cell material owner")]
    MissingCell { element_id: u64 },
    #[error("Cell owner differs from canonical element {element_id}")]
    CellIdentityMismatch { element_id: u64 },
    #[error("heterogeneous Newtonian coefficients at element {element_id} are outside the initial envelope")]
    HeterogeneousCoefficients { element_id: u64 },
    #[error("retained CFD domain differs from assigned material authority")]
    RetainedDomainMismatch,
    #[error(transparent)]
    Properties(#[from] PropertyCompilationError),
    #[error(transparent)]
    Material(#[from] FluidMaterialError),
}

pub fn derive_canonical_fluid_material(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
) -> Result<CfdDomain, FluidMaterialCompilationError> {
    if model.sequence.steps.is_empty()
        || model
            .sequence
            .steps
            .iter()
            .any(|step| step.family != PhysicsFamily::Fluid)
    {
        return Err(FluidMaterialCompilationError::UnsupportedIntent);
    }
    mesh.validate_canonical()
        .map_err(|error| FluidMaterialCompilationError::InvalidMesh(error.to_string()))?;
    properties
        .validate_integrity()
        .map_err(FluidMaterialCompilationError::InvalidPropertyMap)?;
    // Revalidate the retained property authority against original model+mesh. A
    // self-consistent but substituted map is not proof of the authored assignment.
    let expected = compile_property_map(model, Some(mesh))?;
    if &expected != properties {
        return Err(FluidMaterialCompilationError::SourceSubstitution);
    }
    if mesh.topology.volume_elements.is_empty()
        || properties.cell_materials().len() != mesh.topology.volume_elements.len()
        || properties.solid_materials().len() != 0
        || properties.beam_properties().len() != 0
        || properties.shell_properties().len() != 0
    {
        return Err(FluidMaterialCompilationError::NonCellInventory);
    }
    let mut coefficients = None;
    for element in &mesh.topology.volume_elements {
        let owner = properties.cell_material(element.stable_identity).ok_or(
            FluidMaterialCompilationError::MissingCell {
                element_id: element.element_id,
            },
        )?;
        if owner.element_id != element.element_id {
            return Err(FluidMaterialCompilationError::CellIdentityMismatch {
                element_id: element.element_id,
            });
        }
        let current = owner.material.require_incompressible_newtonian()?;
        if coefficients.is_some_and(|prior| prior != current) {
            return Err(FluidMaterialCompilationError::HeterogeneousCoefficients {
                element_id: element.element_id,
            });
        }
        coefficients = Some(current);
    }
    let coefficients = coefficients.ok_or(FluidMaterialCompilationError::NonCellInventory)?;
    let domain = CfdDomain {
        density: coefficients.density,
        dynamic_viscosity: coefficients.dynamic_viscosity,
        // Explicit law admission is laminar; this legacy domain flag is NOT an
        // inferred turbulence closure, velocity/length scale, or material default.
        turbulence_intensity: TurbulenceIntensity::new(0.0),
    };
    if model
        .cfd
        .as_ref()
        .is_some_and(|retained| retained != &domain)
    {
        return Err(FluidMaterialCompilationError::RetainedDomainMismatch);
    }
    Ok(domain)
}

#[cfg(test)]
mod tests;

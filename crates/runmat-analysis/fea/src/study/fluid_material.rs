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
    #[error("a fluid region also contains solid-role elements")]
    MixedFluidRegion,
    #[error("retained CFD domain differs from assigned material authority")]
    RetainedDomainMismatch,
    #[error(transparent)]
    Properties(#[from] PropertyCompilationError),
    #[error(transparent)]
    Material(#[from] FluidMaterialError),
}

/// Fluid cell material and the cells that carry it. Standalone flow owns every element;
/// a coupled flow study (conjugate heat transfer, fluid–structure interaction) owns the cell
/// role regions while solid-role elements belong to the other physics.
#[derive(Clone, Debug, PartialEq)]
pub struct FluidCells {
    pub domain: CfdDomain,
    pub selection: crate::topology::FiniteVolumeCellSelection,
}

/// Steps whose flow is solved by the canonical finite-volume operator.
pub fn is_flow_step(step: &runmat_analysis_core::ProcedureStep) -> bool {
    matches!(
        step.family,
        PhysicsFamily::Fluid
            | PhysicsFamily::Coupled(
                runmat_analysis_core::CoupledPhysicsFamily::ConjugateHeatTransfer
                    | runmat_analysis_core::CoupledPhysicsFamily::FluidStructureInteraction
            )
    )
}

pub fn derive_canonical_fluid_material(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
) -> Result<CfdDomain, FluidMaterialCompilationError> {
    derive_fluid_cells(model, mesh, properties).map(|cells| cells.domain)
}

pub fn derive_fluid_cells(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
) -> Result<FluidCells, FluidMaterialCompilationError> {
    if model.sequence.steps.is_empty() || !model.sequence.steps.iter().all(is_flow_step) {
        return Err(FluidMaterialCompilationError::UnsupportedIntent);
    }
    let coupled = model
        .sequence
        .steps
        .iter()
        .any(|step| step.family != PhysicsFamily::Fluid);
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
        || properties.cell_materials().len() == 0
        || properties.beam_properties().len() != 0
        || properties.shell_properties().len() != 0
        || (!coupled
            && (properties.cell_materials().len() != mesh.topology.volume_elements.len()
                || properties.solid_materials().len() != 0))
    {
        return Err(FluidMaterialCompilationError::NonCellInventory);
    }
    let mut coefficients = None;
    let mut regions = Vec::new();
    let mut porous = Vec::new();
    for element in &mesh.topology.volume_elements {
        let Some(owner) = properties.cell_material(element.stable_identity) else {
            if !coupled {
                return Err(FluidMaterialCompilationError::MissingCell {
                    element_id: element.element_id,
                });
            }
            continue;
        };
        if owner.element_id != element.element_id {
            return Err(FluidMaterialCompilationError::CellIdentityMismatch {
                element_id: element.element_id,
            });
        }
        regions.push(element.region_id.clone());
        // A porous-zone material is resistance in the flowing fluid, not a second fluid.
        if let Some(runmat_analysis_core::MaterialFluidModel::PorousMedium { law }) =
            owner.material.fluid
        {
            porous.push((element.region_id.clone(), law));
            continue;
        }
        let current = owner.material.require_incompressible_fluid()?;
        if coefficients.is_some_and(|prior| prior != current) {
            return Err(FluidMaterialCompilationError::HeterogeneousCoefficients {
                element_id: element.element_id,
            });
        }
        coefficients = Some(current);
    }
    let coefficients = coefficients.ok_or(FluidMaterialCompilationError::NonCellInventory)?;
    let selection = if coupled {
        let selection = crate::topology::FiniteVolumeCellSelection::regions(regions)
            .map_err(|_| FluidMaterialCompilationError::NonCellInventory)?;
        // A fluid region is wholly fluid: no solid-role element may share it.
        if mesh.topology.volume_elements.iter().any(|element| {
            properties.cell_material(element.stable_identity).is_none()
                && selection.includes(&element.region_id)
        }) {
            return Err(FluidMaterialCompilationError::MixedFluidRegion);
        }
        selection
    } else {
        crate::topology::FiniteVolumeCellSelection::AllElements
    };
    let domain = CfdDomain {
        density: coefficients.density,
        dynamic_viscosity: coefficients.dynamic_viscosity,
        // Explicit law admission is laminar; this legacy domain flag is NOT an
        // inferred turbulence closure, velocity/length scale, or material default.
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: coefficients.rheology,
        porous_zones: porous_zones(porous)?,
    };
    if model
        .cfd
        .as_ref()
        .is_some_and(|retained| retained != &domain)
    {
        return Err(FluidMaterialCompilationError::RetainedDomainMismatch);
    }
    Ok(FluidCells { domain, selection })
}

#[cfg(test)]
mod tests;

/// Group porous cells into zones by law; a region carries exactly one law.
fn porous_zones(
    mut cells: Vec<(
        runmat_meshing_core::PersistentEntityId,
        runmat_analysis_core::PorousMediumLaw,
    )>,
) -> Result<Vec<runmat_analysis_core::PorousZone>, FluidMaterialCompilationError> {
    cells.sort_by(|a, b| a.0.cmp(&b.0));
    cells.dedup();
    if cells.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(FluidMaterialCompilationError::NonCellInventory);
    }
    let mut zones: Vec<runmat_analysis_core::PorousZone> = Vec::new();
    for (region, law) in cells {
        match zones.iter_mut().find(|zone| zone.law == law) {
            Some(zone) => zone.regions.push(region),
            None => zones.push(runmat_analysis_core::PorousZone {
                regions: vec![region],
                law,
            }),
        }
    }
    for zone in &mut zones {
        zone.regions.sort();
    }
    Ok(zones)
}

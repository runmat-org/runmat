//! Exact authored Cell material ownership and steady fluid portable intent.
use super::{
    compile_analysis_study, derive_canonical_fluid_material, plan_fluid_study,
    AnalysisStudyCompilationError, AnalysisStudyCompilationInput, CompiledAnalysisStudy,
    FluidMaterialCompilationError, FluidStudyPlanningError, FluidStudyPlanningPolicy,
};
use crate::execution::FluidSteadyWorkload;
use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
use std::collections::BTreeMap;
use thiserror::Error;

/// Compiled definition and exact portable intent, not execution or production admission.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: FluidSteadyWorkload,
}

pub struct FluidStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: FluidStudyPlanningPolicy,
}

pub fn build_fluid_study_product(
    input: FluidStudyProductInput<'_>,
) -> Result<FluidStudyProduct, FluidStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_fluid_study_product(compiled, input.mesh, input.planning)
}

pub fn plan_fluid_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: FluidStudyPlanningPolicy,
) -> Result<FluidStudyProduct, FluidStudyProductError> {
    compiled.validate_against(mesh)?;
    if mesh
        .topology
        .volume_elements
        .iter()
        .any(|element| element.order != ElementOrder::Tet4)
    {
        return Err(FluidStudyProductError::UnsupportedElementOrder);
    }
    derive_canonical_fluid_material(&compiled.model, mesh, &compiled.properties)?;
    let workload = plan_fluid_study(&compiled, planning)?;
    Ok(FluidStudyProduct { compiled, workload })
}

#[derive(Debug, Error)]
pub enum FluidStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Material(#[from] FluidMaterialCompilationError),
    #[error(transparent)]
    Planning(#[from] FluidStudyPlanningError),
    #[error("authored portable steady fluid currently admits Tet4 only")]
    UnsupportedElementOrder,
}

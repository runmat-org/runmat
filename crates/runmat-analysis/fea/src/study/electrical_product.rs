//! Exact compiled definition and portable static electrical intent.
use super::{
    compile_analysis_study, plan_electrical_study, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, CompiledAnalysisStudy, ElectricalStudyPlanningError,
    ElectricalStudyPlanningPolicy,
};
use crate::execution::ElectricalStaticWorkload;
use runmat_analysis_core::{
    AuthoredMaterialRef, MaterialModel, RealizedStudy, StaticElectricalEquation,
};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;
use thiserror::Error;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub struct ElectricalStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: ElectricalStaticWorkload,
}

pub struct ElectricalStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: ElectricalStudyPlanningPolicy,
}

/// Shared exact-geometry compilation and canonical formulation admission, not a solve.
pub fn build_electrical_study_product(
    input: ElectricalStudyProductInput<'_>,
) -> Result<ElectricalStudyProduct, ElectricalStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_electrical_study_product(compiled, input.mesh, input.planning)
}

/// Plan and admit an already compiled (possibly split) study on its mesh.
pub fn plan_electrical_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: ElectricalStudyPlanningPolicy,
) -> Result<ElectricalStudyProduct, ElectricalStudyProductError> {
    compiled.validate_against(mesh)?;
    let workload = plan_electrical_study(&compiled, planning)?;
    let equation = workload
        .equation()
        .map_err(|error| ElectricalStudyProductError::Admission(error.to_string()))?;
    validate_materials(&compiled, equation)?;
    workload
        .formulation(mesh)
        .map_err(|error| ElectricalStudyProductError::Admission(error.to_string()))?;
    Ok(ElectricalStudyProduct { compiled, workload })
}

fn validate_materials(
    compiled: &CompiledAnalysisStudy,
    equation: StaticElectricalEquation,
) -> Result<(), ElectricalStudyProductError> {
    let mut count = 0usize;
    for owner in compiled.properties.solid_materials() {
        count += 1;
        let electrical = owner
            .material
            .require_electrical()
            .map_err(|_| ElectricalStudyProductError::UnsupportedMaterial)?;
        let coefficient = match equation {
            StaticElectricalEquation::Electrostatics => {
                electrical.relative_permittivity.canonical_value()
            }
            StaticElectricalEquation::SteadyConduction => {
                electrical.conductivity_s_per_m.canonical_value()
            }
        };
        // A temperature coefficient is a valid law: standalone conduction evaluates it at
        // the reference temperature; electrothermal conduction at the solved temperature.
        if !coefficient.is_finite()
            || coefficient <= 0.0
            || !electrical.resistive_heating_coefficient.is_finite()
        {
            return Err(ElectricalStudyProductError::UnsupportedMaterial);
        }
    }
    if count == 0 {
        return Err(ElectricalStudyProductError::UnsupportedMaterial);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum ElectricalStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] ElectricalStudyPlanningError),
    #[error("canonical electrical product requires positive constant epsilon or sigma on every volume owner")]
    UnsupportedMaterial,
    #[error("electrical formulation admission failed: {0}")]
    Admission(String),
}

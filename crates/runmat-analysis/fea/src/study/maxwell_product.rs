//! Exact compiled topology and portable harmonic electric-field intent.
use super::{
    compile_analysis_study, plan_maxwell_study, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, CompiledAnalysisStudy, MaxwellStudyPlanningError,
    MaxwellStudyPlanningPolicy,
};
use crate::execution::MaxwellHarmonicWorkload;
use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;
use thiserror::Error;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub struct MaxwellStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: MaxwellHarmonicWorkload,
}

pub struct MaxwellStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: MaxwellStudyPlanningPolicy,
}

/// Shared realization and exact geometry/property compilation, not a solve.
pub fn build_maxwell_study_product(
    input: MaxwellStudyProductInput<'_>,
) -> Result<MaxwellStudyProduct, MaxwellStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_maxwell_study_product(compiled, input.mesh, input.planning)
}

/// Validate sealed post-mesh lineage and the supported positive-loss material,
/// impressed-current and full-exterior PEC envelope before worker admission.
pub fn plan_maxwell_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: MaxwellStudyPlanningPolicy,
) -> Result<MaxwellStudyProduct, MaxwellStudyProductError> {
    compiled.validate_against(mesh)?;
    let workload = plan_maxwell_study(&compiled, planning)?;
    workload
        .formulation(mesh)
        .map_err(|error| MaxwellStudyProductError::Admission(error.to_string()))?;
    workload
        .compile_input(mesh)
        .map_err(|error| MaxwellStudyProductError::Admission(error.to_string()))?;
    Ok(MaxwellStudyProduct { compiled, workload })
}

#[derive(Debug, Error)]
pub enum MaxwellStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] MaxwellStudyPlanningError),
    #[error("maxwell formulation admission failed: {0}")]
    Admission(String),
}

//! Exact compiled definition and portable acoustic cavity-mode or transient intent.
use super::{
    compile_analysis_study, plan_acoustic_response_study, AcousticResponseStudyPlanningError,
    AcousticResponseStudyPlanningPolicy, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, CompiledAnalysisStudy,
};
use crate::execution::AcousticResponseWorkload;
use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct AcousticResponseStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: AcousticResponseWorkload,
}

pub struct AcousticResponseStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: AcousticResponseStudyPlanningPolicy,
}

/// Shared exact-geometry compilation and canonical formulation admission, not a solve.
pub fn build_acoustic_response_study_product(
    input: AcousticResponseStudyProductInput<'_>,
) -> Result<AcousticResponseStudyProduct, AcousticResponseStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_acoustic_response_study_product(compiled, input.mesh, input.planning)
}

/// Validate the sealed post-mesh lineage and canonical formulation before worker admission.
pub fn plan_acoustic_response_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: AcousticResponseStudyPlanningPolicy,
) -> Result<AcousticResponseStudyProduct, AcousticResponseStudyProductError> {
    compiled.validate_against(mesh)?;
    let workload = plan_acoustic_response_study(&compiled, planning)?;
    workload
        .formulation(mesh)
        .map_err(|error| AcousticResponseStudyProductError::Admission(error.to_string()))?;
    Ok(AcousticResponseStudyProduct { compiled, workload })
}

#[derive(Debug, Error)]
pub enum AcousticResponseStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] AcousticResponseStudyPlanningError),
    #[error("acoustic formulation admission failed: {0}")]
    Admission(String),
}

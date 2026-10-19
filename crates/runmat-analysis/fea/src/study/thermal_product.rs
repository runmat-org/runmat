use super::{
    compile_analysis_study, plan_thermal_study_with_fields, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, CompiledAnalysisStudy, ThermalStudyPlanningError,
    ThermalStudyPlanningPolicy,
};
use crate::execution::ThermalWorkload;
use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;
use thiserror::Error;

/// Exact compiled definition and portable thermal intent; not an execution admission.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: ThermalWorkload,
}

pub struct ThermalStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: ThermalStudyPlanningPolicy,
}

/// Compile against exact topology using the shared material and mesh identity boundary.
pub fn build_thermal_study_product(
    input: ThermalStudyProductInput<'_>,
) -> Result<ThermalStudyProduct, ThermalStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_thermal_study_product(compiled, input.mesh, input.planning)
}

/// Derive the complete thermal product after immutable generated inputs have been sealed.
pub fn plan_thermal_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: ThermalStudyPlanningPolicy,
) -> Result<ThermalStudyProduct, ThermalStudyProductError> {
    plan_thermal_study_product_with_fields(compiled, mesh, planning, Vec::new())
}

/// Exact product with closed original fields; storage lookup stays at the host boundary.
pub fn plan_thermal_study_product_with_fields(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: ThermalStudyPlanningPolicy,
    initial_fields: Vec<crate::execution::InitialFieldBinding>,
) -> Result<ThermalStudyProduct, ThermalStudyProductError> {
    compiled.validate_against(mesh)?;
    let workload = plan_thermal_study_with_fields(&compiled, planning, initial_fields)?;
    crate::planning::resource::ThermalObservationResourceFacts::derive(
        mesh,
        &compiled.model.sequence.steps[0],
    )
    .map_err(|error| ThermalStudyPlanningError::Observation(error.to_string()))?;
    Ok(ThermalStudyProduct { compiled, workload })
}

#[derive(Debug, Error)]
pub enum ThermalStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] ThermalStudyPlanningError),
}

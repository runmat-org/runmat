use super::{
    compile_analysis_study, plan_thermal_study, AnalysisStudyCompilationError,
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
    let workload = plan_thermal_study(&compiled, input.planning)?;
    crate::planning::resource::ThermalObservationResourceFacts::derive(
        input.mesh,
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

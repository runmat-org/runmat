//! Exact compiled definition and canonical acoustic frequency plan.

use std::collections::BTreeMap;

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use super::{
    compile_analysis_study, plan_acoustic_study, AcousticStudyPlan, AcousticStudyPlanningError,
    AcousticStudyPlanningPolicy, AnalysisStudyCompilationError, AnalysisStudyCompilationInput,
    CompiledAnalysisStudy,
};

#[cfg(test)]
mod tests;

/// Placement-neutral authored acoustic product created only after exact geometry and meshing.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub plan: AcousticStudyPlan,
}

pub struct AcousticStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: AcousticStudyPlanningPolicy,
}

/// Compile authored intent against the admitted solver mesh and derive portable frequency work.
/// Worker allocation, scheduling, transport, and result publication remain downstream concerns.
pub fn build_acoustic_study_product(
    input: AcousticStudyProductInput<'_>,
) -> Result<AcousticStudyProduct, AcousticStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_acoustic_study_product(compiled, input.mesh, input.planning)
}

/// Derive the complete acoustic product after immutable generated inputs have been sealed.
pub fn plan_acoustic_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: AcousticStudyPlanningPolicy,
) -> Result<AcousticStudyProduct, AcousticStudyProductError> {
    compiled.validate_against(mesh)?;
    validate_canonical_acoustic_materials(&compiled)?;
    let plan = plan_acoustic_study(&compiled, planning)?;
    Ok(AcousticStudyProduct { compiled, plan })
}

fn validate_canonical_acoustic_materials(
    compiled: &CompiledAnalysisStudy,
) -> Result<(), AcousticStudyProductError> {
    let mut count = 0usize;
    for owner in compiled.properties.solid_materials() {
        count = count
            .checked_add(1)
            .ok_or(AcousticStudyProductError::UnsupportedMaterial)?;
        let acoustic = owner
            .material
            .require_acoustic()
            .map_err(|_| AcousticStudyProductError::UnsupportedMaterial)?;
        let density = owner
            .material
            .require_mass_density()
            .map_err(|_| AcousticStudyProductError::UnsupportedMaterial)?
            .canonical_value();
        if !density.is_finite()
            || density <= 0.0
            || !acoustic
                .speed_of_sound_m_per_s
                .canonical_value()
                .is_finite()
            || acoustic.speed_of_sound_m_per_s.canonical_value() <= 0.0
            || !acoustic.damping_ratio.canonical_value().is_finite()
            || !(0.0..1.0).contains(&acoustic.damping_ratio.canonical_value())
        {
            return Err(AcousticStudyProductError::UnsupportedMaterial);
        }
    }
    if count == 0 {
        return Err(AcousticStudyProductError::UnsupportedMaterial);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum AcousticStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] AcousticStudyPlanningError),
    #[error("canonical acoustic product requires lossless fluid density and sound speed on every volume element")]
    UnsupportedMaterial,
}

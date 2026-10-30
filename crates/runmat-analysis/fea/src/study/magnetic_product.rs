//! Exact compiled topology and portable magnetic vector-potential intent.
use super::{
    compile_analysis_study, plan_magnetic_study, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, CompiledAnalysisStudy, MagneticStudyPlanningError,
    MagneticStudyPlanningPolicy,
};
use crate::execution::MagnetostaticWorkload;
use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;
use thiserror::Error;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub struct MagneticStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: MagnetostaticWorkload,
}

pub struct MagneticStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: MagneticStudyPlanningPolicy,
}

/// Shared realization and exact geometry/property compilation, not a solve.
pub fn build_magnetic_study_product(
    input: MagneticStudyProductInput<'_>,
) -> Result<MagneticStudyProduct, MagneticStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_magnetic_study_product(compiled, input.mesh, input.planning)
}

/// Validate sealed post-mesh input lineage and the complete supported magnetic
/// material/current/ground/gauge envelope before downstream worker admission.
pub fn plan_magnetic_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    planning: MagneticStudyPlanningPolicy,
) -> Result<MagneticStudyProduct, MagneticStudyProductError> {
    compiled.validate_against(mesh)?;
    let workload = plan_magnetic_study(&compiled, planning)?;
    let admission =
        |error: &dyn std::fmt::Display| MagneticStudyProductError::Admission(error.to_string());
    // One canonical projection serves native selections, the input and the edge spaces.
    let input = crate::physics::magnetic::compile_canonical_input(
        &workload.model,
        mesh,
        workload.step_id.as_str(),
        workload.relative_residual_target,
    )
    .map_err(|error| admission(&error))?;
    if !workload.model.sequence.steps[0].observations.is_empty() {
        let sampling = crate::results::field::SampledVolumeFieldTopology::derive_from_projection(
            mesh,
            &input.projection,
            crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        )
        .map_err(|error| admission(&error))?;
        workload
            .formulation_with_sampling(mesh, Some(&sampling))
            .map_err(|error| admission(&error))?;
    } else {
        workload
            .formulation(mesh)
            .map_err(|error| admission(&error))?;
    }
    crate::physics::magnetic::validate_projection_envelope(
        &input.projection,
        &input.grounded_faces,
    )
    .map_err(|error| MagneticStudyProductError::Admission(error.to_string()))?;
    Ok(MagneticStudyProduct { compiled, workload })
}

#[derive(Debug, Error)]
pub enum MagneticStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] MagneticStudyPlanningError),
    #[error("magnetic formulation admission failed: {0}")]
    Admission(String),
}

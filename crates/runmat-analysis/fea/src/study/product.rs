use std::collections::BTreeMap;

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_geometry_core::{GeometryDocument, ResolvedGeometryRegions};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use crate::execution::StructuralStudyWorkloads;

use super::{
    compile_analysis_study, plan_structural_study_with_fields, AnalysisStudyCompilationError,
    AnalysisStudyCompilationInput, CompiledAnalysisStudy, StructuralStudyPlanningError,
    StructuralStudyPlanningPolicy,
};

/// Exact placement-neutral structural product created only after geometry and meshing finish.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workloads: StructuralStudyWorkloads,
}

pub struct StructuralStudyProductInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry: &'a GeometryDocument,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub mesh: &'a SolverMeshArtifact,
    pub planning: StructuralStudyPlanningPolicy,
}

/// Compile an authored study against exact topology and derive its portable workloads.
///
/// This is the single post-mesh boundary used by local and remote roots. It deliberately accepts
/// no worker, retry, placement, or transport state.
pub fn build_structural_study_product(
    input: StructuralStudyProductInput<'_>,
) -> Result<StructuralStudyProduct, StructuralStudyProductError> {
    build_structural_study_product_with_fields(input, &[])
}

pub fn build_structural_study_product_with_fields(
    input: StructuralStudyProductInput<'_>,
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<StructuralStudyProduct, StructuralStudyProductError> {
    let compiled = compile_analysis_study(AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    let workloads = plan_structural_study_with_fields(&compiled, input.planning, fields)?;
    Ok(StructuralStudyProduct {
        compiled,
        workloads,
    })
}

#[derive(Debug, Error)]
pub enum StructuralStudyProductError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error(transparent)]
    Planning(#[from] StructuralStudyPlanningError),
}

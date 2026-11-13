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
///
/// Sealed: only [`plan_fluid_study_product`] constructs it, after validating the compiled
/// study against its mesh (canonical content and properties) and planning the workload under
/// `planning`. Holding one is the proof of that planning, so a consumer that receives the
/// product with its mesh compares the bound identities ([`Self::validate_planned_on`])
/// instead of re-compiling properties and re-planning the study.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FluidStudyProduct {
    compiled: CompiledAnalysisStudy,
    workload: FluidSteadyWorkload,
    planning: FluidStudyPlanningPolicy,
}

impl FluidStudyProduct {
    pub fn compiled(&self) -> &CompiledAnalysisStudy {
        &self.compiled
    }

    pub fn workload(&self) -> &FluidSteadyWorkload {
        &self.workload
    }

    pub fn planning(&self) -> FluidStudyPlanningPolicy {
        self.planning
    }

    /// Give up the seal; re-planning is the only way back.
    pub fn into_compiled(self) -> CompiledAnalysisStudy {
        self.compiled
    }

    /// Require that this product was planned on `mesh` under `planning`. The compiled study
    /// is bound to the mesh's canonical digest, which [`plan_fluid_study_product`] checked
    /// against the mesh content; the caller authenticates `mesh` itself.
    pub fn validate_planned_on(
        &self,
        mesh: &SolverMeshArtifact,
        planning: FluidStudyPlanningPolicy,
    ) -> Result<(), FluidStudyProductError> {
        self.validate_on_mesh(mesh)?;
        if self.planning != planning {
            return Err(FluidStudyProductError::PlannedElsewhere);
        }
        Ok(())
    }

    /// Require that this product was planned on `mesh`. Its workload carries the effective
    /// (tightened) numerical and geometry controls the planning produced.
    pub fn validate_on_mesh(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> Result<(), FluidStudyProductError> {
        if self.compiled.solver_mesh_digest != mesh.canonical_digest
            || self.compiled.properties.identity.mesh_digest != Some(mesh.canonical_digest)
        {
            return Err(FluidStudyProductError::PlannedElsewhere);
        }
        Ok(())
    }
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
    Ok(FluidStudyProduct {
        compiled,
        workload,
        planning,
    })
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
    #[error("fluid study product was planned on another mesh or under another policy")]
    PlannedElsewhere,
}

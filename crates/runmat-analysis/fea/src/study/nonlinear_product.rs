//! Authored J2 intent over the exact compiled topology; no placement or numerical execution.
use super::{AnalysisStudyCompilationError, CompiledAnalysisStudy};
use crate::{execution::NonlinearSolidWorkload, solve::nonlinear::NonlinearSolveOptions};
use runmat_analysis_core::{
    AccuracyTarget, AccuracyTolerance, ErrorContributionKind, RequiredAccuracyEvidence,
};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

pub struct NonlinearSolidStudyProductInput<'a> {
    pub study: &'a runmat_analysis_core::RealizedStudy,
    pub geometry: &'a runmat_geometry_core::GeometryDocument,
    pub regions: &'a runmat_geometry_core::ResolvedGeometryRegions,
    pub materials: &'a std::collections::BTreeMap<
        runmat_analysis_core::AuthoredMaterialRef,
        runmat_analysis_core::MaterialModel,
    >,
    pub mesh: &'a SolverMeshArtifact,
    pub controls: NonlinearSolveOptions,
}

pub fn build_nonlinear_solid_study_product(
    input: NonlinearSolidStudyProductInput<'_>,
) -> Result<NonlinearSolidStudyProduct, NonlinearSolidStudyPlanningError> {
    let compiled = super::compile_analysis_study(super::AnalysisStudyCompilationInput {
        study: input.study,
        geometry: input.geometry,
        regions: input.regions,
        materials: input.materials,
        mesh: input.mesh,
    })?;
    plan_nonlinear_solid_study_product(compiled, input.mesh, input.controls)
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearSolidStudyProduct {
    pub compiled: CompiledAnalysisStudy,
    pub workload: NonlinearSolidWorkload,
}

/// Preserve authored declarations and schedule. Relative nonlinear targets may tighten
/// stopping controls, but never replace independent mandatory accuracy evidence.
pub fn plan_nonlinear_solid_study_product(
    compiled: CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    controls: NonlinearSolveOptions,
) -> Result<NonlinearSolidStudyProduct, NonlinearSolidStudyPlanningError> {
    compiled.validate_against(mesh)?;
    // Validate before min(): an authored target must not erase a NaN caller control.
    controls
        .validate_controls()
        .map_err(|e| NonlinearSolidStudyPlanningError::Intent(e.to_string()))?;
    let workload = NonlinearSolidWorkload {
        model: compiled.model.clone(),
        controls,
    };
    workload
        .validate()
        .map_err(|e| NonlinearSolidStudyPlanningError::Intent(e.to_string()))?;
    let mut controls = workload.controls;
    for criterion in &compiled.model.sequence.steps[0].accuracy.criteria {
        if criterion.contribution == ErrorContributionKind::Nonlinear
            && criterion.target == AccuracyTarget::GlobalEquilibrium
            && criterion.evidence == RequiredAccuracyEvidence::ResidualBound
        {
            if let AccuracyTolerance::Relative(target) = criterion.tolerance {
                if !target.is_finite() || target <= 0.0 || target >= 1.0 {
                    return Err(NonlinearSolidStudyPlanningError::InvalidTarget);
                }
                controls.tolerance = controls.tolerance.min(target);
            }
        }
        // The receipt certifies achieved tangent-solve residuals against this target: solve to it.
        if criterion.contribution == ErrorContributionKind::Algebraic
            && criterion.target == AccuracyTarget::GlobalEquilibrium
            && criterion.evidence == RequiredAccuracyEvidence::ResidualBound
        {
            if let AccuracyTolerance::Relative(target) = criterion.tolerance {
                if !target.is_finite() || target <= 0.0 || target >= 1.0 {
                    return Err(NonlinearSolidStudyPlanningError::InvalidTarget);
                }
                controls.linear_tolerance = controls.linear_tolerance.min(target);
            }
        }
    }
    let workload = NonlinearSolidWorkload {
        controls,
        ..workload
    };
    workload
        .validate()
        .map_err(|e| NonlinearSolidStudyPlanningError::Intent(e.to_string()))?;
    Ok(NonlinearSolidStudyProduct { compiled, workload })
}

#[derive(Debug, Error)]
pub enum NonlinearSolidStudyPlanningError {
    #[error(transparent)]
    Compilation(#[from] AnalysisStudyCompilationError),
    #[error("J2 relative nonlinear accuracy target must lie strictly between zero and one")]
    InvalidTarget,
    #[error("unsupported authored J2 intent: {0}")]
    Intent(String),
}

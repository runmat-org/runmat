//! Canonical static scalar-potential finite elements, not reduced EM graph recovery.
//! The initial CPU library envelope is isotropic, constant-property, source-free:
//! div(a grad(phi))=0, a=epsilon or sigma, E=-grad(phi).
//! Production, authored native dispatch, resource admission and restart remain closed.
mod assembly;
mod boundary;
mod recovery;
mod response;
#[cfg(test)]
mod tests;

pub use response::{
    CanonicalElectricalResponse, ElectricalConstitutiveResponse, ElectricalTerminalResponse,
};
use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, ElectromagneticDomain, FormulationRequirement,
    IndependentDomain, PhysicsFamily, ProcedureSettings, RestartPolicy, SolutionSpace,
};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

/// SI vacuum permittivity (CODATA 2022); uncertainty is irrelevant to numerical convergence.
pub const VACUUM_PERMITTIVITY_F_PER_M: f64 = 8.854_187_818_8e-12;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ElectricalStaticError {
    #[error("invalid canonical electrical input: {0}")]
    InvalidInput(String),
    #[error("canonical electrical envelope does not support {0}")]
    Unsupported(String),
    #[error("canonical electrical topology is invalid: {0}")]
    Topology(String),
    #[error("canonical electrical material coverage is invalid: {0}")]
    Material(String),
    #[error("canonical electrical operator is invalid: {0}")]
    Operator(String),
    #[error("canonical electrical boundary is invalid: {0}")]
    Boundary(String),
    #[error("canonical electrical solve did not converge after {iterations} iterations (residual {residual:e})")]
    NotConverged {
        iterations: u32,
        residual: f64,
        diagnostics: Vec<crate::diagnostics::FeaDiagnostic>,
    },
}

/// One authored static/direct step, prescribed scalar potentials/references and natural
/// zero normal flux elsewhere. Uses exact canonical Tet4/Tet10 volume gradients and the
/// shared CPU sparse solve. Unsupported input is rejected rather than silently ignored.
pub fn solve_canonical_static(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
) -> Result<CanonicalElectricalResponse, ElectricalStaticError> {
    if requirement != FormulationRequirement::Experimental {
        return Err(ElectricalStaticError::Unsupported(
            "production qualification".into(),
        ));
    }
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(ElectricalStaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    runmat_analysis_core::validate_model(model)
        .map_err(|error| ElectricalStaticError::InvalidInput(error.to_string()))?;
    let Some(ElectromagneticDomain::Static { equation }) = &model.electromagnetic else {
        return Err(ElectricalStaticError::InvalidInput(
            "explicit static electrical domain required".into(),
        ));
    };
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| ElectricalStaticError::InvalidInput("step not found".into()))?;
    if step.family != PhysicsFamily::Electromagnetic
        || step.procedure != AnalysisProcedure::Static
        || step.solution_space != SolutionSpace::Direct
        || step.independent_domain != IndependentDomain::Static
        || step.settings != ProcedureSettings::None
        || step.restart != RestartPolicy::Fresh
        || !step.state_inputs.is_empty()
    {
        return Err(ElectricalStaticError::Unsupported(
            "non-fresh static/direct electrical step".into(),
        ));
    }
    if model.structural.is_some()
        || model.thermal.is_some()
        || model.thermo_mechanical.is_some()
        || model.electro_thermal.is_some()
        || model.cfd.is_some()
        || !model.connectors.is_empty()
        || !model.interfaces.is_empty()
        || !model.initial_conditions.is_empty()
    {
        return Err(ElectricalStaticError::Unsupported(
            "coupled or initial-state model".into(),
        ));
    }
    if !model.excitations.is_empty()
        || !model.excitation_cases.is_empty()
        || !model.boundaries.is_empty()
    {
        return Err(ElectricalStaticError::Unsupported(
            "source actions or explicit natural/Robin boundaries".into(),
        ));
    }
    let projection = crate::topology::AnalysisProjection::derive(mesh)
        .map_err(|error| ElectricalStaticError::Topology(error.to_string()))?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| ElectricalStaticError::Material(error.to_string()))?;
    let assembled = assembly::ElectricalAssembly::assemble(projection, &properties, *equation)?;
    let assembled = assembled.with_constraints(model, mesh, step)?;
    response::solve(assembled, *equation, residual_target)
}

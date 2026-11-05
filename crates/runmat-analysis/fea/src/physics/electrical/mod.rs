//! Canonical static scalar-potential finite elements, not reduced EM graph recovery.
//! The initial CPU library envelope is isotropic, constant-property, source-free:
//! div(a grad(phi))=0, a=epsilon or sigma, E=-grad(phi).
//! Production, authored native dispatch, resource admission and restart remain closed.
mod assembly;
mod boundary;
mod recovery;
mod response;
pub(crate) use response::closure_target;
pub(crate) mod sources;
mod terminals;
#[cfg(test)]
pub(crate) mod tests;

pub use response::{
    CanonicalElectricalResponse, ElectricalConstitutiveResponse, ElectricalOperatorProof,
    ElectricalTerminalResponse,
};
use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, ElectromagneticDomain, IndependentDomain, PhysicsFamily,
    ProcedureSettings, ProcedureStep, RestartPolicy, SolutionSpace, StateCompleteness, StateKind,
};
use runmat_meshing_core::SolverMeshArtifact;
pub use sources::{CanonicalElectricalSource, ElectricalChargeSources};
use thiserror::Error;

/// SI vacuum permittivity (CODATA 2022); uncertainty is irrelevant to numerical convergence.
pub use runmat_types::physical::VACUUM_ELECTRIC_PERMITTIVITY as VACUUM_PERMITTIVITY_F_PER_M;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ElectricalStaticError {
    #[error("electrical assembly numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
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
    #[error(
        "canonical electrical solve did not converge after {iterations} iterations (residual {residual:e}){}",
        diagnostics
            .iter()
            .map(|diagnostic| format!("; {}: {}", diagnostic.code, diagnostic.message))
            .collect::<String>()
    )]
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
    residual_target: f64,
) -> Result<CanonicalElectricalResponse, ElectricalStaticError> {
    solve_canonical_static_with_admission(
        model,
        mesh,
        step_id,
        residual_target,
        None,
        None,
        crate::excitation::ExcitationSamplePoint::Static,
    )
}

/// A planned solve: the admitted proof envelope and the planned strategy (sparse Cholesky
/// when `direct`, AMG-preconditioned CG otherwise).
#[derive(Clone, Copy)]
pub(crate) struct PlannedElectricalSolve<'a> {
    pub admission: &'a crate::planning::resource::SolverAdmissionResourcePolicy,
    pub direct: bool,
}

impl PlannedElectricalSolve<'_> {
    pub(crate) fn strategy(&self) -> crate::solve::preconditioner::SpdPreconditionerKind {
        if self.direct {
            crate::solve::preconditioner::SpdPreconditionerKind::Direct
        } else {
            crate::solve::preconditioner::SpdPreconditionerKind::Amg
        }
    }
}

pub(crate) fn solve_canonical_static_with_admission(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    residual_target: f64,
    admission: Option<PlannedElectricalSolve<'_>>,
    element_conductivity_s_per_m: Option<&[f64]>,
    sample: crate::excitation::ExcitationSamplePoint<'_>,
) -> Result<CanonicalElectricalResponse, ElectricalStaticError> {
    let step = validate_canonical_input(model, step_id, residual_target)?;
    if terminals::has_floating_terminals(model) {
        return terminals::solve(model, step_id, sample.clone(), |fixed| {
            solve_canonical_static_with_admission(
                fixed,
                mesh,
                step_id,
                residual_target,
                admission,
                element_conductivity_s_per_m,
                sample.clone(),
            )
        });
    }
    let Some(ElectromagneticDomain::Static { equation }) = &model.electromagnetic else {
        unreachable!("validated static electrical domain")
    };
    let projection =
        crate::topology::AnalysisProjection::derive(mesh).map_err(|error| match error {
            crate::topology::AnalysisProjectionError::Accounting(error) => {
                ElectricalStaticError::Accounting(error)
            }
            other => ElectricalStaticError::Topology(other.to_string()),
        })?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| ElectricalStaticError::Material(error.to_string()))?;
    let assembled = assembly::ElectricalAssembly::assemble(
        projection,
        &properties,
        *equation,
        element_conductivity_s_per_m,
    )?;
    let mut assembled = assembled.with_constraints(model, mesh, step, sample)?;
    assembled.charge_sources = sources::assemble(model, mesh, step, &assembled.projection, sample)?;
    response::solve(assembled, *equation, residual_target, admission)
}

/// Steady conduction inside a coupled electrothermal solve: the conductivity of every
/// element is supplied by the caller (the material law evaluated at the element
/// temperature) and terminal sources are sampled at the coupled step's time.
pub(crate) fn solve_conduction_with_conductivity(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    residual_target: f64,
    sample: crate::excitation::ExcitationSamplePoint<'_>,
    element_conductivity_s_per_m: &[f64],
) -> Result<CanonicalElectricalResponse, ElectricalStaticError> {
    let step = validate_canonical_input(model, step_id, residual_target)?;
    if terminals::has_floating_terminals(model) {
        return terminals::solve(model, step_id, sample.clone(), |fixed| {
            solve_conduction_with_conductivity(
                fixed,
                mesh,
                step_id,
                residual_target,
                sample.clone(),
                element_conductivity_s_per_m,
            )
        });
    }
    if !matches!(
        model.electromagnetic,
        Some(ElectromagneticDomain::Static {
            equation: runmat_analysis_core::StaticElectricalEquation::SteadyConduction
        })
    ) {
        return Err(ElectricalStaticError::InvalidInput(
            "coupled conduction requires a steady-conduction model".into(),
        ));
    }
    let projection =
        crate::topology::AnalysisProjection::derive(mesh).map_err(|error| match error {
            crate::topology::AnalysisProjectionError::Accounting(error) => {
                ElectricalStaticError::Accounting(error)
            }
            other => ElectricalStaticError::Topology(other.to_string()),
        })?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| ElectricalStaticError::Material(error.to_string()))?;
    let equation = runmat_analysis_core::StaticElectricalEquation::SteadyConduction;
    let assembled = assembly::ElectricalAssembly::assemble(
        projection,
        &properties,
        equation,
        Some(element_conductivity_s_per_m),
    )?;
    let mut assembled = assembled.with_constraints(model, mesh, step, sample)?;
    assembled.charge_sources = sources::assemble(model, mesh, step, &assembled.projection, sample)?;
    response::solve(assembled, equation, residual_target, None)
}

/// Checks intent without allocating topology or assembling an operator. This numerical
/// entry does not implement sequence execution, observations or state publication.
pub(crate) fn validate_canonical_input<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
    residual_target: f64,
) -> Result<&'a ProcedureStep, ElectricalStaticError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(ElectricalStaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    runmat_analysis_core::validate_model(model)
        .map_err(|error| ElectricalStaticError::InvalidInput(error.to_string()))?;
    let Some(ElectromagneticDomain::Static { .. }) = &model.electromagnetic else {
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
        || step.state_outputs.iter().any(|output| {
            output.kind != StateKind::ElectromagneticStatic
                || output.completeness != StateCompleteness::Complete
        })
        || !step.activations.is_empty()
        || model.sequence.steps.len() != 1
        || !model.sequence.dependencies.is_empty()
        || !model.sequence.transitions.is_empty()
        || !model.sequence.branches.is_empty()
    {
        return Err(ElectricalStaticError::Unsupported(
            "non-standalone static/direct electrical step or unsupported lifecycle".into(),
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
    if !model.boundaries.is_empty()
        || model.excitations.iter().any(|action| {
            !matches!(
                action,
                runmat_analysis_core::ExcitationAction::ElectricChargeDensity(_)
                    | runmat_analysis_core::ExcitationAction::ElectricCurrent(_)
                    | runmat_analysis_core::ExcitationAction::TerminalVoltage(_)
                    | runmat_analysis_core::ExcitationAction::ElectricCharge(_)
                    | runmat_analysis_core::ExcitationAction::CurrentDensity(_)
            )
        })
    {
        return Err(ElectricalStaticError::Unsupported(
            "unsupported electrical source or natural/Robin boundary".into(),
        ));
    }
    let equation = match &model.electromagnetic {
        Some(ElectromagneticDomain::Static { equation }) => *equation,
        _ => unreachable!("validated static electrical domain"),
    };
    let has = |current: bool| {
        model.excitations.iter().any(|action| {
            matches!(
                (current, action),
                (
                    true,
                    runmat_analysis_core::ExcitationAction::ElectricCurrent(_)
                        | runmat_analysis_core::ExcitationAction::CurrentDensity(_)
                ) | (
                    false,
                    runmat_analysis_core::ExcitationAction::ElectricChargeDensity(_)
                )
            )
        })
    };
    if has(false) && equation != runmat_analysis_core::StaticElectricalEquation::Electrostatics {
        return Err(ElectricalStaticError::Unsupported(
            "volume charge belongs to electrostatics; steady conduction takes terminal currents"
                .into(),
        ));
    }
    if has(true) && equation != runmat_analysis_core::StaticElectricalEquation::SteadyConduction {
        return Err(ElectricalStaticError::Unsupported(
            "terminal current belongs to steady conduction".into(),
        ));
    }
    Ok(step)
}

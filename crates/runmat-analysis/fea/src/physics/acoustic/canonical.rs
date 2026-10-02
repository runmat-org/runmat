//! Canonical pressure acoustics, with peak phasors and exp(+i omega t).
//!
//! This CPU library envelope is not admitted as a production or remote workload.
//! K_ij = integral grad(N_i).grad(N_j)/rho; M_ij = integral N_i N_j/(rho c²).
//! A = K - omega² M + i omega B, B_ij = integral N_i N_j/Z.
//! An outward normal velocity supplies b = -i omega integral N_i v_n.
//! Uniform normally incoming pressure supplies b=2 i omega integral N_i Y p_inc,
//! with its own matching outgoing operator. Reported input is net incoming power.

mod assembly;
mod boundary;
mod periodic;
mod recovery;
mod response;
#[cfg(test)]
mod tests;

use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, Frequency, IndependentDomain, PhysicsFamily,
    ProcedureSettings, RestartPolicy, SolutionSpace,
};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

use crate::{
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    topology::AnalysisProjection,
};
use assembly::AcousticAssembly;

pub use response::CanonicalAcousticResponse;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticHarmonicPolicy {
    pub preconditioner: crate::contracts::SolverPreconditioner,
    pub relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations: usize,
}

impl Default for AcousticHarmonicPolicy {
    fn default() -> Self {
        Self {
            preconditioner: crate::contracts::SolverPreconditioner::Ilu0,
            relative_tolerance: 1e-9,
            restart: 64,
            maximum_iterations: 4096,
        }
    }
}

impl AcousticHarmonicPolicy {
    pub fn validate(self, frequency: Frequency) -> Result<(), AcousticHarmonicError> {
        let frequency_hz = frequency.hertz;
        let omega = std::f64::consts::TAU * frequency_hz;
        if !frequency_hz.is_finite()
            || !matches!(
                self.preconditioner,
                crate::contracts::SolverPreconditioner::None
                    | crate::contracts::SolverPreconditioner::Ilu0
            )
            || frequency_hz <= 0.0
            || !omega.is_finite()
            || !(omega * omega).is_finite()
            || !self.relative_tolerance.is_finite()
            || self.relative_tolerance <= 0.0
            || self.relative_tolerance >= 1.0
            || self.restart == 0
            || self.maximum_iterations == 0
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "frequency or Krylov policy".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AcousticHarmonicError {
    #[error("invalid canonical acoustic input: {0}")]
    InvalidInput(String),
    #[error("canonical acoustic envelope does not support {0}")]
    Unsupported(String),
    #[error("canonical acoustic topology is invalid: {0}")]
    Topology(String),
    #[error("canonical acoustic material coverage is invalid: {0}")]
    Material(String),
    #[error("canonical acoustic sparse operator is invalid: {0}")]
    Operator(String),
    #[error("canonical acoustic source or boundary is invalid: {0}")]
    Boundary(String),
    #[error("canonical acoustic Krylov solve broke down (a resonant singular system is possible)")]
    Breakdown,
    #[error("canonical acoustic Krylov solve did not converge")]
    NotConverged,
    #[error("canonical acoustic power closure did not converge: iterations={iterations}, relative_residual={relative_residual:e}, power_relative_error={power_relative_error:e}, power_tolerance={power_tolerance:e}")]
    PowerNotConverged {
        iterations: u32,
        relative_residual: f64,
        power_relative_error: f64,
        power_tolerance: f64,
    },
    #[error("canonical acoustic solve was cancelled")]
    Cancelled,
}

/// Solve one explicitly authored acoustic harmonic step on its sealed canonical mesh.
/// Supports lossless isotropic fluids, prescribed complex pressure, outward normal velocity,
/// rigid walls and positive real specific impedance. Unspecified exterior faces are rigid.
/// Matching translated exterior-face periodic/Bloch constraints use unit-phase pressure
/// relations. Uniform normally incoming incident pressure requires a planar face in one
/// homogeneous medium. Rotated/nonmatching pairing, oblique/spatial incident fields,
/// material damping and radiation
/// fail closed.
/// This entry does not advertise workload admission, bounded-resource or release qualification.
pub fn solve_canonical_harmonic(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step_id: &str,
    frequency: Frequency,
    policy: AcousticHarmonicPolicy,
    power_tolerance: f64,
) -> Result<CanonicalAcousticResponse, AcousticHarmonicError> {
    if !power_tolerance.is_finite() || power_tolerance <= 0.0 || power_tolerance >= 1.0 {
        return Err(AcousticHarmonicError::InvalidInput(
            "power closure tolerance".into(),
        ));
    }
    let step = validate_input(model, step_id, frequency, policy)?;
    let frequency_hz = frequency.hertz;
    let omega = std::f64::consts::TAU * frequency_hz;
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
    let properties = crate::properties::compile_solid_property_map(model, mesh)
        .map_err(|error| AcousticHarmonicError::Material(error.to_string()))?;
    // Shared property compilation validates the sealed mesh and geometry revision.
    // GeometryRevisionRef carries source identity rather than an authored geometry label.
    if model.structural.is_some() || !model.connectors.is_empty() || !model.interfaces.is_empty() {
        return Err(AcousticHarmonicError::Unsupported(
            "explicit structural or coupled interfaces".into(),
        ));
    }
    let sample = ExcitationSamplePoint::frequency(frequency_hz);
    let excitations = realize_step_excitations(model, step, sample)
        .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
    let assembly = AcousticAssembly::assemble(projection, &properties, omega)?.with_boundaries(
        model,
        step,
        &excitations,
        sample,
    )?;
    let assembly = assembly.with_periodic_constraints(mesh, model, step, &excitations)?;
    response::solve(assembly, frequency_hz, policy, power_tolerance)
}

pub(crate) fn validate_input<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
    frequency: Frequency,
    policy: AcousticHarmonicPolicy,
) -> Result<&'a runmat_analysis_core::ProcedureStep, AcousticHarmonicError> {
    policy.validate(frequency)?;
    let frequency_hz = frequency.hertz;
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| AcousticHarmonicError::InvalidInput("step not found".into()))?;
    if step.family != PhysicsFamily::Acoustic
        || step.procedure != AnalysisProcedure::HarmonicResponse
    {
        return Err(AcousticHarmonicError::Unsupported(
            "non-acoustic harmonic step".into(),
        ));
    }
    if step.solution_space != SolutionSpace::Direct
        || !step.state_inputs.is_empty()
        || step.settings != ProcedureSettings::None
    {
        return Err(AcousticHarmonicError::Unsupported(
            "reduced solution space, imported state or procedure settings".into(),
        ));
    }
    let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
        return Err(AcousticHarmonicError::InvalidInput(
            "harmonic step requires frequency domain".into(),
        ));
    };
    if !frequencies.contains(&frequency_hz)
        || frequencies.iter().any(|f| !f.is_finite() || *f <= 0.0)
    {
        return Err(AcousticHarmonicError::InvalidInput(
            "frequency is absent from the valid authored schedule".into(),
        ));
    }
    if step.restart != RestartPolicy::Fresh {
        return Err(AcousticHarmonicError::Unsupported(
            "restart policy without restored state".into(),
        ));
    }
    Ok(step)
}

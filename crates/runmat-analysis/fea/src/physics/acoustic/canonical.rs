//! Canonical pressure acoustics, with peak phasors and exp(+i omega t).
//!
//! This CPU library envelope is not admitted as a production or remote workload.
//! K_ij = integral grad(N_i).grad(N_j)/rho; M_ij = integral N_i N_j/(rho c²).
//! A = K - omega² M + i omega B, B_ij = integral N_i N_j/Z.
//! An outward normal velocity supplies b = -i omega integral N_i v_n.
//! Uniform normally incoming pressure supplies b=2 i omega integral N_i Y p_inc,
//! with its own matching outgoing operator. Reported input is net incoming power.

mod assembly;
mod quantity_layout;
mod boundary;
mod periodic;
mod recovery;
mod response;
mod uniqueness;
pub(crate) use uniqueness::uniqueness_admission;
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
    /// None explicitly requests only physical closure in the experimental CPU
    /// library. Authored study planning always requests source-independent proof.
    #[serde(deserialize_with = "deserialize_uniqueness")]
    pub uniqueness: Option<crate::solve::uniqueness::SparseUniquenessPolicy>,
    pub preconditioner: crate::contracts::SolverPreconditioner,
    pub relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations: usize,
}

fn deserialize_uniqueness<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<crate::solve::uniqueness::SparseUniquenessPolicy>, D::Error> {
    // An explicit null is an experimental request, not a missing-field default.
    serde::Deserialize::deserialize(deserializer)
}

impl Default for AcousticHarmonicPolicy {
    fn default() -> Self {
        Self {
            uniqueness: None,
            preconditioner: crate::contracts::SolverPreconditioner::Ilu0,
            relative_tolerance: 1e-9,
            restart: 64,
            maximum_iterations: 4096,
        }
    }
}

impl AcousticHarmonicPolicy {
    pub fn validate(self, frequency: Frequency) -> Result<(), AcousticHarmonicError> {
        if let Some(policy) = self.uniqueness {
            policy
                .validate()
                .map_err(|error| AcousticHarmonicError::InvalidInput(error.to_string()))?;
        }
        let frequency_hz = frequency.canonical_value();
        let omega = frequency
            .angular_frequency()
            .phasor_derivative_rate()
            .canonical_value();
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
    #[error("canonical acoustic uniqueness admission or diagnostic failed: {0}")]
    Uniqueness(crate::solve::uniqueness::SparseUniquenessError),
    #[error("canonical acoustic uniqueness is indeterminate for original operator {operator_identity:?}: {reason:?}, completed_columns={completed_columns}, iterations={iterations}, defect_upper={defect_infinity_upper:?}; this is not a singularity diagnosis")]
    UniquenessIndeterminate {
        operator_identity: runmat_execution::Digest,
        reason: crate::solve::uniqueness::SparseUniquenessUncertainty,
        completed_columns: usize,
        iterations: u64,
        defect_infinity_upper: Option<f64>,
    },
}

/// Solve one explicitly authored acoustic harmonic step on its sealed canonical mesh.
/// Supports lossless isotropic fluids, prescribed complex pressure, outward normal velocity,
/// rigid walls, positive real specific impedance and first-order local outgoing radiation.
/// Unspecified exterior faces are rigid. Radiation is exact for planar normal incidence;
/// it is not a general exterior-domain DtN, spherical radiation or PML condition.
/// Matching translated exterior-face periodic/Bloch constraints use unit-phase pressure
/// relations. Uniform normally incoming incident pressure requires a planar face in one
/// homogeneous medium. Rotated/nonmatching pairing, oblique/spatial incident fields and
/// material damping fail closed.
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
    // Count-only component ceiling precedes projection, assembly and diagnostics.
    let uniqueness = uniqueness_admission(mesh, policy.uniqueness)?;
    let frequency_hz = frequency.canonical_value();
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
    let assembly =
        AcousticAssembly::assemble(projection, &properties, frequency.angular_frequency())?
            .with_boundaries(model, step, &excitations, sample)?;
    let assembly = assembly.with_periodic_constraints(mesh, model, step, &excitations)?;
    response::solve_admitted(assembly, frequency_hz, policy, power_tolerance, uniqueness)
}

pub(crate) fn validate_input<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
    frequency: Frequency,
    policy: AcousticHarmonicPolicy,
) -> Result<&'a runmat_analysis_core::ProcedureStep, AcousticHarmonicError> {
    policy.validate(frequency)?;
    let frequency_hz = frequency.canonical_value();
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

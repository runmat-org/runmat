//! Small-strain canonical-solid equilibrium. Trial point histories are never committed
//! by an iteration or a rejected line-search/cutback candidate.
use runmat_execution::Digest;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use thiserror::Error;

use crate::{
    assembly::{
        nonlinear_solid::{
            CanonicalNonlinearSolidBasis, NonlinearSolidError, NonlinearSolidEvaluation,
            NonlinearSolidState,
        },
        AssemblySummary,
    },
    constraints::AffineConstraintReduction,
};

mod advance;
mod contact;
#[cfg(test)]
pub(crate) mod continuum_qualification;
mod equilibrium;
mod preparation;
mod state;
pub use contact::AcceptedContactEnergy;
pub(crate) use contact::SolidContactCoupling;
pub use state::AcceptedSolidState;
mod linear_correction;
mod policy;
#[cfg(test)]
mod tests;

pub(crate) use policy::SolidNewtonPolicy;

#[derive(Debug, Error)]
pub(crate) enum SolidNewtonError {
    #[error("invalid canonical solid Newton input: {0}")]
    Invalid(String),
    #[error("canonical solid constitutive/equilibrium evaluation failed: {0}")]
    Evaluation(String),
    #[error(transparent)]
    Contact(#[from] crate::physics::structural::contact::ContactOperatorError),
    #[error(transparent)]
    General(#[from] crate::solve::general::GeneralSolveError),
    #[error("canonical solid Newton cancelled")]
    Cancelled,
    #[error("simulation usage admission failed: {0}")]
    Usage(#[from] LedgerError),
    #[error(
        "canonical solid increment at factor {factor} failed after {cutbacks} cutbacks: {reason}"
    )]
    NotConverged {
        factor: f64,
        cutbacks: usize,
        reason: String,
    },
}

pub(crate) struct AcceptedSolidIncrement {
    pub state: AcceptedSolidState,
    pub evaluation: NonlinearSolidEvaluation,
    pub relative_residual: f64,
    /// Undamped pending Newton correction, normalized by current/committed free motion.
    pub relative_increment: f64,
    pub correction_norm: f64,
    pub iterations: usize,
    pub backtracks: usize,
    pub cutbacks: usize,
    /// All trial work since the preceding accepted state, including rejected cutbacks.
    pub attempted_work: SolidAttemptWork,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SolidAttemptWork {
    pub iterations: usize,
    /// Includes the unapplied convergence solve; Newton update caps are unchanged.
    pub linear_solves: usize,
    pub backtracks: usize,
    /// Constitutive assembly evaluations during trial iterations and acceptance replay.
    pub tangent_evaluations: usize,
    /// Largest achieved relative residual of any completed tangent (linear correction) solve,
    /// in the solved reduced equation space. Covers cutback trials too, so it bounds the
    /// algebraic error of every accepted correction.
    pub maximum_linear_relative_residual: f64,
}

pub(crate) struct SolidNewtonProblem<'a> {
    basis: &'a CanonicalNonlinearSolidBasis,
    /// Homogeneous equality reduction; its offset is zero.
    reduction: AffineConstraintReduction,
    /// Prescribed displacement at unit load factor. Supports move with the load factor like
    /// every applied action: the accepted state at factor λ carries `λ * prescribed`.
    prescribed: Vec<f64>,
    physical_force: &'a [f64],
    /// Dead plus reference-configuration follower force: the scale of a relative residual.
    scale_force: Vec<f64>,
    follower: Option<&'a crate::assembly::nonlinear_solid::FollowerLoadOperator>,
    scope_identity: Digest,
    policy: SolidNewtonPolicy,
    contact: Option<contact::PreparedSolidContact<'a>>,
    solver_admission: Option<&'a crate::planning::resource::SolverAdmissionResourcePolicy>,
}

impl SolidNewtonProblem<'_> {
    fn evaluate(
        &self,
        u: &[f64],
        committed: &NonlinearSolidState,
    ) -> Result<NonlinearSolidEvaluation, SolidNewtonError> {
        self.basis.evaluate(u, committed).map_err(|e| match e {
            NonlinearSolidError::Cancelled => SolidNewtonError::Cancelled,
            NonlinearSolidError::Accounting(error) => SolidNewtonError::Usage(error),
            other => SolidNewtonError::Evaluation(other.to_string()),
        })
    }

    fn residual(
        &self,
        evaluation: &NonlinearSolidEvaluation,
        factor: f64,
    ) -> Result<Vec<f64>, SolidNewtonError> {
        let full = measured_solid(
            || {
                solid_count(
                    evaluation
                        .internal_force
                        .len()
                        .min(self.physical_force.len()),
                    2,
                )
            },
            || {
                Ok(evaluation
                    .internal_force
                    .iter()
                    .zip(self.physical_force)
                    .map(|(internal, force)| internal - factor * force)
                    .collect::<Vec<_>>())
            },
        )?;
        if full.iter().any(|v| !v.is_finite()) {
            return Err(SolidNewtonError::Evaluation(
                "physical residual overflow".into(),
            ));
        }
        let reduced = measured_solid(
            || solid_count(self.reduction.recovery_coefficients.len(), 2),
            || {
                self.reduction
                    .transpose_apply(&full)
                    .map_err(constraint_error)
            },
        )?;
        if reduced.iter().any(|v| !v.is_finite()) {
            return Err(SolidNewtonError::Evaluation(
                "reduced residual overflow".into(),
            ));
        }
        Ok(reduced)
    }
}

fn solid_count(elements: usize, per_element: u64) -> Result<OperationCount, LedgerError> {
    OperationCount::from_operations(
        u64::try_from(elements).map_err(|_| LedgerError::CountOverflow)?,
    )
    .checked_mul(per_element)
    .map_err(LedgerError::from)
}

fn measured_solid<R>(
    quote: impl FnOnce() -> Result<OperationCount, LedgerError>,
    run: impl FnOnce() -> Result<R, SolidNewtonError>,
) -> Result<R, SolidNewtonError> {
    if UsageContext::capture().is_none() {
        return run();
    }
    crate::algebra::capacity::measured_sync(UsageCategory::Solve, quote()?, run)
}

fn constraint_error(error: impl std::fmt::Display) -> SolidNewtonError {
    SolidNewtonError::Invalid(error.to_string())
}

// Hypot accumulation avoids norm overflow for otherwise finite inputs.
fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |a, v| a.hypot(*v))
}
fn relative_norm(values: &[f64], scale: f64) -> f64 {
    let value = norm(values);
    if value == 0.0 {
        0.0
    } else if scale > 0.0 {
        value / scale
    } else {
        f64::INFINITY
    }
}

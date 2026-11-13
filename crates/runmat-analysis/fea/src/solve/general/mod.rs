//! Admitted real GENERAL solve. Local operator nonsingularity is neither
//! coercivity nor uniqueness/stability of the surrounding nonlinear problem.
//!
//! One sparse LU factorization (shared sparse layer) is both the nonsingularity certificate
//! (`solve::certificate`) and the direct solver; the solution is accepted only after the
//! residual of the original operator meets the tolerance.
use super::certificate::{certify_nonsingular, CertificateError, NonsingularityCertificate};
use crate::algebra::{spmv, RealSparseMatrix};
use thiserror::Error;
mod envelope;
pub use envelope::GeneralSolveEnvelope;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralSolvePolicy {
    /// Accepted `‖A x − b‖₂ / ‖b‖₂` of the refined direct solution.
    pub relative_tolerance: f64,
    /// Upper limit (exclusive) of the certified contraction bound `‖I − A F⁻¹‖`.
    pub contraction_limit: f64,
}
impl GeneralSolvePolicy {
    pub const fn bounded(relative_tolerance: f64) -> Self {
        Self {
            relative_tolerance,
            contraction_limit: super::certificate::DEFAULT_CONTRACTION_LIMIT,
        }
    }
    pub(crate) fn validate(self) -> Result<(), GeneralSolveError> {
        let open_unit = |value: f64| value.is_finite() && value > 0.0 && value < 1.0;
        if !open_unit(self.relative_tolerance) || !open_unit(self.contraction_limit) {
            return Err(GeneralSolveError::Invalid);
        }
        Ok(())
    }
}
#[derive(Debug, Error, PartialEq)]
pub enum GeneralSolveError {
    #[error("invalid general solve inventory or controls")]
    Invalid,
    #[error("general solve resource arithmetic overflow")]
    Overflow,
    #[error("general solve exceeds admitted work or storage")]
    Admission,
    #[error("general solve cancelled")]
    Cancelled,
    #[error("general operator action or nonfinite arithmetic")]
    Operator,
    #[error("general operator nonsingularity not established: {0}")]
    NonsingularityNotEstablished(String),
    #[error("general direct solve residual {relative_residual:e} above tolerance")]
    NotConverged {
        iterations: u32,
        relative_residual: f64,
    },
}

#[derive(Debug)]
pub struct GeneralSolveResult {
    pub solution: Vec<f64>,
    /// Refinement steps after the factor solve (zero for a zero right-hand side).
    pub iterations: u32,
    pub relative_residual: f64,
    pub certificate: NonsingularityCertificate,
}

/// Admission precedes all numerical allocation. The matrix/RHS are borrowed and additional to
/// this component envelope. The certificate is source-independent and precedes even a zero-RHS
/// return. No SPD label, pivot repair or fallback.
pub fn solve_general_system(
    matrix: &RealSparseMatrix,
    rhs: &[f64],
    policy: GeneralSolvePolicy,
    admitted: GeneralSolveEnvelope,
) -> Result<GeneralSolveResult, GeneralSolveError> {
    cancelled()?;
    if rhs.len() != matrix.rows() || rhs.iter().any(|v| !v.is_finite()) {
        return Err(GeneralSolveError::Invalid);
    }
    let required = GeneralSolveEnvelope::derive(matrix, policy)?;
    if required.resident_bytes > admitted.resident_bytes
        || required.scalar_operations > admitted.scalar_operations
    {
        return Err(GeneralSolveError::Admission);
    }
    let certified =
        certify_nonsingular(matrix, policy.contraction_limit).map_err(certificate_error)?;
    cancelled()?;
    let load = rhs.iter().fold(0.0_f64, |norm, b| norm.hypot(*b));
    if load == 0.0 {
        return Ok(GeneralSolveResult {
            solution: vec![0.0; rhs.len()],
            iterations: 0,
            relative_residual: 0.0,
            certificate: certified.certificate,
        });
    }
    let solution = certified.solve(matrix, rhs).map_err(certificate_error)?;
    cancelled()?;
    let action = spmv(matrix, &solution).map_err(|_| GeneralSolveError::Operator)?;
    let residual = action
        .iter()
        .zip(rhs)
        .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b));
    let relative_residual = residual / load;
    if !relative_residual.is_finite() {
        return Err(GeneralSolveError::Operator);
    }
    if relative_residual > policy.relative_tolerance {
        return Err(GeneralSolveError::NotConverged {
            iterations: 1,
            relative_residual,
        });
    }
    Ok(GeneralSolveResult {
        solution,
        iterations: 1,
        relative_residual,
        certificate: certified.certificate,
    })
}
fn cancelled() -> Result<(), GeneralSolveError> {
    if crate::is_fea_cancelled() {
        Err(GeneralSolveError::Cancelled)
    } else {
        Ok(())
    }
}
fn certificate_error(error: CertificateError) -> GeneralSolveError {
    match error {
        CertificateError::Cancelled => GeneralSolveError::Cancelled,
        CertificateError::Shape => GeneralSolveError::Invalid,
        other => GeneralSolveError::NonsingularityNotEstablished(other.to_string()),
    }
}

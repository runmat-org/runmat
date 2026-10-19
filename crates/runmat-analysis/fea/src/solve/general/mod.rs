//! Admitted real GENERAL solve. Local operator nonsingularity is neither
//! coercivity nor uniqueness/stability of the surrounding nonlinear problem.
use super::{
    krylov::{restarted_gmres, right_ilu::SparseRightIluError, GmresPolicy},
    uniqueness::{
        prepare_sparse_uniqueness, SparseUniquenessError, SparseUniquenessPolicy,
        SparseUniquenessUncertainty, SparseUniquenessVerdict,
    },
};
use crate::algebra::{spmv, RealSparseMatrix, RightInverseCertificate};
use thiserror::Error;
mod envelope;
pub use envelope::GeneralSolveEnvelope;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralSolvePolicy {
    pub relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations: usize,
    pub uniqueness: SparseUniquenessPolicy,
}
impl GeneralSolvePolicy {
    fn krylov(self) -> GmresPolicy {
        GmresPolicy {
            relative_tolerance: self.relative_tolerance,
            restart: self.restart,
            maximum_iterations: self.maximum_iterations,
        }
    }
    pub(crate) fn validate(self) -> Result<(), GeneralSolveError> {
        if self.restart == 0
            || self.maximum_iterations == 0
            || !self.relative_tolerance.is_finite()
            || self.relative_tolerance <= 0.0
            || self.relative_tolerance >= 1.0
        {
            return Err(GeneralSolveError::Invalid);
        }
        self.uniqueness
            .validate()
            .map_err(GeneralSolveError::Uniqueness)
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
    #[error("general operator uniqueness not established: {0:?}")]
    UniquenessNotEstablished(SparseUniquenessUncertainty),
    #[error("general uniqueness check: {0}")]
    Uniqueness(SparseUniquenessError),
    #[error("general correction did not converge after {iterations} iterations")]
    NotConverged {
        iterations: u32,
        relative_residual: f64,
    },
}

#[derive(Debug)]
pub struct GeneralSolveResult {
    pub solution: Vec<f64>,
    pub iterations: u32,
    pub relative_residual: f64,
    pub certificate: RightInverseCertificate,
}

/// Admission precedes all numerical allocation. The matrix/RHS are borrowed and
/// additional to this component envelope. Proof construction is source-independent
/// and precedes even a zero-RHS return. No SPD label, pivot repair or fallback.
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
    let proof_budget =
        super::uniqueness::SparseUniquenessEnvelope::derive(matrix, policy.uniqueness)
            .map_err(GeneralSolveError::Uniqueness)?;
    let (verdict, factor) = prepare_sparse_uniqueness(matrix, policy.uniqueness, proof_budget)
        .map_err(|e| {
            if e == SparseUniquenessError::Cancelled {
                GeneralSolveError::Cancelled
            } else {
                GeneralSolveError::Uniqueness(e)
            }
        })?;
    let certificate = match verdict {
        SparseUniquenessVerdict::Verified { certificate, .. } => certificate,
        SparseUniquenessVerdict::Indeterminate { reason, .. } => {
            return Err(GeneralSolveError::UniquenessNotEstablished(reason))
        }
    };
    cancelled()?;
    let ilu = factor.ok_or(GeneralSolveError::Operator)?;
    let result = restarted_gmres(
        rhs,
        &vec![0.0; rhs.len()],
        policy.krylov(),
        |y| {
            cancelled()?;
            let x = ilu.apply(y, crate::is_fea_cancelled).map_err(ilu_error)?;
            spmv(matrix, &x).map_err(|_| GeneralSolveError::Operator)
        },
        |a, b| {
            let d = crate::algebra::dot_canonical(a, b);
            if d.is_finite() {
                Ok(d)
            } else {
                Err(GeneralSolveError::Operator)
            }
        },
        crate::is_fea_cancelled,
    )?
    .ok_or(GeneralSolveError::Operator)?;
    cancelled()?;
    if result.cancelled {
        return Err(GeneralSolveError::Cancelled);
    }
    let solution = ilu
        .apply(&result.solution, crate::is_fea_cancelled)
        .map_err(ilu_error)?;
    let action = spmv(matrix, &solution).map_err(|_| GeneralSolveError::Operator)?;
    let residual = action
        .iter()
        .zip(rhs)
        .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b));
    let load = rhs.iter().fold(0.0_f64, |norm, b| norm.hypot(*b));
    let relative_residual = if load == 0.0 {
        residual
    } else {
        residual / load
    };
    if !relative_residual.is_finite() {
        return Err(GeneralSolveError::Operator);
    }
    if !result.converged || result.breakdown || relative_residual > policy.relative_tolerance {
        return Err(GeneralSolveError::NotConverged {
            iterations: result.iterations,
            relative_residual,
        });
    }
    cancelled()?;
    Ok(GeneralSolveResult {
        solution,
        iterations: result.iterations,
        relative_residual,
        certificate,
    })
}
fn cancelled() -> Result<(), GeneralSolveError> {
    if crate::is_fea_cancelled() {
        Err(GeneralSolveError::Cancelled)
    } else {
        Ok(())
    }
}
fn ilu_error(e: SparseRightIluError) -> GeneralSolveError {
    match e {
        SparseRightIluError::Cancelled => GeneralSolveError::Cancelled,
        _ => GeneralSolveError::Operator,
    }
}

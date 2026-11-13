//! Source-independent candidate construction followed by outward uniqueness proof.
use super::krylov::{restarted_gmres, right_ilu::SparseRightIlu, GmresFailure, GmresPolicy};
use crate::algebra::{spmv, RealSparseMatrix, RightInverseCheck, RightInverseVerdict};
use thiserror::Error;

mod envelope;
pub use envelope::SparseUniquenessEnvelope;
mod record;
pub use record::SparseUniquenessRecord;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseUniquenessPolicy {
    pub candidate_relative_tolerance: f64,
    pub restart: usize,
    pub maximum_iterations_per_column: usize,
}
impl SparseUniquenessPolicy {
    pub const fn bounded() -> Self {
        Self {
            candidate_relative_tolerance: 1.0e-6,
            restart: 64,
            maximum_iterations_per_column: 128,
        }
    }

    pub fn validate(self) -> Result<(), SparseUniquenessError> {
        if !self.candidate_relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.candidate_relative_tolerance)
            || self.candidate_relative_tolerance == 0.0
            || self.restart == 0
            || self.maximum_iterations_per_column == 0
        {
            return Err(SparseUniquenessError::Invalid);
        }
        Ok(())
    }

    fn krylov(self) -> GmresPolicy {
        GmresPolicy {
            relative_tolerance: self.candidate_relative_tolerance,
            restart: self.restart,
            maximum_iterations: self.maximum_iterations_per_column,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseUniquenessUncertainty {
    FactorUnavailable,
    CandidateNotConverged,
    DefectNotContractive,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SparseUniquenessVerdict {
    Verified {
        certificate: crate::algebra::RightInverseCertificate,
        iterations: u64,
    },
    Indeterminate {
        reason: SparseUniquenessUncertainty,
        completed_columns: usize,
        iterations: u64,
        defect_infinity_upper: Option<f64>,
    },
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SparseUniquenessError {
    #[error("invalid sparse uniqueness shape or candidate policy")]
    Invalid,
    #[error("sparse uniqueness envelope overflow")]
    Overflow,
    #[error("sparse uniqueness admission exceeded")]
    Admission,
    #[error("sparse uniqueness cancelled")]
    Cancelled,
    #[error("sparse uniqueness operator or certificate construction failed")]
    Operator,
    #[error("sparse uniqueness numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
}

/// Additional task workspace/work excludes the original borrowed operator. No
/// physical RHS is consulted, no dense inverse retained, and a successful GMRES
/// candidate alone never establishes uniqueness. Callers must admit this full
/// envelope in their task plan, not merely the proof checker's envelope.
pub fn certify_sparse_uniqueness(
    matrix: &RealSparseMatrix,
    policy: SparseUniquenessPolicy,
    admitted: SparseUniquenessEnvelope,
) -> Result<SparseUniquenessVerdict, SparseUniquenessError> {
    prepare_sparse_uniqueness(matrix, policy, admitted).map(|(verdict, _)| verdict)
}

/// Retain the same original candidate factor for a subsequent physical RHS.
/// The factor is construction-only workspace, never portable proof authority.
pub(crate) fn prepare_sparse_uniqueness(
    matrix: &RealSparseMatrix,
    policy: SparseUniquenessPolicy,
    admitted: SparseUniquenessEnvelope,
) -> Result<(SparseUniquenessVerdict, Option<SparseRightIlu>), SparseUniquenessError> {
    let required = SparseUniquenessEnvelope::derive(matrix, policy)?;
    if required.scalar_operations > admitted.scalar_operations
        || required.resident_bytes > admitted.resident_bytes
    {
        return Err(SparseUniquenessError::Admission);
    }
    cancelled()?;
    let ilu = match SparseRightIlu::factor(matrix, crate::is_fea_cancelled) {
        Ok(ilu) => ilu,
        Err(super::krylov::right_ilu::SparseRightIluError::Cancelled) => {
            return Err(SparseUniquenessError::Cancelled)
        }
        Err(super::krylov::right_ilu::SparseRightIluError::Usage(error)) => {
            return Err(SparseUniquenessError::Usage(error))
        }
        Err(_) => {
            return Ok((
                SparseUniquenessVerdict::Indeterminate {
                    reason: SparseUniquenessUncertainty::FactorUnavailable,
                    completed_columns: 0,
                    iterations: 0,
                    defect_infinity_upper: None,
                },
                None,
            ))
        }
    };
    let checker_envelope =
        crate::algebra::RightInverseWorkEnvelope::derive(matrix.rows(), matrix.values().len())
            .map_err(|_| SparseUniquenessError::Overflow)?;
    let mut checker = RightInverseCheck::new(matrix, checker_envelope)
        .map_err(|_| SparseUniquenessError::Operator)?;
    let mut basis = vec![0.0; matrix.rows()];
    let zero = vec![0.0; matrix.rows()];
    let mut iterations = 0_u64;
    for index in 0..matrix.rows() {
        cancelled()?;
        basis[index] = 1.0;
        let result = restarted_gmres(
            &basis,
            &zero,
            policy.krylov(),
            |y| {
                cancelled()?;
                let x = ilu.apply(y, crate::is_fea_cancelled).map_err(ilu_error)?;
                spmv(matrix, &x).map_err(|_| SparseUniquenessError::Operator)
            },
            crate::is_fea_cancelled,
        )
        .map_err(|error| match error {
            GmresFailure::Operator(error) => error,
            GmresFailure::Usage(error) => SparseUniquenessError::Usage(error),
        })?
        .ok_or(SparseUniquenessError::Operator)?;
        cancelled()?;
        if result.cancelled {
            return Err(SparseUniquenessError::Cancelled);
        }
        iterations = iterations
            .checked_add(u64::from(result.iterations))
            .ok_or(SparseUniquenessError::Overflow)?;
        if !result.converged || result.breakdown {
            return Ok((
                SparseUniquenessVerdict::Indeterminate {
                    reason: SparseUniquenessUncertainty::CandidateNotConverged,
                    completed_columns: index,
                    iterations,
                    defect_infinity_upper: None,
                },
                Some(ilu),
            ));
        }
        let column = ilu
            .apply(&result.solution, crate::is_fea_cancelled)
            .map_err(ilu_error)?;
        checker
            .push_column(index, &column)
            .map_err(|_| SparseUniquenessError::Operator)?;
        basis[index] = 0.0;
    }
    cancelled()?;
    let verdict = match checker
        .finish()
        .map_err(|_| SparseUniquenessError::Operator)?
    {
        RightInverseVerdict::Verified(certificate) => SparseUniquenessVerdict::Verified {
            certificate,
            iterations,
        },
        RightInverseVerdict::Indeterminate {
            defect_infinity_upper,
        } => SparseUniquenessVerdict::Indeterminate {
            reason: SparseUniquenessUncertainty::DefectNotContractive,
            completed_columns: matrix.rows(),
            iterations,
            defect_infinity_upper: Some(defect_infinity_upper),
        },
    };
    Ok((verdict, Some(ilu)))
}

fn cancelled() -> Result<(), SparseUniquenessError> {
    if crate::is_fea_cancelled() {
        Err(SparseUniquenessError::Cancelled)
    } else {
        Ok(())
    }
}

fn ilu_error(error: super::krylov::right_ilu::SparseRightIluError) -> SparseUniquenessError {
    match error {
        super::krylov::right_ilu::SparseRightIluError::Cancelled => {
            SparseUniquenessError::Cancelled
        }
        super::krylov::right_ilu::SparseRightIluError::Usage(error) => {
            SparseUniquenessError::Usage(error)
        }
        _ => SparseUniquenessError::Operator,
    }
}

#[cfg(test)]
mod tests;

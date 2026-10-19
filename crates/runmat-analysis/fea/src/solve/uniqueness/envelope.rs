use super::{SparseUniquenessError as Error, SparseUniquenessPolicy};
use crate::{
    algebra::{RealSparseMatrix, RightInverseWorkEnvelope},
    solve::krylov::{right_ilu::SparseRightIluEnvelope, GmresWorkEnvelope},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseUniquenessEnvelope {
    pub scalar_operations: u64,
    pub resident_bytes: u64,
}

impl SparseUniquenessEnvelope {
    pub fn derive(
        matrix: &RealSparseMatrix,
        policy: SparseUniquenessPolicy,
    ) -> Result<Self, Error> {
        if matrix.rows() != matrix.columns() {
            return Err(Error::Invalid);
        }
        let width = matrix
            .pattern()
            .row_offsets()
            .windows(2)
            .map(|r| r[1] - r[0])
            .max()
            .unwrap_or(0);
        Self::for_counts(
            matrix.rows(),
            matrix.values().len(),
            usize::try_from(width).map_err(|_| Error::Overflow)?,
            policy,
        )
    }

    /// Sampling/factorization-free upper facts for a governed planner. The
    /// complete actual operator is checked again against the admitted envelope.
    pub fn for_counts(
        dimension: usize,
        entries: usize,
        maximum_row_entries: usize,
        policy: SparseUniquenessPolicy,
    ) -> Result<Self, Error> {
        policy.validate()?;
        if dimension == 0
            || maximum_row_entries > dimension
            || !policy.candidate_relative_tolerance.is_finite()
            || policy.candidate_relative_tolerance <= 0.0
            || policy.candidate_relative_tolerance >= 1.0
        {
            return Err(Error::Invalid);
        }
        let n = u64::try_from(dimension).map_err(|_| Error::Overflow)?;
        let z = u64::try_from(entries).map_err(|_| Error::Overflow)?;
        let width = u64::try_from(maximum_row_entries).map_err(|_| Error::Overflow)?;
        if z > mul(n, width)? {
            return Err(Error::Invalid);
        }
        let check =
            RightInverseWorkEnvelope::derive(dimension, entries).map_err(|_| Error::Overflow)?;
        let gmres =
            GmresWorkEnvelope::derive(dimension, policy.krylov()).map_err(|_| Error::Invalid)?;
        // Actual fixed-support ILU elimination scans each retained lower entry
        // against at most max_row upper entries, with <=64 search comparisons.
        let ilu = SparseRightIluEnvelope::for_counts(n, z, width).ok_or(Error::Overflow)?;
        let factor_work = ilu.factor_scalar_operations;
        let application_work = ilu.application_scalar_operations;
        let column_work = add(
            mul(
                gmres.operator_applications_upper_bound,
                add(application_work, mul(z, 4)?)?,
            )?,
            add(
                mul(gmres.dot_calls_upper_bound, mul(n, 8)?)?,
                gmres.recurrence_scalar_operations_upper_bound,
            )?,
        )?;
        let work = add(
            add(check.scalar_operations, factor_work)?,
            mul(n, add(column_work, application_work)?)?,
        )?;
        // Shared GMRES peak + retained ILU support + checker/identity encoding +
        // basis/zero and coexisting recovered/apply vectors, doubled payload allowances.
        let bytes = add(
            add(check.resident_bytes, gmres.resident_bytes_upper_bound)?,
            add(
                ilu.resident_bytes_upper_bound,
                add(mul(add(n, 1)?, 16)?, mul(n, 80)?)?,
            )?,
        )?;
        if bytes > isize::MAX as u64 {
            return Err(Error::Overflow);
        }
        Ok(Self {
            scalar_operations: work,
            resident_bytes: bytes,
        })
    }
}

fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

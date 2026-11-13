//! Complete additional serial solve payload/work: the certifying sparse LU (factor model by
//! counts, monotone in dimension and nonzeros), its estimator solves, the refined solve and the
//! residual check. Source CSR/RHS and caller state excluded.
use super::{GeneralSolveError as Error, GeneralSolvePolicy};
use crate::{algebra::RealSparseMatrix, solve::certificate};
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralSolveEnvelope {
    pub resident_bytes: u64,
    pub scalar_operations: u64,
}
impl GeneralSolveEnvelope {
    pub fn derive(matrix: &RealSparseMatrix, policy: GeneralSolvePolicy) -> Result<Self, Error> {
        if matrix.rows() != matrix.columns() {
            return Err(Error::Invalid);
        }
        Self::for_counts(matrix.rows(), matrix.values().len(), policy)
    }
    pub fn for_counts(n: usize, z: usize, policy: GeneralSolvePolicy) -> Result<Self, Error> {
        policy.validate()?;
        if n == 0 || (z as u128) > (n as u128) * (n as u128) {
            return Err(Error::Invalid);
        }
        let (factor_entries, factor_flops) =
            runmat_sparse_solvers::estimate::quick_direct_estimate(n, z, 3);
        // LU stores L and U.
        let factor_entries = 2.0 * factor_entries;
        let n64 = u64::try_from(n).map_err(|_| Error::Overflow)?;
        let z64 = u64::try_from(z).map_err(|_| Error::Overflow)?;
        let resident_bytes = add(
            certificate::nonsingularity_reservation_bytes(n, z),
            // Solution, action and refinement vectors.
            mul(n64, 64)?,
        )?;
        // Factorization, then every estimator and refinement solve (forward + backward
        // substitution over the factor) plus operator products.
        let solves = add(certificate::ESTIMATOR_SOLVES_UPPER_BOUND, 4)?;
        let per_solve = add(to_u64(2.0 * factor_entries)?, mul(z64, 2)?)?;
        let scalar_operations = add(
            to_u64(2.0 * factor_flops)?,
            add(mul(solves, per_solve)?, mul(n64, 32)?)?,
        )?;
        if resident_bytes >= u64::MAX / 2 || resident_bytes > isize::MAX as u64 {
            return Err(Error::Overflow);
        }
        Ok(Self {
            resident_bytes,
            scalar_operations,
        })
    }
}
fn to_u64(value: f64) -> Result<u64, Error> {
    if value.is_finite() && value >= 0.0 && value < u64::MAX as f64 {
        Ok(value.ceil() as u64)
    } else {
        Err(Error::Overflow)
    }
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

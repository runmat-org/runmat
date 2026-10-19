//! Complete additional serial solve payload/work, including proof
//! construction and RHS application with the same retained factor. Source CSR/RHS and caller state excluded.
use super::{GeneralSolveError as Error, GeneralSolvePolicy};
use crate::{
    algebra::RealSparseMatrix,
    solve::{
        krylov::{right_ilu::SparseRightIluEnvelope, GmresWorkEnvelope},
        uniqueness::SparseUniquenessEnvelope,
    },
};
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
        let width = matrix
            .pattern()
            .row_offsets()
            .windows(2)
            .map(|v| v[1] - v[0])
            .max()
            .unwrap_or(0);
        Self::for_counts(
            matrix.rows(),
            matrix.values().len(),
            usize::try_from(width).map_err(|_| Error::Overflow)?,
            policy,
        )
    }
    pub fn for_counts(
        n: usize,
        z: usize,
        width: usize,
        policy: GeneralSolvePolicy,
    ) -> Result<Self, Error> {
        policy.validate()?;
        let proof = SparseUniquenessEnvelope::for_counts(n, z, width, policy.uniqueness)
            .map_err(Error::Uniqueness)?;
        let gmres = GmresWorkEnvelope::derive(n, policy.krylov()).map_err(|_| Error::Invalid)?;
        let n = u64::try_from(n).map_err(|_| Error::Overflow)?;
        let z = u64::try_from(z).map_err(|_| Error::Overflow)?;
        let ilu = SparseRightIluEnvelope::for_counts(n, z, width as u64).ok_or(Error::Overflow)?;
        // Retained proof, zero/recovered/action vectors, and callbacks coexist.
        // Sum proof and solve peaks conservatively instead of claiming an exact RSS.
        let resident_bytes = add(
            proof.resident_bytes,
            add(
                ilu.resident_bytes_upper_bound,
                add(gmres.resident_bytes_upper_bound, mul(n, 96)?)?,
            )?,
        )?;
        let vector_work = mul(gmres.dot_calls_upper_bound, mul(n, 8)?)?;
        let actions = mul(
            add(gmres.operator_applications_upper_bound, 2)?,
            add(ilu.application_scalar_operations, mul(z, 4)?)?,
        )?;
        let solve_work = add(
            actions,
            add(
                vector_work,
                add(gmres.recurrence_scalar_operations_upper_bound, mul(n, 32)?)?,
            )?,
        )?;
        let scalar_operations = add(proof.scalar_operations, solve_work)?;
        if resident_bytes > isize::MAX as u64 {
            return Err(Error::Overflow);
        }
        Ok(Self {
            resident_bytes,
            scalar_operations,
        })
    }
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

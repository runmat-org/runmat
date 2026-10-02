//! Source-independent sufficient nonsingularity evidence for any real square CSR.
//! B is a candidate inverse, not necessarily an exact factor inverse. If
//! ||I - A B||∞ < 1, the Neumann series proves AB, hence A, invertible.
//! Every scalar product/addition is enclosed outward; no RHS or Krylov residual
//! enters the proof. Failure of this sufficient test is NOT singularity evidence.

use super::RealSparseMatrix;
use runmat_execution::Digest;
use sha2::{Digest as _, Sha256};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RightInverseWorkEnvelope {
    pub scalar_operations: u64,
    pub resident_bytes: u64,
}

impl RightInverseWorkEnvelope {
    /// Check workspace plus conservative numerical-identity codec construction.
    /// Excludes the borrowed matrix and production of each candidate column.
    pub fn derive(dimension: usize, entries: usize) -> Result<Self, RightInverseError> {
        let n = u64::try_from(dimension).map_err(|_| RightInverseError::Overflow)?;
        let z = u64::try_from(entries).map_err(|_| RightInverseError::Overflow)?;
        let scalar_operations = n
            .checked_mul(z.checked_add(n).ok_or(RightInverseError::Overflow)?)
            .and_then(|v| v.checked_mul(16))
            .ok_or(RightInverseError::Overflow)?;
        let resident_bytes = z
            .checked_mul(128)
            .and_then(|v| n.checked_mul(64).and_then(|w| v.checked_add(w)))
            .and_then(|v| v.checked_add(1024 * 1024))
            .ok_or(RightInverseError::Overflow)?;
        Ok(Self {
            scalar_operations,
            resident_bytes,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RightInverseError {
    #[error("right-inverse check requires a nonempty square operator")]
    Shape,
    #[error("right-inverse check envelope overflow")]
    Overflow,
    #[error("right-inverse check exceeds admitted workspace or work")]
    Admission,
    #[error("right-inverse column is missing, out of order or nonfinite")]
    Column,
    #[error("right-inverse operator identity failed")]
    Identity,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RightInverseCertificate {
    operator_identity: Digest,
    inverse_identity: Digest,
    dimension: usize,
    defect_infinity_upper_bits: u64,
}

impl RightInverseCertificate {
    pub fn operator_identity(&self) -> Digest {
        self.operator_identity
    }
    pub fn inverse_identity(&self) -> Digest {
        self.inverse_identity
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn defect_infinity_upper(&self) -> f64 {
        f64::from_bits(self.defect_infinity_upper_bits)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RightInverseVerdict {
    Verified(RightInverseCertificate),
    Indeterminate { defect_infinity_upper: f64 },
}

pub struct RightInverseCheck<'a> {
    matrix: &'a RealSparseMatrix,
    operator_identity: Digest,
    inverse_hash: Sha256,
    row_bounds: Vec<f64>,
    next_column: usize,
}

impl<'a> RightInverseCheck<'a> {
    pub fn new(
        matrix: &'a RealSparseMatrix,
        admitted: RightInverseWorkEnvelope,
    ) -> Result<Self, RightInverseError> {
        if matrix.rows() == 0 || matrix.rows() != matrix.columns() {
            return Err(RightInverseError::Shape);
        }
        let required = RightInverseWorkEnvelope::derive(matrix.rows(), matrix.values().len())?;
        if required.scalar_operations > admitted.scalar_operations
            || required.resident_bytes > admitted.resident_bytes
        {
            return Err(RightInverseError::Admission);
        }
        let operator_identity = matrix
            .numerical_identity()
            .map_err(|_| RightInverseError::Identity)?;
        let mut inverse_hash = Sha256::new();
        inverse_hash.update(b"runmat/right-inverse-columns/1");
        inverse_hash.update((matrix.rows() as u64).to_le_bytes());
        Ok(Self {
            matrix,
            operator_identity,
            inverse_hash,
            row_bounds: vec![0.0; matrix.rows()],
            next_column: 0,
        })
    }

    /// Columns must be complete, in canonical order. Only O(n) check workspace
    /// is retained; the caller can reuse/drop each B column immediately.
    pub fn push_column(&mut self, index: usize, column: &[f64]) -> Result<(), RightInverseError> {
        if index != self.next_column
            || index >= self.matrix.rows()
            || column.len() != self.matrix.rows()
            || column.iter().any(|v| !v.is_finite())
        {
            return Err(RightInverseError::Column);
        }
        for value in column {
            self.inverse_hash.update(value.to_bits().to_le_bytes());
        }
        for (row, bound) in self.row_bounds.iter_mut().enumerate() {
            let mut lo = if row == index { 1.0 } else { 0.0 };
            let mut hi = lo;
            let start = self.matrix.pattern().row_offsets()[row] as usize;
            let end = self.matrix.pattern().row_offsets()[row + 1] as usize;
            for entry in start..end {
                let a = self.matrix.values()[entry];
                let b = column[self.matrix.pattern().column_indices()[entry] as usize];
                if a == 0.0 || b == 0.0 {
                    continue;
                }
                let product = a * b;
                lo = (lo - product.next_up()).next_down();
                hi = (hi - product.next_down()).next_up();
            }
            let absolute = if lo.is_finite() && hi.is_finite() {
                lo.abs().max(hi.abs())
            } else {
                f64::INFINITY
            };
            *bound = (*bound + absolute).next_up();
        }
        self.next_column += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<RightInverseVerdict, RightInverseError> {
        if self.next_column != self.matrix.rows() {
            return Err(RightInverseError::Column);
        }
        let upper = self.row_bounds.into_iter().fold(0.0, f64::max);
        if upper < 1.0 {
            Ok(RightInverseVerdict::Verified(RightInverseCertificate {
                operator_identity: self.operator_identity,
                inverse_identity: Digest::from_bytes(self.inverse_hash.finalize().into()),
                dimension: self.matrix.rows(),
                defect_infinity_upper_bits: upper.to_bits(),
            }))
        } else {
            Ok(RightInverseVerdict::Indeterminate {
                defect_infinity_upper: upper,
            })
        }
    }
}

#[cfg(test)]
mod tests;

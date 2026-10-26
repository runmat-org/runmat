//! Shared count-only admission facts for the fixed-support factor and its apply.
#[cfg(test)]
use crate::algebra::RealSparseMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseRightIluEnvelope {
    pub resident_bytes_upper_bound: u64,
    pub factor_scalar_operations: u64,
    pub application_scalar_operations: u64,
}
impl SparseRightIluEnvelope {
    #[cfg(test)]
    pub(crate) fn derive(matrix: &RealSparseMatrix) -> Option<Self> {
        if matrix.rows() != matrix.columns() {
            return None;
        }
        let width = matrix
            .pattern()
            .row_offsets()
            .windows(2)
            .map(|r| r[1] - r[0])
            .max()
            .unwrap_or(0);
        Self::for_counts(matrix.rows() as u64, matrix.values().len() as u64, width)
    }
    /// Counts may be conservative upper facts; no factorization or matrix allocation.
    pub(crate) fn for_counts(n: u64, entries: u64, maximum_row_entries: u64) -> Option<Self> {
        if entries > n.checked_mul(maximum_row_entries)? {
            return None;
        }
        let word = std::mem::size_of::<usize>() as u64;
        let payload = n
            .checked_add(1)?
            .checked_mul(word)?
            .checked_add(entries.checked_mul(word.checked_add(8)?)?)?
            .checked_add(n.checked_mul(word)?)?;
        // Four retained vectors; doubled payload/header growth, original CSR excluded.
        let resident_bytes_upper_bound = payload
            .checked_mul(2)?
            .checked_add(8 * std::mem::size_of::<Vec<usize>>() as u64)?;
        if resident_bytes_upper_bound > isize::MAX as u64 {
            return None;
        }
        let factor_scalar_operations = entries
            .checked_mul(maximum_row_entries)?
            .checked_mul(128)?
            .checked_add(n.checked_mul(128)?)?;
        let application_scalar_operations =
            entries.checked_mul(8)?.checked_add(n.checked_mul(16)?)?;
        Some(Self {
            resident_bytes_upper_bound,
            factor_scalar_operations,
            application_scalar_operations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::SparseRightIlu;
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
    #[test]
    fn original_factor_vectors_fit_shared_envelope_and_overflow_rejects() {
        let mut a = RealSparseAssembler::square(
            3,
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        );
        for i in 0..3 {
            a.add(i, i, 3.0);
            if i > 0 {
                a.add(i, i - 1, -1.0);
                a.add(i - 1, i, -1.0);
            }
        }
        let matrix = a.finish().unwrap();
        let envelope = SparseRightIluEnvelope::derive(&matrix).unwrap();
        assert_eq!(
            envelope,
            SparseRightIluEnvelope::for_counts(3, 7, 3).unwrap()
        );
        let factor = SparseRightIlu::factor(&matrix, || false).unwrap();
        assert!(factor.retained_bytes() < envelope.resident_bytes_upper_bound);
        assert!(SparseRightIluEnvelope::for_counts(u64::MAX, u64::MAX, u64::MAX).is_none());
        assert!(SparseRightIluEnvelope::for_counts(3, 10, 3).is_none());
    }
}

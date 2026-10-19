//! Exact material-independent support of Tᵀ(A ∪ B)T. Admission precedes scratch;
//! exact counts precede output columns. No tree nodes or dense reduced matrix.
use super::{AffineConstraintReduction, AffineTransposeIndex, ConstraintAlgebraError};
use crate::algebra::{SparsePattern, SparsePatternError, SparsePatternStatistics};
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub struct AffineSupportBudget {
    pub maximum_resident_bytes: u64,
    pub maximum_scalar_operations: u64,
}
#[derive(Debug, Error)]
pub enum AffineSupportError {
    #[error("affine symbolic support exceeds planning admission")]
    Budget,
    #[error("affine symbolic inventory arithmetic overflow")]
    Overflow,
    #[error("affine symbolic planning cancelled")]
    Cancelled,
    #[error("affine symbolic allocation unavailable")]
    Allocation,
    #[error(transparent)]
    Constraint(#[from] ConstraintAlgebraError),
    #[error(transparent)]
    Sparse(#[from] SparsePatternError),
}
pub struct AffineSparseSupportPlan<'a> {
    reduction: &'a AffineConstraintReduction,
    left: &'a SparsePattern,
    right: &'a SparsePattern,
    transpose: AffineTransposeIndex,
    marks: Vec<usize>,
    touched: Vec<usize>,
    offsets: Vec<u64>,
    statistics: SparsePatternStatistics,
    planning_bytes: u64,
    scalar_operations: u64,
}
impl<'a> AffineSparseSupportPlan<'a> {
    pub fn prepare(
        reduction: &'a AffineConstraintReduction,
        left: &'a SparsePattern,
        right: &'a SparsePattern,
        budget: AffineSupportBudget,
    ) -> Result<Self, AffineSupportError> {
        cancel()?;
        if budget.maximum_scalar_operations == 0 {
            return Err(AffineSupportError::Budget);
        }
        let n = reduction.reduced_dof_count;
        let offsets = n.checked_add(1).ok_or(AffineSupportError::Overflow)?;
        // Validation partition, transpose offsets/cursors/entries, marker/touched
        // scratch and retained output offsets. Caller-owned sources are additional.
        let words = reduction
            .full_dof_count
            .checked_add(offsets.checked_mul(3).ok_or(AffineSupportError::Overflow)?)
            .and_then(|v| v.checked_add(reduction.recovery_column_indices.len().checked_mul(2)?))
            .and_then(|v| v.checked_add(n.checked_mul(2)?))
            .ok_or(AffineSupportError::Overflow)?;
        let planning_bytes = u64::try_from(words)
            .ok()
            .and_then(|v| v.checked_mul(8))
            .ok_or(AffineSupportError::Overflow)?;
        if planning_bytes > budget.maximum_resident_bytes {
            return Err(AffineSupportError::Budget);
        }
        reduction.validate()?;
        if left.rows() != reduction.full_dof_count || left.columns() != reduction.full_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch.into());
        }
        let union = left.union_statistics(right)?;
        let mut expansions = 0u64;
        for row in 0..reduction.full_dof_count {
            cancel()?;
            let l = reduction.recovery_row_offsets[row + 1] - reduction.recovery_row_offsets[row];
            for column in left.union_columns(right, row) {
                let column = column as usize;
                let r = reduction.recovery_row_offsets[column + 1]
                    - reduction.recovery_row_offsets[column];
                expansions = expansions
                    .checked_add(
                        u64::try_from(l)
                            .ok()
                            .and_then(|l| l.checked_mul(r as u64))
                            .ok_or(AffineSupportError::Overflow)?,
                    )
                    .ok_or(AffineSupportError::Overflow)?;
            }
        }
        // Two visits plus bounded in-place ordering of touched columns. This is
        // conservative index/comparison work, not measured instructions or time.
        let log = usize::BITS - n.max(1).leading_zeros();
        let partition_log = usize::BITS - reduction.full_dof_count.max(1).leading_zeros();
        let scalar_operations = expansions
            .checked_mul(16 + 8 * u64::from(log))
            .and_then(|v| v.checked_add((union.nonzeros as u64).checked_mul(8)?))
            .and_then(|v| {
                v.checked_add((words as u64).checked_mul(16 + 8 * u64::from(partition_log))?)
            })
            .ok_or(AffineSupportError::Overflow)?;
        if scalar_operations > budget.maximum_scalar_operations {
            return Err(AffineSupportError::Budget);
        }
        let transpose = AffineTransposeIndex::try_new(reduction)?;
        let mut marks = reserve(n)?;
        marks.resize(n, 0);
        let touched = reserve(n)?;
        let mut result = Self {
            reduction,
            left,
            right,
            transpose,
            marks,
            touched,
            offsets: reserve(offsets)?,
            statistics: SparsePatternStatistics {
                rows: n,
                columns: n,
                nonzeros: 0,
                maximum_row_nonzeros: 0,
            },
            planning_bytes,
            scalar_operations,
        };
        result.offsets.push(0);
        for row in 0..n {
            result.collect(row, 1)?;
            result.statistics.maximum_row_nonzeros = result
                .statistics
                .maximum_row_nonzeros
                .max(result.touched.len());
            result.statistics.nonzeros = result
                .statistics
                .nonzeros
                .checked_add(result.touched.len())
                .ok_or(AffineSupportError::Overflow)?;
            result.offsets.push(result.statistics.nonzeros as u64);
        }
        Ok(result)
    }
    pub fn statistics(&self) -> SparsePatternStatistics {
        self.statistics
    }
    pub fn planning_resident_bytes(&self) -> u64 {
        self.planning_bytes
    }
    pub fn scalar_operations_upper_bound(&self) -> u64 {
        self.scalar_operations
    }
    /// Caller admits statistics and solver demand before invoking this method.
    pub fn materialize(mut self) -> Result<SparsePattern, AffineSupportError> {
        let mut columns = reserve(self.statistics.nonzeros)?;
        for row in 0..self.statistics.rows {
            self.collect(row, 2)?;
            self.touched.sort_unstable();
            columns.extend(self.touched.iter().map(|v| *v as u64));
        }
        cancel()?;
        Ok(SparsePattern::try_new(
            self.statistics.rows as u64,
            self.statistics.columns as u64,
            self.offsets,
            columns,
        )?)
    }
    fn collect(&mut self, row: usize, pass: usize) -> Result<(), AffineSupportError> {
        cancel()?;
        let epoch = row
            .checked_mul(2)
            .and_then(|r| r.checked_add(pass))
            .ok_or(AffineSupportError::Overflow)?;
        self.touched.clear();
        for position in self.transpose.column_offsets[row]..self.transpose.column_offsets[row + 1] {
            let (full, _) = self.transpose.row_entries[position];
            for column in self.left.union_columns(self.right, full) {
                cancel()?;
                let column = column as usize;
                for entry in self.reduction.recovery_row_offsets[column]
                    ..self.reduction.recovery_row_offsets[column + 1]
                {
                    let reduced = self.reduction.recovery_column_indices[entry];
                    if self.marks[reduced] != epoch {
                        self.marks[reduced] = epoch;
                        self.touched.push(reduced);
                    }
                }
            }
        }
        Ok(())
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>, AffineSupportError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| AffineSupportError::Allocation)?;
    Ok(values)
}
fn cancel() -> Result<(), AffineSupportError> {
    if crate::is_fea_cancelled() {
        Err(AffineSupportError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;

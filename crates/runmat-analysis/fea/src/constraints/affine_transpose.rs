//! Original ordered transpose index shared by symbolic planning and scoped application.
use super::{AffineConstraintReduction, ConstraintAlgebraError};
use crate::{algebra::ScopedSpmvError, planning::resource::LocalExecutionPlan};
/// Column ownership makes Tᵀ application race-free without a worker-sized reduced vector.
/// Entries in each column retain the original row/entry order and the serial sum order.
pub(crate) struct AffineTransposeIndex {
    pub(super) column_offsets: Vec<usize>,
    pub(super) row_entries: Vec<(usize, usize)>,
}

impl AffineTransposeIndex {
    pub(crate) fn try_new(
        reduction: &AffineConstraintReduction,
    ) -> Result<Self, ConstraintAlgebraError> {
        reduction.validate()?;
        let count = reduction.reduced_dof_count.checked_add(1).ok_or_else(|| {
            ConstraintAlgebraError::InvalidMatrix("transpose dimension overflow".into())
        })?;
        let mut column_offsets = zeros(count, 0usize)?;
        for &column in &reduction.recovery_column_indices {
            column_offsets[column + 1] += 1;
        }
        for column in 0..reduction.reduced_dof_count {
            column_offsets[column + 1] += column_offsets[column];
        }
        let mut cursor = zeros(count, 0usize)?;
        cursor.copy_from_slice(&column_offsets);
        let mut row_entries = zeros(reduction.recovery_column_indices.len(), (0usize, 0usize))?;
        for row in 0..reduction.full_dof_count {
            for entry in
                reduction.recovery_row_offsets[row]..reduction.recovery_row_offsets[row + 1]
            {
                let column = reduction.recovery_column_indices[entry];
                row_entries[cursor[column]] = (row, entry);
                cursor[column] += 1;
            }
        }
        Ok(Self {
            column_offsets,
            row_entries,
        })
    }

    pub(crate) fn apply(
        &self,
        reduction: &AffineConstraintReduction,
        full: &[f64],
        plan: LocalExecutionPlan,
    ) -> Result<Vec<f64>, ScopedSpmvError> {
        let mut reduced = vec![0.0; reduction.reduced_dof_count];
        let cancellation = crate::progress::cancellation_token();
        crate::planning::resource::execute_scoped_slices_mut(
            plan,
            &mut reduced,
            |chunk, output| {
                for (offset, value) in output.iter_mut().enumerate() {
                    if cancellation.is_cancelled() {
                        return Err(crate::algebra::SpmvWorkError::Cancelled);
                    }
                    let column = chunk.start() as usize + offset;
                    for position in self.column_offsets[column]..self.column_offsets[column + 1] {
                        let (row, entry) = self.row_entries[position];
                        *value += reduction.recovery_coefficients[entry] * full[row];
                    }
                }
                Ok(())
            },
        )?;
        Ok(reduced)
    }
}

fn zeros<T: Clone>(count: usize, value: T) -> Result<Vec<T>, ConstraintAlgebraError> {
    let mut data = Vec::new();
    data.try_reserve_exact(count).map_err(|_| {
        ConstraintAlgebraError::InvalidMatrix("affine transpose allocation unavailable".into())
    })?;
    data.resize(count, value);
    Ok(data)
}

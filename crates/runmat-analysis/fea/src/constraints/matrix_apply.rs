//! Sparse application of canonical constraint matrices and their transposes.

use super::{ConstraintAlgebraError, ResolvedConstraintMatrix};
use crate::{
    algebra::{ScopedSpmvError, SpmvWorkError},
    planning::resource::{execute_scoped_chunks, execute_scoped_slices_mut, LocalExecutionPlan},
};

// At most 256 coefficient/value pairs and one product per worker, far below
// the admitted 8 MiB fixed solve scratch per local thread. The transpose
// retains only its final full-DOF output, never a full vector per worker.
const TRANSPOSE_ENTRIES_PER_WORKER_WAVE: usize = 256;

impl ResolvedConstraintMatrix {
    pub fn apply(&self, full: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if full.len() != self.full_dof_count {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        Ok(self.apply_validated(full))
    }

    pub(crate) fn apply_validated(&self, full: &[f64]) -> Vec<f64> {
        (0..self.row_count())
            .map(|row| {
                let start = self.row_offsets[row];
                let end = self.row_offsets[row + 1];
                (start..end)
                    .map(|entry| self.coefficients[entry] * full[self.column_indices[entry]])
                    .sum()
            })
            .collect()
    }

    pub(crate) fn apply_validated_scoped(
        &self,
        full: &[f64],
        plan: LocalExecutionPlan,
    ) -> Result<Vec<f64>, ScopedSpmvError> {
        if full.len() != self.full_dof_count {
            return Err(ScopedSpmvError::DimensionMismatch);
        }
        let mut result = vec![0.0; self.row_count()];
        let cancellation = crate::progress::cancellation_token();
        execute_scoped_slices_mut(plan, &mut result, |chunk, output| {
            for (offset, value) in output.iter_mut().enumerate() {
                if cancellation.is_cancelled() {
                    return Err(SpmvWorkError::Cancelled);
                }
                let row = chunk.start() as usize + offset;
                let start = self.row_offsets[row];
                let end = self.row_offsets[row + 1];
                *value = (start..end)
                    .map(|entry| self.coefficients[entry] * full[self.column_indices[entry]])
                    .sum();
            }
            Ok(())
        })?;
        Ok(result)
    }

    pub fn transpose_apply(&self, rows: &[f64]) -> Result<Vec<f64>, ConstraintAlgebraError> {
        self.validate()?;
        if rows.len() != self.row_count() {
            return Err(ConstraintAlgebraError::OperatorDimensionMismatch);
        }
        Ok(self.transpose_apply_validated(rows))
    }

    pub(crate) fn transpose_apply_validated(&self, rows: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.full_dof_count];
        for (row, value) in rows.iter().copied().enumerate() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for entry in start..end {
                full[self.column_indices[entry]] += self.coefficients[entry] * value;
            }
        }
        full
    }

    pub(crate) fn transpose_apply_validated_scoped(
        &self,
        rows: &[f64],
        plan: LocalExecutionPlan,
    ) -> Result<Vec<f64>, ScopedSpmvError> {
        if rows.len() != self.row_count() {
            return Err(ScopedSpmvError::DimensionMismatch);
        }
        let mut full = vec![0.0; self.full_dof_count];
        let wave_capacity =
            usize::from(plan.thread_count().get()) * TRANSPOSE_ENTRIES_PER_WORKER_WAVE;
        let mut wave = Vec::with_capacity(wave_capacity.min(self.coefficients.len()));
        let mut wave_start = 0;
        for (row, value) in rows.iter().copied().enumerate() {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                wave.push((self.coefficients[entry], value));
                if wave.len() == wave_capacity {
                    self.scatter_transpose_wave(&wave, wave_start, &mut full, plan)?;
                    wave_start += wave.len();
                    wave.clear();
                }
            }
        }
        if !wave.is_empty() {
            self.scatter_transpose_wave(&wave, wave_start, &mut full, plan)?;
        }
        Ok(full)
    }

    fn scatter_transpose_wave(
        &self,
        wave: &[(f64, f64)],
        start_entry: usize,
        full: &mut [f64],
        plan: LocalExecutionPlan,
    ) -> Result<(), ScopedSpmvError> {
        let cancellation = crate::progress::cancellation_token();
        let chunks = execute_scoped_chunks(plan, wave.len() as u64, |chunk| {
            let mut values = Vec::with_capacity(chunk.len() as usize);
            for (coefficient, row_value) in
                &wave[chunk.start() as usize..chunk.end_exclusive() as usize]
            {
                if cancellation.is_cancelled() {
                    return Err(SpmvWorkError::Cancelled);
                }
                values.push(coefficient * row_value);
            }
            Ok(values)
        })?;
        let mut entry = start_entry;
        for values in chunks {
            for value in values {
                full[self.column_indices[entry]] += value;
                entry += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        planning::resource::{LocalThreadCount, ScopedLocalExecutionError},
        progress::{replace_fea_progress_context, FeaCancellationToken, FeaProgressContext},
    };

    #[test]
    fn scoped_constraint_projection_preserves_row_and_scatter_order_across_waves() {
        let row_count = 829;
        let matrix = ResolvedConstraintMatrix {
            full_dof_count: 9,
            row_offsets: (0..=row_count).collect(),
            column_indices: (0..row_count).map(|row| (row * 7) % 9).collect(),
            coefficients: (0..row_count)
                .map(|row| 0.1 + row as f64 * 0.00001)
                .collect(),
            right_hand_side: vec![0.0; row_count],
            row_identities: (0..row_count)
                .map(|row| runmat_execution::Digest::sha256(row.to_le_bytes()))
                .collect(),
            constraint_ids: (0..row_count).map(|row| format!("row_{row}")).collect(),
        };
        let full = [0.3, -0.2, 5.0, 0.1, -3.0, 1.0, 0.7, -0.8, 4.0];
        let rows = (0..row_count)
            .map(|row| 1.0 + row as f64 * 0.001)
            .collect::<Vec<_>>();
        let expected_forward = matrix.apply_validated(&full);
        let expected_transpose = matrix.transpose_apply_validated(&rows);
        for thread_count in [2, 3, 7] {
            let plan =
                LocalExecutionPlan::fixed(LocalThreadCount::new(thread_count).unwrap()).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    matrix.apply_validated_scoped(&full, plan).unwrap(),
                    expected_forward
                );
                assert_eq!(
                    matrix
                        .transpose_apply_validated_scoped(&rows, plan)
                        .unwrap(),
                    expected_transpose
                );
            }
        }
        assert!(
            std::mem::size_of::<(f64, f64)>() * 256
                + std::mem::size_of::<f64>() * 256
                + std::mem::size_of::<Vec<f64>>() * 2
                < 8 * 1024 * 1024
        );
    }

    #[test]
    fn scoped_constraint_projection_rejects_bad_dimensions_nested_work_and_cancellation() {
        let matrix = ResolvedConstraintMatrix {
            full_dof_count: 2,
            row_offsets: vec![0, 2],
            column_indices: vec![0, 1],
            coefficients: vec![1.0, -1.0],
            right_hand_side: vec![0.0],
            row_identities: vec![runmat_execution::Digest::sha256(b"constraint")],
            constraint_ids: vec!["tie".into()],
        };
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
        assert!(matches!(
            matrix.apply_validated_scoped(&[1.0], plan),
            Err(ScopedSpmvError::DimensionMismatch)
        ));
        assert!(matches!(
            matrix.transpose_apply_validated_scoped(&[], plan),
            Err(ScopedSpmvError::DimensionMismatch)
        ));
        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            Ok::<_, ()>(matrix.transpose_apply_validated_scoped(&[2.0], plan))
        })
        .unwrap();
        assert!(matches!(
            &nested[0],
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::NestedExecution
            ))
        ));
        let cancellation = FeaCancellationToken::new();
        cancellation.cancel();
        let _context = replace_fea_progress_context(FeaProgressContext {
            cancellation,
            ..FeaProgressContext::default()
        });
        assert!(matches!(
            matrix.transpose_apply_validated_scoped(&[2.0], plan),
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::Work {
                    source: SpmvWorkError::Cancelled,
                    ..
                }
            ))
        ));
        assert!(matches!(
            matrix.apply_validated_scoped(&[1.0, 2.0], plan),
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::Work {
                    source: SpmvWorkError::Cancelled,
                    ..
                }
            ))
        ));
    }
}

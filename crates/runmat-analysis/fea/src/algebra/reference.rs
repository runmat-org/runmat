use super::RealSparseMatrix;
use crate::planning::resource::{
    execute_scoped_chunks, execute_scoped_slices_mut, LocalExecutionPlan, ScopedLocalExecutionError,
};

#[derive(Debug, thiserror::Error)]
pub enum ScopedSpmvError {
    #[error("sparse numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
    #[error("sparse matrix/vector dimension mismatch")]
    DimensionMismatch,
    #[error("vector reduction dimension mismatch")]
    ReductionDimensionMismatch,
    #[error("vector update dimension mismatch")]
    VectorDimensionMismatch,
    #[error("{0}")]
    Execution(#[from] ScopedLocalExecutionError<SpmvWorkError>),
}

#[derive(Debug, thiserror::Error)]
pub enum SpmvWorkError {
    #[error("sparse operator application cancelled")]
    Cancelled,
}

const DOT_REDUCTION_BLOCK_SIZE: usize = 256;

pub(crate) fn dot_canonical(left: &[f64], right: &[f64]) -> f64 {
    let mut total = 0.0;
    for block in 0..left
        .len()
        .min(right.len())
        .div_ceil(DOT_REDUCTION_BLOCK_SIZE)
    {
        let start = block * DOT_REDUCTION_BLOCK_SIZE;
        let end = (start + DOT_REDUCTION_BLOCK_SIZE)
            .min(left.len())
            .min(right.len());
        let mut partial = 0.0;
        for index in start..end {
            partial += left[index] * right[index];
        }
        total += partial;
    }
    total
}

/// Dot product under the admitted local CPU plan. Fixed input blocks establish one reduction
/// order independent of worker count; workers only own disjoint ranges of those blocks.
pub fn dot_scoped(
    left: &[f64],
    right: &[f64],
    plan: LocalExecutionPlan,
) -> Result<f64, ScopedSpmvError> {
    if left.len() != right.len() {
        return Err(ScopedSpmvError::ReductionDimensionMismatch);
    }
    if left.is_empty() {
        return Ok(0.0);
    }
    let cancellation = crate::progress::cancellation_token();
    let block_count = left.len().div_ceil(DOT_REDUCTION_BLOCK_SIZE);
    let chunks = execute_scoped_chunks(plan, block_count as u64, |chunk| {
        let mut partials = Vec::with_capacity((chunk.end_exclusive() - chunk.start()) as usize);
        for block in chunk.start() as usize..chunk.end_exclusive() as usize {
            let start = block * DOT_REDUCTION_BLOCK_SIZE;
            let end = (start + DOT_REDUCTION_BLOCK_SIZE).min(left.len());
            let mut sum = 0.0;
            for index in start..end {
                if cancellation.is_cancelled() {
                    return Err(SpmvWorkError::Cancelled);
                }
                sum += left[index] * right[index];
            }
            partials.push(sum);
        }
        Ok(partials)
    })?;
    Ok(chunks.into_iter().flatten().sum())
}

/// In-place `y += alpha*x` under the admitted plan. Disjoint output slices keep the
/// allocation envelope unchanged, and every element uses the reference update order.
pub fn axpy_scoped(
    alpha: f64,
    x: &[f64],
    y: &mut [f64],
    plan: LocalExecutionPlan,
) -> Result<(), ScopedSpmvError> {
    if x.len() != y.len() {
        return Err(ScopedSpmvError::VectorDimensionMismatch);
    }
    let cancellation = crate::progress::cancellation_token();
    execute_scoped_slices_mut(plan, y, |chunk, output| {
        for (offset, value) in output.iter_mut().enumerate() {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            *value += alpha * x[chunk.start() as usize + offset];
        }
        Ok(())
    })?;
    Ok(())
}

/// In-place scaling of disjoint vector slices under the admitted local CPU plan.
/// The reference multiplication order is unchanged and no extra full vector is retained.
pub fn scale_scoped(
    values: &mut [f64],
    factor: f64,
    plan: LocalExecutionPlan,
) -> Result<(), ScopedSpmvError> {
    let cancellation = crate::progress::cancellation_token();
    execute_scoped_slices_mut(plan, values, |_chunk, output| {
        for value in output {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            *value *= factor;
        }
        Ok(())
    })?;
    Ok(())
}

/// In-place PCG direction recurrence `y = x + beta*y`, preserving the reference
/// operation order independently for every vector entry.
pub fn direction_update_scoped(
    x: &[f64],
    y: &mut [f64],
    beta: f64,
    plan: LocalExecutionPlan,
) -> Result<(), ScopedSpmvError> {
    if x.len() != y.len() {
        return Err(ScopedSpmvError::VectorDimensionMismatch);
    }
    let cancellation = crate::progress::cancellation_token();
    execute_scoped_slices_mut(plan, y, |chunk, output| {
        for (offset, value) in output.iter_mut().enumerate() {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            *value = x[chunk.start() as usize + offset] + beta * *value;
        }
        Ok(())
    })?;
    Ok(())
}

/// Builds `left - right` in one result buffer under the admitted plan.
pub fn vector_sub_scoped(
    left: &[f64],
    right: &[f64],
    plan: LocalExecutionPlan,
) -> Result<Vec<f64>, ScopedSpmvError> {
    if left.len() != right.len() {
        return Err(ScopedSpmvError::VectorDimensionMismatch);
    }
    let mut result = vec![0.0; left.len()];
    let cancellation = crate::progress::cancellation_token();
    execute_scoped_slices_mut(plan, &mut result, |chunk, output| {
        for (offset, value) in output.iter_mut().enumerate() {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            let index = chunk.start() as usize + offset;
            *value = left[index] - right[index];
        }
        Ok(())
    })?;
    Ok(result)
}

/// Replaces an existing result buffer with `left - right` under the admitted plan.
/// Affine CG uses this for its independently checked final equation residual,
/// without allocating another full reduced vector.
pub fn subtract_into_scoped(
    left: &[f64],
    right: &[f64],
    result: &mut [f64],
    plan: LocalExecutionPlan,
) -> Result<(), ScopedSpmvError> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(ScopedSpmvError::VectorDimensionMismatch);
    }
    let cancellation = crate::progress::cancellation_token();
    execute_scoped_slices_mut(plan, result, |chunk, output| {
        for (offset, value) in output.iter_mut().enumerate() {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            let index = chunk.start() as usize + offset;
            *value = left[index] - right[index];
        }
        Ok(())
    })?;
    Ok(())
}

/// Builds a normalized Krylov basis vector without retaining a second output copy.
pub fn normalized_copy_scoped(
    input: &[f64],
    divisor: f64,
    plan: LocalExecutionPlan,
) -> Result<Vec<f64>, ScopedSpmvError> {
    let mut result = vec![0.0; input.len()];
    let cancellation = crate::progress::cancellation_token();
    execute_scoped_slices_mut(plan, &mut result, |chunk, output| {
        for (offset, value) in output.iter_mut().enumerate() {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            *value = input[chunk.start() as usize + offset] / divisor;
        }
        Ok(())
    })?;
    Ok(result)
}

pub fn spmv(matrix: &RealSparseMatrix, x: &[f64]) -> Result<Vec<f64>, &'static str> {
    if x.len() != matrix.columns() {
        return Err("sparse matrix/vector dimension mismatch");
    }
    let mut result = vec![0.0; matrix.rows()];
    if matrix.values().len() >= 32_768
        && matrix.rows() >= 1_024
        && runmat_cpu_execution::current_threads() > 1
    {
        // The installed pool is reservation authority supplied by the native caller.
        // Row accumulation order and the single output allocation remain unchanged.
        runmat_cpu_execution::for_each_mut_indexed(&mut result, |row, value| {
            *value = spmv_row(matrix, x, row)
        });
    } else {
        for (row, value) in result.iter_mut().enumerate() {
            *value = spmv_row(matrix, x, row);
        }
    }
    Ok(result)
}

/// Applies independent CSR rows under the admitted local CPU plan.
///
/// Each row retains the reference implementation's entry order, so the result is bitwise
/// identical for every supported partition count. Chunk vectors are merged in row order;
/// callers must account for one additional full-size temporary output at this boundary.
pub fn spmv_scoped(
    matrix: &RealSparseMatrix,
    x: &[f64],
    plan: LocalExecutionPlan,
) -> Result<Vec<f64>, ScopedSpmvError> {
    if x.len() != matrix.columns() {
        return Err(ScopedSpmvError::DimensionMismatch);
    }
    if matrix.rows() == 0 {
        return Ok(Vec::new());
    }
    let cancellation = crate::progress::cancellation_token();
    let chunks = execute_scoped_chunks(plan, matrix.rows() as u64, |chunk| {
        let mut values = Vec::with_capacity((chunk.end_exclusive() - chunk.start()) as usize);
        for row in chunk.start() as usize..chunk.end_exclusive() as usize {
            if cancellation.is_cancelled() {
                return Err(SpmvWorkError::Cancelled);
            }
            values.push(spmv_row(matrix, x, row));
        }
        Ok(values)
    })?;
    let mut result = Vec::with_capacity(matrix.rows());
    for mut values in chunks {
        result.append(&mut values);
    }
    Ok(result)
}

fn spmv_row(matrix: &RealSparseMatrix, x: &[f64], row: usize) -> f64 {
    matrix
        .pattern()
        .row_range(row)
        .map(|entry| matrix.values()[entry] * x[matrix.pattern().column(entry)])
        .sum()
}

pub fn residual(
    matrix: &RealSparseMatrix,
    x: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if rhs.len() != matrix.rows() {
        return Err("sparse residual dimension mismatch");
    }
    Ok(rhs
        .iter()
        .zip(spmv(matrix, x)?)
        .map(|(right, applied)| right - applied)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler, RealSparseMatrix};
    use crate::planning::resource::LocalThreadCount;
    use crate::progress::{replace_fea_progress_context, FeaCancellationToken, FeaProgressContext};

    #[test]
    fn scoped_dot_uses_canonical_chunks_and_rejects_invalid_inputs() {
        let left = [3.0, -4.0, 0.2, 7.0, 1.0e-12, -5.0, 9.0];
        let right = [-1.0, 2.0, 3.0, -0.5, 4.0, 6.0, 1.0];
        let expected = dot_canonical(&left, &right);
        assert_eq!(
            dot_scoped(&left, &right, LocalExecutionPlan::serial()).unwrap(),
            expected
        );
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        assert!((dot_scoped(&left, &right, plan).unwrap() - expected).abs() <= 1.0e-12);
        assert!(matches!(
            dot_scoped(&left, &right[..6], plan),
            Err(ScopedSpmvError::ReductionDimensionMismatch)
        ));
        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            Ok::<_, ()>(dot_scoped(&left, &right, plan))
        })
        .unwrap();
        assert!(matches!(
            &nested[0],
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::NestedExecution
            ))
        ));
    }

    #[test]
    fn scoped_dot_preserves_bits_across_worker_partitions_and_block_boundaries() {
        let left = (0..1_031)
            .map(|index| match index % 4 {
                0 => 1.0e12,
                1 => -1.0e12,
                2 => 1.0e-4,
                _ => (index as f64) * 1.0e-7,
            })
            .collect::<Vec<_>>();
        let right = (0..left.len())
            .map(|index| if index % 3 == 0 { -0.25 } else { 1.0 })
            .collect::<Vec<_>>();
        let expected = dot_scoped(&left, &right, LocalExecutionPlan::serial()).unwrap();
        for workers in [2, 3, 4, 7] {
            let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(workers).unwrap()).unwrap();
            assert_eq!(dot_scoped(&left, &right, plan).unwrap(), expected);
        }
    }

    #[test]
    fn scoped_dot_propagates_cancellation() {
        let cancellation = FeaCancellationToken::new();
        cancellation.cancel();
        let _context = replace_fea_progress_context(FeaProgressContext {
            cancellation,
            ..FeaProgressContext::default()
        });
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        assert!(matches!(
            dot_scoped(&[1.0; 8], &[2.0; 8], plan),
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::Work {
                    source: SpmvWorkError::Cancelled,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn scoped_vector_updates_preserve_reference_bits_and_reject_bad_dimensions() {
        let x = [3.0, -4.0, 0.2, 7.0, 1.0e-12, -5.0, 9.0];
        let base = [0.5, 8.0, -3.0, 2.0, -2.0, 1.0, -7.0];
        let mut expected = base;
        for (value, input) in expected.iter_mut().zip(x) {
            *value += -0.25 * input;
            *value = input + 0.75 * *value;
        }
        for threads in [1, 2, 3, 7, 8] {
            let plan =
                LocalExecutionPlan::from_thread_count(LocalThreadCount::new(threads).unwrap())
                    .unwrap();
            let mut actual = base;
            axpy_scoped(-0.25, &x, &mut actual, plan).unwrap();
            direction_update_scoped(&x, &mut actual, 0.75, plan).unwrap();
            assert_eq!(actual, expected);
            scale_scoped(&mut actual, -1.25, plan).unwrap();
            let mut scaled_expected = expected;
            for value in &mut scaled_expected {
                *value *= -1.25;
            }
            assert_eq!(actual, scaled_expected);
            assert_eq!(
                vector_sub_scoped(&actual, &x, plan).unwrap(),
                actual
                    .iter()
                    .zip(x)
                    .map(|(left, right)| left - right)
                    .collect::<Vec<_>>()
            );
            let mut reused = [f64::NAN; 7];
            subtract_into_scoped(&actual, &x, &mut reused, plan).unwrap();
            assert_eq!(
                reused.to_vec(),
                vector_sub_scoped(&actual, &x, plan).unwrap()
            );
            assert_eq!(
                normalized_copy_scoped(&actual, 2.5, plan).unwrap(),
                actual.iter().map(|value| value / 2.5).collect::<Vec<_>>()
            );
        }
        assert!(matches!(
            axpy_scoped(1.0, &x[..6], &mut expected, LocalExecutionPlan::serial()),
            Err(ScopedSpmvError::VectorDimensionMismatch)
        ));
        assert!(matches!(
            vector_sub_scoped(&x[..6], &expected, LocalExecutionPlan::serial()),
            Err(ScopedSpmvError::VectorDimensionMismatch)
        ));
        assert!(matches!(
            subtract_into_scoped(&x, &expected, &mut [0.0; 6], LocalExecutionPlan::serial()),
            Err(ScopedSpmvError::VectorDimensionMismatch)
        ));
    }

    #[test]
    fn scoped_vector_updates_propagate_cancellation_without_a_result() {
        let cancellation = FeaCancellationToken::new();
        cancellation.cancel();
        let _context = replace_fea_progress_context(FeaProgressContext {
            cancellation,
            ..FeaProgressContext::default()
        });
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        let mut values = [1.0; 8];
        for result in [
            axpy_scoped(2.0, &[1.0; 8], &mut values, plan).map(|_| Vec::new()),
            scale_scoped(&mut values, 2.0, plan).map(|_| Vec::new()),
            vector_sub_scoped(&[2.0; 8], &[1.0; 8], plan),
            subtract_into_scoped(&[2.0; 8], &[1.0; 8], &mut values, plan).map(|_| Vec::new()),
            normalized_copy_scoped(&[2.0; 8], 2.0, plan),
        ] {
            assert!(matches!(
                result,
                Err(ScopedSpmvError::Execution(
                    ScopedLocalExecutionError::Work {
                        source: SpmvWorkError::Cancelled,
                        ..
                    }
                ))
            ));
        }
    }

    #[test]
    fn identity_application_and_residual_are_deterministic() {
        let identity = RealSparseMatrix::diagonal(
            vec![1.0, 1.0, 1.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let x = vec![3.0, -2.0, 7.0];
        assert_eq!(spmv(&identity, &x).unwrap(), x);
        assert_eq!(residual(&identity, &x, &x).unwrap(), vec![0.0; 3]);
    }

    #[test]
    fn scoped_application_preserves_reference_row_bits_across_partition_counts() {
        let mut assembler =
            RealSparseAssembler::square(7, OperatorRole::Stiffness, MatrixProperties::GENERAL);
        for row in 0..7 {
            assembler.add(row, row, 3.5 + row as f64);
            assembler.add(row, (row + 1) % 7, -0.25);
            assembler.add(row, (row + 3) % 7, 1.0e-12);
        }
        let matrix = assembler.finish().unwrap();
        let x = vec![3.0, -4.0, 0.2, 7.0, 1.0e-12, -5.0, 9.0];
        let expected = spmv(&matrix, &x).unwrap();
        for threads in [1, 2, 3, 7, 8] {
            let plan =
                LocalExecutionPlan::from_thread_count(LocalThreadCount::new(threads).unwrap())
                    .unwrap();
            assert_eq!(spmv_scoped(&matrix, &x, plan).unwrap(), expected);
        }
        assert!(matches!(
            spmv_scoped(&matrix, &x[..6], LocalExecutionPlan::serial()),
            Err(ScopedSpmvError::DimensionMismatch)
        ));
    }

    #[test]
    fn scoped_application_propagates_cancellation_from_worker_chunks() {
        let matrix = RealSparseMatrix::diagonal(
            vec![1.0; 8],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let cancellation = FeaCancellationToken::new();
        cancellation.cancel();
        let _context = replace_fea_progress_context(FeaProgressContext {
            cancellation,
            ..FeaProgressContext::default()
        });
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
        assert!(matches!(
            spmv_scoped(&matrix, &[1.0; 8], plan),
            Err(ScopedSpmvError::Execution(
                ScopedLocalExecutionError::Work {
                    source: SpmvWorkError::Cancelled,
                    ..
                }
            ))
        ));
    }
}

#[cfg(test)]
mod persistent_pool_tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
    // witness: FEA-SCALE-PARALLEL-001/shared_parallel_kernels @
    //   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
    //   boundary: Pooled CSR matrix-vector rows preserve serial bits at 2 and 4 threads on a 2,048-row
    //   matrix.
    // witness: FEA-SHARED-EXECUTION-001/serial_reference_and_equivalence_policy @
    //   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
    //   boundary: Pooled CSR products equal the serial reference bitwise.
    #[test]
    fn installed_pool_csr_rows_preserve_serial_bits_and_output_shape() {
        let mut assembler =
            RealSparseAssembler::square(2048, OperatorRole::Stiffness, MatrixProperties::GENERAL);
        for row in 0..2048 {
            for offset in 0..32 {
                assembler.add(
                    row,
                    (row + offset) % 2048,
                    if offset % 2 == 0 { 1e12 } else { -1e12 + 0.25 },
                );
            }
        }
        let matrix = assembler.finish().unwrap();
        let x = (0..2048)
            .map(|index| 1. + index as f64 * 1e-9)
            .collect::<Vec<_>>();
        let serial = spmv(&matrix, &x).unwrap();
        for threads in [2, 4] {
            let parallel = runmat_cpu_execution::with_cpu_pool(threads, || spmv(&matrix, &x))
                .unwrap()
                .unwrap();
            assert_eq!(parallel, serial);
        }
    }
}

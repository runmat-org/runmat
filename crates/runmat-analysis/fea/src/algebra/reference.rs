use super::RealSparseMatrix;
use crate::planning::resource::{
    execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError,
};

#[derive(Debug, thiserror::Error)]
pub enum ScopedSpmvError {
    #[error("sparse matrix/vector dimension mismatch")]
    DimensionMismatch,
    #[error("{0}")]
    Execution(#[from] ScopedLocalExecutionError<SpmvWorkError>),
}

#[derive(Debug, thiserror::Error)]
pub enum SpmvWorkError {
    #[error("sparse operator application cancelled")]
    Cancelled,
}

pub fn spmv(matrix: &RealSparseMatrix, x: &[f64]) -> Result<Vec<f64>, &'static str> {
    if x.len() != matrix.columns() {
        return Err("sparse matrix/vector dimension mismatch");
    }
    let mut result = vec![0.0; matrix.rows()];
    for (row, result_value) in result.iter_mut().enumerate() {
        *result_value = spmv_row(matrix, x, row);
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

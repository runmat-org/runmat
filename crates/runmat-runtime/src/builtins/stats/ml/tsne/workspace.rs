//! Checked, cooperatively filled dense workspaces for t-SNE geometry.

use super::{cooperation::TsneCooperation, internal_resource, internal_with_source};
use crate::BuiltinResult;
use nalgebra::DMatrix;

pub(super) async fn zero_matrix(
    rows: usize,
    cols: usize,
    cooperation: &mut TsneCooperation,
) -> BuiltinResult<DMatrix<f64>> {
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| internal_resource("tsne: matrix workspace shape overflow"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|source| {
        internal_with_source("tsne: matrix workspace exceeds available memory", source)
    })?;
    for _ in 0..count {
        values.push(0.0);
        cooperation.completed(1).await?;
    }
    Ok(DMatrix::from_vec(rows, cols, values))
}

pub(super) async fn zero_rows(
    rows: usize,
    cols: usize,
    cooperation: &mut TsneCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    rows.checked_mul(cols)
        .ok_or_else(|| internal_resource("tsne: projection shape overflow"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(rows).map_err(|source| {
        internal_with_source(
            "tsne: projected observations exceed available memory",
            source,
        )
    })?;
    for _ in 0..rows {
        let mut values = Vec::new();
        values.try_reserve_exact(cols).map_err(|source| {
            internal_with_source("tsne: projected features exceed available memory", source)
        })?;
        for _ in 0..cols {
            values.push(0.0);
            cooperation.completed(1).await?;
        }
        output.push(values);
        cooperation.completed(1).await?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn workspace_capacity_failure_preserves_operational_cause_without_retry() {
        let mut cooperation = TsneCooperation::new(None);
        let error = futures::executor::block_on(zero_matrix(usize::MAX, 1, &mut cooperation))
            .expect_err("capacity overflow must fail without attempting an allocation");
        assert_eq!(error.identifier(), Some("RunMat:tsne:Internal"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error
            .source()
            .expect("original reservation failure")
            .is::<std::collections::TryReserveError>());
    }

    #[test]
    fn checked_workspaces_keep_column_major_shape_without_infallible_bulk_fill() {
        let mut cooperation = TsneCooperation::new(None);
        let matrix = futures::executor::block_on(zero_matrix(2, 3, &mut cooperation)).unwrap();
        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix.as_slice(), &[0.0; 6]);

        let rows = futures::executor::block_on(zero_rows(3, 2, &mut cooperation)).unwrap();
        assert_eq!(rows, vec![vec![0.0; 2]; 3]);

        let error = futures::executor::block_on(zero_matrix(usize::MAX, 2, &mut cooperation))
            .expect_err("overflowing workspace must reject before allocation");
        assert!(error.message().contains("shape overflow"));
        assert_eq!(error.identifier(), Some("RunMat:tsne:Internal"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}

//! Checked, cooperatively filled dense workspaces for t-SNE geometry.

use super::{build_runtime_error, cooperation::TsneCooperation, ERROR_INTERNAL, NAME};
use crate::BuiltinResult;
use nalgebra::DMatrix;

fn allocation_error(
    message: &'static str,
    source: std::collections::TryReserveError,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_source(source);
    if let Some(identifier) = ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) async fn zero_matrix(
    rows: usize,
    cols: usize,
    cooperation: &mut TsneCooperation,
) -> BuiltinResult<DMatrix<f64>> {
    let count = rows.checked_mul(cols).ok_or_else(|| {
        build_runtime_error("tsne: matrix workspace shape overflow")
            .with_builtin(NAME)
            .build()
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|source| {
        allocation_error("tsne: matrix workspace exceeds available memory", source)
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
    rows.checked_mul(cols).ok_or_else(|| {
        build_runtime_error("tsne: projection shape overflow")
            .with_builtin(NAME)
            .build()
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(rows).map_err(|source| {
        allocation_error(
            "tsne: projected observations exceed available memory",
            source,
        )
    })?;
    for _ in 0..rows {
        let mut values = Vec::new();
        values.try_reserve_exact(cols).map_err(|source| {
            allocation_error("tsne: projected features exceed available memory", source)
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

    #[test]
    fn checked_workspaces_keep_column_major_shape_without_infallible_bulk_fill() {
        let mut cooperation = TsneCooperation::new(None);
        let matrix = futures::executor::block_on(zero_matrix(2, 3, &mut cooperation)).unwrap();
        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix.as_slice(), &[0.0; 6]);

        let rows = futures::executor::block_on(zero_rows(3, 2, &mut cooperation)).unwrap();
        assert_eq!(rows, vec![vec![0.0; 2]; 3]);

        let error = futures::executor::block_on(zero_matrix(usize::MAX, 2, &mut cooperation))
            .err()
            .expect("overflowing workspace must reject before allocation");
        assert!(error.message().contains("shape overflow"));
    }
}

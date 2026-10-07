use super::{invalid, Options, PreparedData, MAX_WORK_CELLS};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use nalgebra::{DMatrix, SymmetricEigen};
use runmat_value::Tensor;
use std::cmp::Ordering;

pub(super) fn tsne_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) async fn prepare_data_cooperative(
    x: &Tensor,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<PreparedData> {
    let (x_rows, x_cols) = tsne_matrix_dimensions(x);
    let mut rows = Vec::new();
    let mut good_rows = Vec::new();
    rows.try_reserve_exact(x_rows).map_err(|source| {
        super::internal_with_source("tsne: observations exceed available memory", source)
    })?;
    good_rows.try_reserve_exact(x_rows).map_err(|source| {
        super::internal_with_source("tsne: row indices exceed available memory", source)
    })?;
    for row in 0..x_rows {
        let mut out = Vec::new();
        out.try_reserve_exact(x_cols).map_err(|source| {
            super::internal_with_source("tsne: features exceed available memory", source)
        })?;
        let mut complete = true;
        for col in 0..x_cols {
            let value = tensor::tensor_value_f64(x, row + col * x_rows);
            cooperation.completed(1).await?;
            if value.is_infinite() {
                return Err(invalid("tsne: X must not contain Inf values"));
            }
            if value.is_nan() {
                complete = false;
                continue;
            }
            out.push(value);
        }
        if complete {
            good_rows.push(row);
            rows.push(out);
        }
        cooperation.completed(1).await?;
    }
    Ok(PreparedData {
        rows,
        good_rows,
        original_rows: x_rows,
    })
}

pub(super) fn needs_pca(options: &Options, feature_count: usize) -> bool {
    options.num_pca_components > 0 && options.num_pca_components < feature_count
}

pub(super) fn guard_dense_feature_work(feature_count: usize) -> BuiltinResult<()> {
    if feature_count.saturating_mul(feature_count) > MAX_WORK_CELLS {
        return Err(invalid(format!(
            "tsne: PCA and Mahalanobis distance support at most {MAX_WORK_CELLS} dense feature cells"
        )));
    }
    Ok(())
}

pub(super) async fn standardize_rows(
    rows: &mut [Vec<f64>],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> crate::BuiltinResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let cols = rows[0].len();
    for col in 0..cols {
        let mut total = 0.0;
        for row in rows.iter() {
            total += row[col];
            cooperation.completed(1).await?;
        }
        let mean = total / rows.len() as f64;
        let mut squared = 0.0;
        for row in rows.iter() {
            let diff = row[col] - mean;
            squared += diff * diff;
            cooperation.completed(1).await?;
        }
        let var = squared / (rows.len().saturating_sub(1).max(1) as f64);
        let scale = var.sqrt();
        for row in rows.iter_mut() {
            row[col] = if scale > 0.0 {
                (row[col] - mean) / scale
            } else {
                0.0
            };
            cooperation.completed(1).await?;
        }
    }
    Ok(())
}

pub(super) async fn apply_pca_if_requested(
    rows: Vec<Vec<f64>>,
    components: usize,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if components == 0 || rows.is_empty() || components >= rows[0].len() {
        return Ok(rows);
    }
    let n = rows.len();
    let p = rows[0].len();
    let k = components.min(p);
    let mut centered = rows;
    let mut means = Vec::new();
    means.try_reserve_exact(p).map_err(|source| {
        super::internal_with_source("tsne: PCA means exceed available memory", source)
    })?;
    for col in 0..p {
        let mut total = 0.0;
        for row in &centered {
            total += row[col];
            cooperation.completed(1).await?;
        }
        means.push(total / n as f64);
    }
    for row in &mut centered {
        for col in 0..p {
            row[col] -= means[col];
            cooperation.completed(1).await?;
        }
    }
    let mut cov: DMatrix<f64> = super::workspace::zero_matrix(p, p, cooperation).await?;
    let denom = n.saturating_sub(1).max(1) as f64;
    for row in &centered {
        for a in 0..p {
            for b in a..p {
                cov[(a, b)] += row[a] * row[b] / denom;
                cooperation.completed(1).await?;
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            cov[(a, b)] = cov[(b, a)];
            cooperation.completed(1).await?;
        }
    }
    let eig = SymmetricEigen::new(cov);
    let mut order: Vec<(usize, f64)> = Vec::new();
    order.try_reserve_exact(p).map_err(|source| {
        super::internal_with_source("tsne: PCA ordering exceeds available memory", source)
    })?;
    for idx in 0..p {
        order.push((idx, eig.eigenvalues[idx]));
        cooperation.completed(1).await?;
    }
    cooperation
        .stable_sort_by(&mut order, |left, right| {
            right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal)
        })
        .await?;
    let mut projected = super::workspace::zero_rows(n, k, cooperation).await?;
    for (new_col, (eig_col, _)) in order.into_iter().take(k).enumerate() {
        for row in 0..n {
            let mut value = 0.0;
            for old_col in 0..p {
                value += centered[row][old_col] * eig.eigenvectors[(old_col, eig_col)];
                cooperation.completed(1).await?;
            }
            projected[row][new_col] = value;
            cooperation.completed(1).await?;
        }
    }
    Ok(projected)
}

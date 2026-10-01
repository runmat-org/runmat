use super::{invalid, Options, PreparedData, MAX_WORK_CELLS};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use nalgebra::{DMatrix, SymmetricEigen};
use runmat_value::Tensor;
use std::cmp::Ordering;

pub(super) fn prepare_data(x: &Tensor) -> BuiltinResult<PreparedData> {
    let values = tensor::tensor_values_f64_cow(x);
    let mut rows = Vec::new();
    let mut good_rows = Vec::new();
    for row in 0..x.rows {
        let mut out = Vec::with_capacity(x.cols);
        let mut complete = true;
        for col in 0..x.cols {
            let value = matrix_value(values.as_ref(), x.rows, row, col);
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
    }
    Ok(PreparedData {
        rows,
        good_rows,
        original_rows: x.rows,
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

pub(super) fn standardize_rows(rows: &mut [Vec<f64>]) {
    if rows.is_empty() {
        return;
    }
    let cols = rows[0].len();
    for col in 0..cols {
        let mean = rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64;
        let var = rows
            .iter()
            .map(|row| {
                let diff = row[col] - mean;
                diff * diff
            })
            .sum::<f64>()
            / (rows.len().saturating_sub(1).max(1) as f64);
        let scale = var.sqrt();
        for row in rows.iter_mut() {
            row[col] = if scale > 0.0 {
                (row[col] - mean) / scale
            } else {
                0.0
            };
        }
    }
}

pub(super) fn apply_pca_if_requested(
    rows: Vec<Vec<f64>>,
    components: usize,
) -> BuiltinResult<Vec<Vec<f64>>> {
    if components == 0 || rows.is_empty() || components >= rows[0].len() {
        return Ok(rows);
    }
    let n = rows.len();
    let p = rows[0].len();
    let k = components.min(p);
    let mut centered = rows;
    let means = (0..p)
        .map(|col| centered.iter().map(|row| row[col]).sum::<f64>() / n as f64)
        .collect::<Vec<_>>();
    for row in &mut centered {
        for col in 0..p {
            row[col] -= means[col];
        }
    }
    let mut cov: DMatrix<f64> = DMatrix::zeros(p, p);
    let denom = n.saturating_sub(1).max(1) as f64;
    for row in &centered {
        for a in 0..p {
            for b in a..p {
                cov[(a, b)] += row[a] * row[b] / denom;
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            cov[(a, b)] = cov[(b, a)];
        }
    }
    let eig = SymmetricEigen::new(cov);
    let mut order: Vec<(usize, f64)> = (0..p)
        .map(|idx| (idx, eig.eigenvalues[idx]))
        .collect::<Vec<_>>();
    order.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let mut projected = vec![vec![0.0; k]; n];
    for (new_col, (eig_col, _)) in order.into_iter().take(k).enumerate() {
        for row in 0..n {
            let mut value = 0.0;
            for old_col in 0..p {
                value += centered[row][old_col] * eig.eigenvectors[(old_col, eig_col)];
            }
            projected[row][new_col] = value;
        }
    }
    Ok(projected)
}

pub(super) fn matrix_value(values: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    values[row + col * rows]
}

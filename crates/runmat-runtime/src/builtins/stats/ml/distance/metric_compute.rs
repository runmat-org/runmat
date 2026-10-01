//! Host pairwise metric evaluation over admitted real observation matrices.

use std::cmp::Ordering;

use runmat_value::Tensor;

use super::{distance_error, internal_error, DistanceMetric, EPS, PDIST_NAME};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

pub(super) fn condensed_distances(
    name: &'static str,
    x: &Tensor,
    metric: &DistanceMetric,
) -> BuiltinResult<Tensor> {
    let rows = x.rows;
    let len = rows
        .checked_mul(rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal_error(name, "pdist: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(name, "pdist: failed to allocate output vector"))?;
    for col in 0..rows {
        for row in (col + 1)..rows {
            out.push(row_distance(name, x, row, x, col, metric)?);
        }
    }
    Tensor::new(out, vec![1, len]).map_err(|err| internal_error(name, format!("pdist: {err}")))
}

pub(super) fn distance_matrix(
    name: &'static str,
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
) -> BuiltinResult<Tensor> {
    let len = x
        .rows
        .checked_mul(y.rows)
        .ok_or_else(|| internal_error(name, "pdist2: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(name, "pdist2: failed to allocate output matrix"))?;
    for col in 0..y.rows {
        for row in 0..x.rows {
            out.push(row_distance(name, x, row, y, col, metric)?);
        }
    }
    Tensor::new(out, vec![x.rows, y.rows])
        .map_err(|err| internal_error(name, format!("pdist2: {err}")))
}

pub(super) fn row_distance(
    name: &'static str,
    x: &Tensor,
    xi: usize,
    y: &Tensor,
    yi: usize,
    metric: &DistanceMetric,
) -> BuiltinResult<f64> {
    if row_has_nan(x, xi) || row_has_nan(y, yi) {
        return Ok(f64::NAN);
    }
    match metric {
        DistanceMetric::Euclidean => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff;
            }
            Ok(sum.sqrt())
        }
        DistanceMetric::SquareEuclidean => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff;
            }
            Ok(sum)
        }
        DistanceMetric::Cityblock => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                sum += (row_value(x, xi, col) - row_value(y, yi, col)).abs();
            }
            Ok(sum)
        }
        DistanceMetric::Chebychev => {
            let mut max = 0.0;
            for col in 0..x.cols {
                let diff = (row_value(x, xi, col) - row_value(y, yi, col)).abs();
                if diff > max {
                    max = diff;
                }
            }
            Ok(max)
        }
        DistanceMetric::Minkowski(p) => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                sum += (row_value(x, xi, col) - row_value(y, yi, col))
                    .abs()
                    .powf(*p);
            }
            Ok(sum.powf(1.0 / p))
        }
        DistanceMetric::Seuclidean(scale) => {
            let mut sum = 0.0;
            for (col, scale_value) in scale.iter().enumerate().take(x.cols) {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff / *scale_value;
            }
            Ok(sum.sqrt())
        }
        DistanceMetric::Mahalanobis(inv_covariance) => {
            let diff: Vec<f64> = (0..x.cols)
                .map(|col| row_value(x, xi, col) - row_value(y, yi, col))
                .collect();
            let mut total = 0.0;
            for row in 0..x.cols {
                for col in 0..x.cols {
                    total += diff[row] * matrix_value(inv_covariance, x.cols, row, col) * diff[col];
                }
            }
            if total < -EPS {
                return Err(distance_error(
                    name,
                    format!("{name}: mahalanobis covariance produced a negative quadratic form"),
                ));
            }
            Ok(total.max(0.0).sqrt())
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0;
            let mut x_norm = 0.0;
            let mut y_norm = 0.0;
            for col in 0..x.cols {
                let xv = row_value(x, xi, col);
                let yv = row_value(y, yi, col);
                dot += xv * yv;
                x_norm += xv * xv;
                y_norm += yv * yv;
            }
            if x_norm <= EPS || y_norm <= EPS {
                Ok(f64::NAN)
            } else {
                Ok(1.0 - dot / (x_norm.sqrt() * y_norm.sqrt()))
            }
        }
        DistanceMetric::Correlation => correlation_distance(x, xi, y, yi),
        DistanceMetric::Hamming => {
            let mut unequal = 0usize;
            for col in 0..x.cols {
                if row_value(x, xi, col) != row_value(y, yi, col) {
                    unequal += 1;
                }
            }
            Ok(unequal as f64 / x.cols as f64)
        }
        DistanceMetric::Jaccard => {
            let mut unequal_nonzero = 0usize;
            let mut nonzero_union = 0usize;
            for col in 0..x.cols {
                let xv = row_value(x, xi, col);
                let yv = row_value(y, yi, col);
                if xv != 0.0 || yv != 0.0 {
                    nonzero_union += 1;
                    if xv != yv {
                        unequal_nonzero += 1;
                    }
                }
            }
            if nonzero_union == 0 {
                Ok(0.0)
            } else {
                Ok(unequal_nonzero as f64 / nonzero_union as f64)
            }
        }
        DistanceMetric::Spearman => spearman_distance(x, xi, y, yi),
    }
}

pub(super) fn row_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    tensor::tensor_value_f64(tensor, col * tensor.rows + row)
}

fn row_has_nan(tensor: &Tensor, row: usize) -> bool {
    (0..tensor.cols).any(|col| row_value(tensor, row, col).is_nan())
}

pub(super) fn matrix_value(data: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    data[col * rows + row]
}

fn correlation_distance(x: &Tensor, xi: usize, y: &Tensor, yi: usize) -> BuiltinResult<f64> {
    let cols = x.cols;
    if cols == 0 {
        return Ok(f64::NAN);
    }
    let x_mean = (0..cols).map(|col| row_value(x, xi, col)).sum::<f64>() / cols as f64;
    let y_mean = (0..cols).map(|col| row_value(y, yi, col)).sum::<f64>() / cols as f64;
    let mut dot = 0.0;
    let mut x_norm = 0.0;
    let mut y_norm = 0.0;
    for col in 0..cols {
        let xv = row_value(x, xi, col) - x_mean;
        let yv = row_value(y, yi, col) - y_mean;
        dot += xv * yv;
        x_norm += xv * xv;
        y_norm += yv * yv;
    }
    if x_norm <= EPS || y_norm <= EPS {
        Ok(f64::NAN)
    } else {
        Ok(1.0 - dot / (x_norm.sqrt() * y_norm.sqrt()))
    }
}

fn spearman_distance(x: &Tensor, xi: usize, y: &Tensor, yi: usize) -> BuiltinResult<f64> {
    let x_rank = row_ranks(x, xi);
    let y_rank = row_ranks(y, yi);
    let rank_x = Tensor::new(x_rank, vec![1, x.cols])
        .map_err(|err| internal_error(PDIST_NAME, format!("spearman: {err}")))?;
    let rank_y = Tensor::new(y_rank, vec![1, y.cols])
        .map_err(|err| internal_error(PDIST_NAME, format!("spearman: {err}")))?;
    correlation_distance(&rank_x, 0, &rank_y, 0)
}

fn row_ranks(tensor: &Tensor, row: usize) -> Vec<f64> {
    let mut values = (0..tensor.cols)
        .map(|col| (col, row_value(tensor, row, col)))
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    let mut ranks = vec![0.0; tensor.cols];
    let mut idx = 0usize;
    while idx < values.len() {
        let start = idx;
        let value = values[idx].1;
        while idx < values.len() && values[idx].1 == value {
            idx += 1;
        }
        let rank = (start + 1 + idx) as f64 / 2.0;
        for entry in &values[start..idx] {
            ranks[entry.0] = rank;
        }
    }
    ranks
}

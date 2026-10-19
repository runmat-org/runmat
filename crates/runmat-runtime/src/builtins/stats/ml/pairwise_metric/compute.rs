//! Host pairwise metric evaluation over admitted real observation matrices.

use std::cmp::Ordering;

use runmat_value::Tensor;

use super::{distance_error, internal_error, DistanceMetric, MetricCaller, EPS};
use crate::builtins::common::tensor;
use crate::BuiltinResult;

/// Projects a dense value into the observation-matrix geometry used by the
/// pairwise metric domain.
///
/// A rank-one value is one observation with N variables, while a rank-zero
/// value has no observation extent. This is intentionally distinct from the
/// general-purpose `Tensor::rows`/`Tensor::cols` projection.
pub(in crate::builtins::stats::ml) fn observation_matrix_dimensions(
    tensor: &Tensor,
) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(in crate::builtins::stats::ml) fn condensed_distances(
    name: MetricCaller,
    x: &Tensor,
    metric: &DistanceMetric,
) -> BuiltinResult<Tensor> {
    let rows = observation_matrix_dimensions(x).0;
    let len = rows
        .checked_mul(rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal_error(name, format!("{name}: output size overflow")))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(name, format!("{name}: failed to allocate output vector")))?;
    for col in 0..rows {
        for row in (col + 1)..rows {
            out.push(row_distance(name, x, row, x, col, metric)?);
        }
    }
    Tensor::new(out, vec![1, len]).map_err(|err| internal_error(name, format!("{name}: {err}")))
}

pub(in crate::builtins::stats::ml) fn distance_matrix(
    name: MetricCaller,
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
) -> BuiltinResult<Tensor> {
    let x_rows = observation_matrix_dimensions(x).0;
    let y_rows = observation_matrix_dimensions(y).0;
    let len = x_rows
        .checked_mul(y_rows)
        .ok_or_else(|| internal_error(name, format!("{name}: output size overflow")))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(name, format!("{name}: failed to allocate output matrix")))?;
    for col in 0..y_rows {
        for row in 0..x_rows {
            out.push(row_distance(name, x, row, y, col, metric)?);
        }
    }
    Tensor::new(out, vec![x_rows, y_rows])
        .map_err(|err| internal_error(name, format!("{name}: {err}")))
}

pub(in crate::builtins::stats::ml) fn row_distance(
    name: MetricCaller,
    x: &Tensor,
    xi: usize,
    y: &Tensor,
    yi: usize,
    metric: &DistanceMetric,
) -> BuiltinResult<f64> {
    if row_has_nan(x, xi) || row_has_nan(y, yi) {
        return Ok(f64::NAN);
    }
    let columns = observation_matrix_dimensions(x).1;
    match metric {
        DistanceMetric::Euclidean => Ok(scaled_lp_distance(x, xi, y, yi, 2.0)),
        DistanceMetric::SquareEuclidean => {
            let mut sum = 0.0;
            for col in 0..columns {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff;
            }
            Ok(sum)
        }
        DistanceMetric::Cityblock => {
            let mut sum = 0.0;
            for col in 0..columns {
                sum += (row_value(x, xi, col) - row_value(y, yi, col)).abs();
            }
            Ok(sum)
        }
        DistanceMetric::Chebychev => {
            let mut max = 0.0;
            for col in 0..columns {
                let diff = (row_value(x, xi, col) - row_value(y, yi, col)).abs();
                if diff > max {
                    max = diff;
                }
            }
            Ok(max)
        }
        DistanceMetric::Minkowski(p) => Ok(scaled_lp_distance(x, xi, y, yi, *p)),
        DistanceMetric::Seuclidean(scale) => {
            let mut sum = 0.0;
            for (col, scale_value) in scale.iter().enumerate().take(columns) {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff / *scale_value;
            }
            Ok(sum.sqrt())
        }
        DistanceMetric::Mahalanobis(inv_covariance) => {
            let diff: Vec<f64> = (0..columns)
                .map(|col| row_value(x, xi, col) - row_value(y, yi, col))
                .collect();
            let mut total = 0.0;
            for row in 0..columns {
                for col in 0..columns {
                    total +=
                        diff[row] * matrix_value(inv_covariance, columns, row, col) * diff[col];
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
            for col in 0..columns {
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
            for col in 0..columns {
                if row_value(x, xi, col) != row_value(y, yi, col) {
                    unequal += 1;
                }
            }
            Ok(unequal as f64 / columns as f64)
        }
        DistanceMetric::Jaccard => {
            let mut unequal_nonzero = 0usize;
            let mut nonzero_union = 0usize;
            for col in 0..columns {
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
        DistanceMetric::Spearman => spearman_distance(name, x, xi, y, yi),
    }
}

/// Accumulate a positive-order Lp distance without overflowing intermediate
/// powers when the representable final distance is finite.
fn scaled_lp_distance(x: &Tensor, xi: usize, y: &Tensor, yi: usize, p: f64) -> f64 {
    let mut scale = 0.0_f64;
    let columns = observation_matrix_dimensions(x).1;
    for col in 0..columns {
        let difference = (row_value(x, xi, col) - row_value(y, yi, col)).abs();
        if difference.is_nan() {
            return f64::NAN;
        }
        scale = scale.max(difference);
    }
    if scale == 0.0 || scale.is_infinite() {
        return scale;
    }
    let mut normalized_sum = 0.0;
    for col in 0..columns {
        let difference = (row_value(x, xi, col) - row_value(y, yi, col)).abs();
        normalized_sum += (difference / scale).powf(p);
    }
    scale * normalized_sum.powf(1.0 / p)
}

pub(in crate::builtins::stats::ml) fn row_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    let rows = observation_matrix_dimensions(tensor).0;
    tensor::tensor_value_f64(tensor, col * rows + row)
}

fn row_has_nan(tensor: &Tensor, row: usize) -> bool {
    let columns = observation_matrix_dimensions(tensor).1;
    (0..columns).any(|col| row_value(tensor, row, col).is_nan())
}

pub(super) fn matrix_value(data: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    data[col * rows + row]
}

fn correlation_distance(x: &Tensor, xi: usize, y: &Tensor, yi: usize) -> BuiltinResult<f64> {
    let cols = observation_matrix_dimensions(x).1;
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

fn spearman_distance(
    name: MetricCaller,
    x: &Tensor,
    xi: usize,
    y: &Tensor,
    yi: usize,
) -> BuiltinResult<f64> {
    let x_rank = row_ranks(x, xi);
    let y_rank = row_ranks(y, yi);
    let x_columns = observation_matrix_dimensions(x).1;
    let y_columns = observation_matrix_dimensions(y).1;
    let rank_x = Tensor::new(x_rank, vec![1, x_columns])
        .map_err(|err| internal_error(name, format!("spearman: {err}")))?;
    let rank_y = Tensor::new(y_rank, vec![1, y_columns])
        .map_err(|err| internal_error(name, format!("spearman: {err}")))?;
    correlation_distance(&rank_x, 0, &rank_y, 0)
}

fn row_ranks(tensor: &Tensor, row: usize) -> Vec<f64> {
    let columns = observation_matrix_dimensions(tensor).1;
    let mut values = (0..columns)
        .map(|col| (col, row_value(tensor, row, col)))
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    let mut ranks = vec![0.0; columns];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn rank_one_values_remain_single_observations_with_multiple_variables() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid rank-one tensor");
        let y = Tensor::new(vec![4.0, 6.0, 8.0], vec![3]).expect("valid rank-one tensor");

        assert_eq!(observation_matrix_dimensions(&x), (1, 3));
        let distances = distance_matrix(MetricCaller::Pdist2, &x, &y, &DistanceMetric::Euclidean)
            .expect("rank-one inputs are one observation each");
        assert_eq!(distances.shape(), &[1, 1]);
        assert!((distances.materialize_f64()[0] - 50.0_f64.sqrt()).abs() < 1.0e-12);

        let scalar = Tensor::new(vec![1.0], vec![]).expect("valid rank-zero tensor");
        assert_eq!(observation_matrix_dimensions(&scalar), (0, 0));
        let condensed =
            condensed_distances(MetricCaller::Pdist, &scalar, &DistanceMetric::Euclidean)
                .expect("rank-zero input retains empty observation geometry");
        assert_eq!(condensed.shape(), &[1, 0]);
    }
}

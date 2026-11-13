//! Scale and covariance geometry used by standardized and Mahalanobis metrics.

use super::compute::{matrix_value, observation_matrix_dimensions, row_value};
use super::{distance_error, internal_error, MetricCaller, EPS};
use crate::BuiltinResult;
use runmat_value::Tensor;

pub(super) fn variance_scale(name: MetricCaller, x: &Tensor) -> BuiltinResult<Vec<f64>> {
    let (rows, cols) = observation_matrix_dimensions(x);
    let mut counts = vec![0usize; cols];
    let mut means = vec![0.0; cols];
    for row in 0..rows {
        for col in 0..cols {
            let value = row_value(x, row, col);
            if !value.is_nan() {
                counts[col] += 1;
                means[col] += value;
            }
        }
    }
    for col in 0..cols {
        if counts[col] < 2 {
            return Err(distance_error(
                name,
                format!("{name}: at least two non-NaN observations are required per column for seuclidean scale"),
            ));
        }
        means[col] /= counts[col] as f64;
    }
    let mut vars = vec![0.0; cols];
    for row in 0..rows {
        for col in 0..cols {
            let value = row_value(x, row, col);
            if !value.is_nan() {
                let diff = value - means[col];
                vars[col] += diff * diff;
            }
        }
    }
    for col in 0..cols {
        vars[col] /= (counts[col] - 1) as f64;
        if vars[col] <= EPS {
            vars[col] = f64::NAN;
        }
    }
    Ok(vars)
}

pub(super) fn covariance_matrix(name: MetricCaller, x: &Tensor) -> BuiltinResult<Vec<f64>> {
    let (rows, cols) = observation_matrix_dimensions(x);
    if rows < 2 {
        return Err(distance_error(
            name,
            format!("{name}: at least two observations are required for mahalanobis covariance"),
        ));
    }
    let mut means = vec![0.0; cols];
    for row in 0..rows {
        for (col, mean) in means.iter_mut().enumerate().take(cols) {
            *mean += row_value(x, row, col);
        }
    }
    for mean in &mut means {
        *mean /= rows as f64;
    }
    let len = cols
        .checked_mul(cols)
        .ok_or_else(|| internal_error(name, "mahalanobis covariance size overflow"))?;
    let mut covariance = Vec::new();
    covariance
        .try_reserve(len)
        .map_err(|_| internal_error(name, "mahalanobis covariance allocation failed"))?;
    covariance.resize(len, 0.0);
    accumulate_covariance(&mut covariance, x, &means);
    for value in &mut covariance {
        *value /= (rows - 1) as f64;
    }
    Ok(covariance)
}

fn accumulate_covariance(out: &mut [f64], data: &Tensor, means: &[f64]) {
    let (rows, cols) = observation_matrix_dimensions(data);
    for row in 0..rows {
        for a in 0..cols {
            let av = row_value(data, row, a) - means[a];
            for b in 0..cols {
                let bv = row_value(data, row, b) - means[b];
                out[b * cols + a] += av * bv;
            }
        }
    }
}

pub(super) fn invert_spd_matrix(
    name: MetricCaller,
    matrix: &[f64],
    size: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    validate_symmetric_positive_definite(name, matrix, size, label)?;
    let width = size
        .checked_mul(2)
        .ok_or_else(|| internal_error(name, format!("{label}: augmented width overflow")))?;
    let len = size
        .checked_mul(width)
        .ok_or_else(|| internal_error(name, format!("{label}: augmented matrix overflow")))?;
    let mut aug = Vec::new();
    aug.try_reserve(len).map_err(|_| {
        internal_error(name, format!("{label}: augmented matrix allocation failed"))
    })?;
    aug.resize(len, 0.0);
    for row in 0..size {
        for col in 0..size {
            aug[row * width + col] = matrix_value(matrix, size, row, col);
        }
        aug[row * width + size + row] = 1.0;
    }
    for pivot in 0..size {
        let mut best = pivot;
        let mut best_abs = aug[pivot * width + pivot].abs();
        for row in (pivot + 1)..size {
            let value = aug[row * width + pivot].abs();
            if value > best_abs {
                best = row;
                best_abs = value;
            }
        }
        if best_abs <= EPS {
            return Err(distance_error(
                name,
                format!("{name}: {label} must be nonsingular"),
            ));
        }
        if best != pivot {
            for col in 0..width {
                aug.swap(pivot * width + col, best * width + col);
            }
        }
        let pivot_value = aug[pivot * width + pivot];
        for col in 0..width {
            aug[pivot * width + col] /= pivot_value;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = aug[row * width + pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..width {
                aug[row * width + col] -= factor * aug[pivot * width + col];
            }
        }
    }
    let inv_len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(name, format!("{label}: inverse size overflow")))?;
    let mut inv = Vec::new();
    inv.try_reserve(inv_len)
        .map_err(|_| internal_error(name, format!("{label}: inverse allocation failed")))?;
    inv.resize(inv_len, 0.0);
    for row in 0..size {
        for col in 0..size {
            inv[col * size + row] = aug[row * width + size + col];
        }
    }
    Ok(inv)
}

fn validate_symmetric_positive_definite(
    name: MetricCaller,
    matrix: &[f64],
    size: usize,
    label: &str,
) -> BuiltinResult<()> {
    for row in 0..size {
        for col in 0..size {
            let a = matrix_value(matrix, size, row, col);
            let b = matrix_value(matrix, size, col, row);
            if !a.is_finite() || (a - b).abs() > 1.0e-9 {
                return Err(distance_error(
                    name,
                    format!("{name}: {label} must be symmetric positive definite"),
                ));
            }
        }
    }
    let len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(name, format!("{label}: Cholesky size overflow")))?;
    let mut chol = Vec::new();
    chol.try_reserve(len)
        .map_err(|_| internal_error(name, format!("{label}: Cholesky allocation failed")))?;
    chol.resize(len, 0.0);
    for i in 0..size {
        for j in 0..=i {
            let mut sum = matrix_value(matrix, size, i, j);
            for k in 0..j {
                sum -= chol[i * size + k] * chol[j * size + k];
            }
            if i == j {
                if sum <= EPS {
                    return Err(distance_error(
                        name,
                        format!("{name}: {label} must be symmetric positive definite"),
                    ));
                }
                chol[i * size + j] = sum.sqrt();
            } else {
                chol[i * size + j] = sum / chol[j * size + j];
            }
        }
    }
    Ok(())
}

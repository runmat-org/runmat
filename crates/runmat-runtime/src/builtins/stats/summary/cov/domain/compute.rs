use super::super::*;
use super::geometry::Matrix;

pub(in super::super) fn covariance_dense(
    matrix: &Matrix,
    weight: &CovWeightSpec,
) -> BuiltinResult<Tensor> {
    let cols = matrix.cols;
    let rows = matrix.rows;

    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(cov_internal_error);
    }

    let mut result = vec![f64::NAN; cols * cols];

    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let denom = match normalization {
                CovNormalization::Unbiased => ((rows as f64) - 1.0).max(1.0),
                CovNormalization::Biased => rows as f64,
            };
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut sum = 0.0;
                let mut valid = true;
                for &value in column {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    sum += value;
                }
                *mean_slot = if valid { sum / (rows as f64) } else { f64::NAN };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_unweighted_pair(matrix, i, j, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
        CovWeightSpec::Vector(weights) => {
            if weights.len() != rows {
                return Err(cov_error_with_detail(
                    &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
                    format!("expected {rows} elements"),
                ));
            }
            let sum_w: f64 = weights.iter().sum();
            if sum_w <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }
            let denom = sum_w - 1.0;
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut weighted_sum = 0.0;
                let mut valid = true;
                for (row, &value) in column.iter().enumerate() {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    weighted_sum += weights[row] * value;
                }
                *mean_slot = if valid {
                    weighted_sum / sum_w
                } else {
                    f64::NAN
                };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_weighted_pair(matrix, i, j, weights, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
    }

    Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error)
}

pub(in super::super) fn filter_complete_rows(
    matrix: &Matrix,
    weight: CovWeightSpec,
) -> (Matrix, CovWeightSpec) {
    if matrix.rows == 0 {
        return (
            Matrix {
                data: Vec::new(),
                rows: 0,
                cols: matrix.cols,
            },
            weight,
        );
    }

    let mut valid_rows = Vec::new();
    for row in 0..matrix.rows {
        let mut is_valid = true;
        for col in 0..matrix.cols {
            if !matrix.get(row, col).is_finite() {
                is_valid = false;
                break;
            }
        }
        if is_valid {
            valid_rows.push(row);
        }
    }

    if valid_rows.len() == matrix.rows {
        // No filtering required.
        return (matrix.clone(), weight);
    }

    let mut data = Vec::with_capacity(valid_rows.len() * matrix.cols);
    for col in 0..matrix.cols {
        for &row in &valid_rows {
            data.push(matrix.get(row, col));
        }
    }

    let filtered_matrix = Matrix {
        data,
        rows: valid_rows.len(),
        cols: matrix.cols,
    };

    let filtered_weight = match weight {
        CovWeightSpec::Scalar(norm) => CovWeightSpec::Scalar(norm),
        CovWeightSpec::Vector(vec) => {
            let mut filtered = Vec::with_capacity(valid_rows.len());
            for &row in &valid_rows {
                filtered.push(vec[row]);
            }
            CovWeightSpec::Vector(filtered)
        }
    };

    (filtered_matrix, filtered_weight)
}

pub(in super::super) fn covariance_pairwise(
    matrix: &Matrix,
    weight: &CovWeightSpec,
) -> BuiltinResult<Tensor> {
    let cols = matrix.cols;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(cov_internal_error);
    }
    let mut result = vec![f64::NAN; cols * cols];
    for i in 0..cols {
        let variance = covariance_pair(matrix, i, i, weight);
        set_entry(&mut result, cols, i, i, sanitize_covariance(true, variance));
        for j in (i + 1)..cols {
            let value = covariance_pair(matrix, i, j, weight);
            set_entry(&mut result, cols, i, j, sanitize_covariance(false, value));
        }
    }
    Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error)
}

fn covariance_unweighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_weighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    weights: &[f64],
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
        if weight == 0.0 {
            continue;
        }
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += weight * (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_pair(matrix: &Matrix, lhs: usize, rhs: usize, weight: &CovWeightSpec) -> f64 {
    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for row in 0..matrix.rows {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                }
            }
            covariance_unweighted_slice(&xs, &ys, *normalization)
        }
        CovWeightSpec::Vector(weights) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut ws = Vec::new();
            for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                    ws.push(weight);
                }
            }
            covariance_weighted_slice(&xs, &ys, &ws)
        }
    }
}

fn covariance_unweighted_slice(xs: &[f64], ys: &[f64], normalization: CovNormalization) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len());
    if n == 0 {
        return f64::NAN;
    }
    let denom = match normalization {
        CovNormalization::Unbiased => (n as f64) - 1.0,
        CovNormalization::Biased => n as f64,
    };
    if denom <= 0.0 {
        return f64::NAN;
    }
    let sum_x: f64 = xs.iter().take(n).sum();
    let sum_y: f64 = ys.iter().take(n).sum();
    let mean_x = sum_x / (n as f64);
    let mean_y = sum_y / (n as f64);
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn covariance_weighted_slice(xs: &[f64], ys: &[f64], weights: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() || weights.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len()).min(weights.len());
    if n == 0 {
        return f64::NAN;
    }
    let sum_w: f64 = weights.iter().take(n).sum();
    if sum_w <= 0.0 {
        return f64::NAN;
    }
    let denom = sum_w - 1.0;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for idx in 0..n {
        mean_x += weights[idx] * xs[idx];
        mean_y += weights[idx] * ys[idx];
    }
    mean_x /= sum_w;
    mean_y /= sum_w;
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += weights[idx] * (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn sanitize_covariance(is_diag: bool, value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if is_diag && value < 0.0 && value > -1.0e-12 {
        0.0
    } else {
        value
    }
}

fn set_entry(buffer: &mut [f64], dim: usize, row: usize, col: usize, value: f64) {
    let idx = row + col * dim;
    buffer[idx] = value;
    if row != col {
        let symmetrical = col + row * dim;
        buffer[symmetrical] = value;
    }
}

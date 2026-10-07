use super::{internal_error, invalid_argument, EPS};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use nalgebra::{DMatrix, DVector};
use runmat_value::Tensor;

#[derive(Clone, Debug)]
struct PreparedRidge {
    singular_values: Vec<f64>,
    right_singular_vectors: DMatrix<f64>,
    u_transpose_y: Vec<f64>,
    x_means: Vec<f64>,
    x_stds: Vec<f64>,
    y_mean: f64,
    sv_tolerance: f64,
    predictors: usize,
}

fn ridge_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn ridge_compute(
    y: Tensor,
    x: Tensor,
    k: Tensor,
    scaled: bool,
) -> BuiltinResult<Tensor> {
    let y_values = vector_values(&y, "y")?;
    if x.shape().len() > 2 {
        return Err(invalid_argument("ridge: X must be a 2-D numeric matrix"));
    }
    let (x_rows, _) = ridge_matrix_dimensions(&x);
    if x_rows != y_values.len() {
        return Err(invalid_argument(
            "ridge: length(y) must match the number of rows in X",
        ));
    }
    let lambdas = ridge_parameters(&k)?;
    let prepared = prepare_data(&x, &y_values)?;
    let rows = if scaled {
        prepared.predictors
    } else {
        prepared
            .predictors
            .checked_add(1)
            .ok_or_else(|| internal_error("ridge: output row count overflow"))?
    };
    let cols = lambdas.len();
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal_error("ridge: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error("ridge: output allocation failed"))?;
    for lambda in lambdas {
        let beta_scaled = solve_ridge(&prepared, lambda)?;
        if scaled {
            out.extend(beta_scaled.iter().copied());
        } else {
            let beta_original = beta_scaled
                .iter()
                .zip(prepared.x_stds.iter())
                .map(|(coef, std)| if *std > EPS { coef / std } else { 0.0 })
                .collect::<Vec<_>>();
            let intercept = prepared.y_mean
                - prepared
                    .x_means
                    .iter()
                    .zip(beta_original.iter())
                    .map(|(mean, coef)| mean * coef)
                    .sum::<f64>();
            out.push(intercept);
            out.extend(beta_original);
        }
    }
    Tensor::new(out, vec![rows, cols]).map_err(|err| internal_error(format!("ridge: {err}")))
}

fn vector_values(tensor: &Tensor, label: &str) -> BuiltinResult<Vec<f64>> {
    let (rows, cols) = ridge_matrix_dimensions(tensor);
    if tensor.shape().len() > 2 || !(rows == 1 || cols == 1) {
        return Err(invalid_argument(format!("ridge: {label} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

fn ridge_parameters(tensor: &Tensor) -> BuiltinResult<Vec<f64>> {
    let values = vector_values(tensor, "k")?;
    if values.is_empty() {
        return Err(invalid_argument(
            "ridge: k must contain at least one ridge parameter",
        ));
    }
    for value in &values {
        if !value.is_finite() || *value < 0.0 {
            return Err(invalid_argument(
                "ridge: k values must be nonnegative finite scalars",
            ));
        }
    }
    Ok(values)
}

fn prepare_data(x: &Tensor, y: &[f64]) -> BuiltinResult<PreparedRidge> {
    let (rows, cols) = ridge_matrix_dimensions(x);
    let x_values = tensor::tensor_values_f64_cow(x);
    let mut clean_rows = Vec::with_capacity(rows);
    for row in 0..rows {
        let y_value = y[row];
        let mut has_nan = y_value.is_nan();
        let mut has_nonfinite = y_value.is_infinite();
        for col in 0..cols {
            let value = x_value(&x_values, rows, row, col);
            has_nan |= value.is_nan();
            has_nonfinite |= value.is_infinite();
        }
        if has_nonfinite {
            return Err(invalid_argument(
                "ridge: X and y must not contain Inf values",
            ));
        }
        if !has_nan {
            clean_rows.push(row);
        }
    }
    if clean_rows.len() < 2 {
        return Err(invalid_argument(
            "ridge: at least two complete observations are required",
        ));
    }
    let n = clean_rows.len();
    let mut x_means = vec![0.0; cols];
    let mut y_mean = 0.0;
    for &row in &clean_rows {
        y_mean += y[row];
        for (col, mean) in x_means.iter_mut().enumerate().take(cols) {
            *mean += x_value(&x_values, rows, row, col);
        }
    }
    y_mean /= n as f64;
    for mean in &mut x_means {
        *mean /= n as f64;
    }
    let mut x_stds = vec![0.0; cols];
    for &row in &clean_rows {
        for (col, std) in x_stds.iter_mut().enumerate().take(cols) {
            let diff = x_value(&x_values, rows, row, col) - x_means[col];
            *std += diff * diff;
        }
    }
    for std in &mut x_stds {
        *std = (*std / (n - 1) as f64).sqrt();
        if *std <= EPS {
            *std = 1.0;
        }
    }
    let mut z_data = Vec::new();
    z_data
        .try_reserve(
            n.checked_mul(cols)
                .ok_or_else(|| internal_error("ridge: standardized design matrix size overflow"))?,
        )
        .map_err(|_| internal_error("ridge: standardized design matrix allocation failed"))?;
    for col in 0..cols {
        for &row in &clean_rows {
            z_data.push((x_value(&x_values, rows, row, col) - x_means[col]) / x_stds[col]);
        }
    }
    let y_centered = clean_rows
        .iter()
        .map(|&row| y[row] - y_mean)
        .collect::<Vec<_>>();
    let y_centered = DVector::from_column_slice(&y_centered);
    let z = DMatrix::from_column_slice(n, cols, &z_data);
    let svd = z.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| internal_error("ridge: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| internal_error("ridge: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let largest_sv = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let sv_tolerance = (n.max(cols) as f64) * f64::EPSILON * largest_sv.max(1.0);
    let u_transpose_y = (0..singular_values.len())
        .map(|index| u.column(index).dot(&y_centered))
        .collect::<Vec<_>>();
    Ok(PreparedRidge {
        singular_values,
        right_singular_vectors: v_t.transpose(),
        u_transpose_y,
        x_means,
        x_stds,
        y_mean,
        sv_tolerance,
        predictors: cols,
    })
}

fn solve_ridge(data: &PreparedRidge, lambda: f64) -> BuiltinResult<DVector<f64>> {
    let mut beta = DVector::zeros(data.predictors);
    for (index, singular_value) in data.singular_values.iter().copied().enumerate() {
        let denominator = singular_value.mul_add(singular_value, lambda);
        if denominator <= data.sv_tolerance {
            continue;
        }
        let contribution = (singular_value / denominator) * data.u_transpose_y[index];
        for row in 0..data.predictors {
            beta[row] += data.right_singular_vectors[(row, index)] * contribution;
        }
    }
    Ok(beta)
}

fn x_value(values: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    values[col * rows + row]
}

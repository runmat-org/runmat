use super::*;

pub(super) fn covariance_matrix_extents(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn validate_covariance_matrix(name: &'static str, tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape().len() > 2 {
        return Err(conversion_error(
            name,
            format!("{name}: covariance matrix must be two-dimensional"),
        ));
    }
    let (rows, cols) = covariance_matrix_extents(tensor.shape());
    if rows != cols {
        return Err(conversion_error(
            name,
            format!("{name}: covariance matrix must be square"),
        ));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    for &value in values.iter() {
        if !value.is_finite() {
            return Err(conversion_error(
                name,
                format!("{name}: covariance matrix must contain finite values"),
            ));
        }
    }
    for idx in 0..rows {
        let variance = values[idx + idx * rows];
        if variance < 0.0 {
            return Err(conversion_error(
                name,
                format!("{name}: covariance matrix diagonal entries must be nonnegative"),
            ));
        }
    }
    for col in 0..cols {
        for row in 0..col {
            let a = values[row + col * rows];
            let b = values[col + row * rows];
            let tol = 1.0e-10 * a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > tol {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
            let variance_row = values[row + row * rows];
            let variance_col = values[col + col * rows];
            let max_covariance = (variance_row * variance_col).sqrt();
            let bound_tol = 1.0e-10 * max_covariance.max(a.abs()).max(1.0);
            if a.abs() > max_covariance + bound_tol {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance magnitude exceeds variance bounds"),
                ));
            }
        }
    }
    validate_positive_semidefinite(name, rows, &values)?;
    Ok(())
}

fn validate_positive_semidefinite(
    name: &'static str,
    n: usize,
    values: &[f64],
) -> BuiltinResult<()> {
    let scale = (0..n)
        .map(|index| values[index + index * n].abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 1.0e-10 * scale;
    let mut lower = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..=row {
            let mut residual = values[row + col * n];
            for k in 0..col {
                residual -= lower[row * n + k] * lower[col * n + k];
            }
            if row == col {
                if residual < -tolerance {
                    return Err(conversion_error(
                        name,
                        format!("{name}: covariance matrix must be positive semidefinite"),
                    ));
                }
                lower[row * n + col] = residual.max(0.0).sqrt();
            } else {
                let pivot = lower[col * n + col];
                if pivot > tolerance.sqrt() {
                    lower[row * n + col] = residual / pivot;
                } else if residual.abs() > tolerance {
                    return Err(conversion_error(
                        name,
                        format!("{name}: covariance matrix must be positive semidefinite"),
                    ));
                }
            }
        }
    }
    Ok(())
}

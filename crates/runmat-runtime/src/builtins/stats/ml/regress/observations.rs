use super::*;

pub(super) fn vector_values(tensor: &Tensor) -> BuiltinResult<Vec<f64>> {
    let (rows, cols) = regress_matrix_dimensions(tensor.shape());
    if tensor.shape().len() > 2 || !(rows == 1 || cols == 1) {
        return Err(invalid("regress: y must be a vector"));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

pub(super) fn complete_rows(x: &Tensor, y: &[f64]) -> BuiltinResult<PreparedData> {
    let x_values = tensor::tensor_values_f64_cow(x);
    let (x_rows, x_cols) = regress_matrix_dimensions(x.shape());
    let mut rows = Vec::new();
    let mut y_values = Vec::new();
    for row in 0..x_rows {
        let y_value = y[row];
        let mut has_nan = y_value.is_nan();
        let mut has_inf = y_value.is_infinite();
        let start = rows.len();
        for col in 0..x_cols {
            let value = matrix_value(&x_values, x_rows, row, col);
            has_nan |= value.is_nan();
            has_inf |= value.is_infinite();
            rows.push(value);
        }
        if has_inf {
            return Err(invalid("regress: X and y must not contain Inf values"));
        }
        if has_nan {
            rows.truncate(start);
            continue;
        }
        y_values.push(y_value);
    }
    let observations = y_values.len();
    Ok(PreparedData {
        design: DMatrix::from_row_slice(observations, x_cols, &rows),
        y: DVector::from_column_slice(&y_values),
    })
}

pub(super) fn matrix_value(values: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    values[row + col * rows]
}

use super::*;

pub(super) fn matrix_dims(matrix: &Value) -> (usize, usize) {
    match matrix {
        Value::Tensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::ComplexTensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::LogicalArray(array) => (logical_rows(array), logical_cols(array)),
        Value::SparseTensor(matrix) => (matrix.rows, matrix.cols),
        _ => (1, 1),
    }
}

#[cfg(test)]
pub(super) fn is_diagonal_matrix(matrix: &Value) -> bool {
    match matrix {
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(tensor);
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row != col && values[row + col * tensor.rows()] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            let values = complex_tensor_values_pair_view(tensor);
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row != col && values.get(row + col * tensor.rows()) != (0.0, 0.0) {
                        return false;
                    }
                }
            }
            true
        }
        Value::LogicalArray(array) => {
            let rows = logical_rows(array);
            let cols = logical_cols(array);
            (0..cols)
                .all(|col| (0..rows).all(|row| row == col || array.data[row + col * rows] == 0))
        }
        _ => true,
    }
}

#[cfg(test)]
pub(super) fn is_triangular_matrix(matrix: &Value) -> bool {
    is_upper_triangular(matrix) || is_lower_triangular(matrix)
}

#[cfg(test)]
pub(super) fn is_upper_triangular(matrix: &Value) -> bool {
    match matrix {
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(tensor);
            for col in 0..tensor.cols() {
                for row in (col + 1)..tensor.rows() {
                    if values[row + col * tensor.rows()] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            let values = complex_tensor_values_pair_view(tensor);
            for col in 0..tensor.cols() {
                for row in (col + 1)..tensor.rows() {
                    if values.get(row + col * tensor.rows()) != (0.0, 0.0) {
                        return false;
                    }
                }
            }
            true
        }
        Value::LogicalArray(array) => {
            let rows = logical_rows(array);
            let cols = logical_cols(array);
            (0..cols).all(|col| ((col + 1)..rows).all(|row| array.data[row + col * rows] == 0))
        }
        _ => true,
    }
}

#[cfg(test)]
pub(super) fn is_lower_triangular(matrix: &Value) -> bool {
    match matrix {
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(tensor);
            for col in 0..tensor.cols() {
                for row in 0..col.min(tensor.rows()) {
                    if values[row + col * tensor.rows()] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            let values = complex_tensor_values_pair_view(tensor);
            for col in 0..tensor.cols() {
                for row in 0..col.min(tensor.rows()) {
                    if values.get(row + col * tensor.rows()) != (0.0, 0.0) {
                        return false;
                    }
                }
            }
            true
        }
        Value::LogicalArray(array) => {
            let rows = logical_rows(array);
            let cols = logical_cols(array);
            (0..cols).all(|col| (0..col.min(rows)).all(|row| array.data[row + col * rows] == 0))
        }
        _ => true,
    }
}

#[cfg(test)]
pub(super) fn is_hermitian_matrix(matrix: &Value) -> bool {
    let (rows, cols) = matrix_dims(matrix);
    if rows != cols {
        return false;
    }
    match matrix {
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(tensor);
            for col in 0..cols {
                for row in 0..rows {
                    if values[row + col * rows] != values[col + row * rows] {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            let values = complex_tensor_values_pair_view(tensor);
            for col in 0..cols {
                for row in 0..rows {
                    let lhs = values.get(row + col * rows);
                    let rhs = values.get(col + row * rows);
                    if lhs != (rhs.0, -rhs.1) {
                        return false;
                    }
                }
            }
            true
        }
        Value::LogicalArray(array) => (0..cols).all(|col| {
            (0..rows).all(|row| array.data[row + col * rows] == array.data[col + row * rows])
        }),
        _ => false,
    }
}

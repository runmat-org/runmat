use super::*;

pub(super) fn infer_decomposition_type(matrix: &Value) -> DecompositionType {
    let (rows, cols) = matrix_dims(matrix);
    if is_diagonal_matrix(matrix) {
        DecompositionType::Diagonal
    } else if rows == cols && is_triangular_matrix(matrix) {
        DecompositionType::Triangular
    } else if rows == cols {
        DecompositionType::Lu
    } else {
        DecompositionType::Qr
    }
}

pub(super) fn validate_type_compatibility(
    matrix: &Value,
    decomp_type: DecompositionType,
) -> BuiltinResult<()> {
    if decomp_type == DecompositionType::Chol {
        ensure_chol_numeric_boundary(matrix)?;
    }
    let (rows, cols) = matrix_dims(matrix);
    match decomp_type {
        DecompositionType::Lu
        | DecompositionType::Chol
        | DecompositionType::Triangular
        | DecompositionType::Diagonal
            if rows != cols =>
        {
            Err(invalid(format!(
                "decomposition: type '{}' requires a square matrix",
                decomposition_type_name(decomp_type)
            )))
        }
        DecompositionType::Diagonal if !is_diagonal_matrix(matrix) => Err(invalid(
            "decomposition: type 'diagonal' requires a diagonal matrix",
        )),
        DecompositionType::Triangular if !is_triangular_matrix(matrix) => Err(invalid(
            "decomposition: type 'triangular' requires a triangular matrix",
        )),
        DecompositionType::Chol if !is_hermitian_matrix(matrix) => Err(invalid(
            "decomposition: type 'chol' requires a symmetric or Hermitian matrix",
        )),
        _ => Ok(()),
    }
}

pub(super) async fn validate_chol_positive_definite(matrix: &Value) -> BuiltinResult<()> {
    use crate::builtins::math::linalg::factor::cholesky::factor_upper;
    ensure_chol_numeric_boundary(matrix)?;
    let (order, _) = matrix_dims(matrix);
    let factor = match matrix {
        Value::Tensor(value) => {
            let entries = tensor::tensor_values_f64_cow(value);
            factor_upper(order, |row, col| {
                Complex64::new(entries[row + col * order], 0.0)
            })
        }
        Value::ComplexTensor(value) => {
            let entries = complex_tensor_values_pair_view(value);
            factor_upper(order, |row, col| {
                let (real, imaginary) = entries.get(row + col * order);
                Complex64::new(real, imaginary)
            })
        }
        Value::LogicalArray(value) => factor_upper(order, |row, col| {
            Complex64::new(f64::from(value.data[row + col * order]), 0.0)
        }),
        _ => {
            return Err(invalid(
                "decomposition: expected normalized coefficient matrix",
            ))
        }
    }
    .map_err(|_| internal("decomposition: unable to allocate Cholesky validation workspace"))?;
    if factor.info != 0 {
        return Err(invalid(
            "decomposition: type 'chol' requires a positive definite coefficient matrix",
        ));
    }
    Ok(())
}

/// Cholesky's floating mathematical boundary is narrower than the opaque
/// object's exact structural storage. Check before even the Hermitian probe
/// can materialize integer coefficients as binary64.
fn ensure_chol_numeric_boundary(matrix: &Value) -> BuiltinResult<()> {
    if crate::builtins::common::validation::is_typed_complex_integer(matrix)
        || ensure_exact_integer_scalar_boundary(matrix, "coefficient matrix").is_err()
    {
        // Preserve the constructor's former nested-Cholesky error mapping.
        return Err(invalid(
            "decomposition: type 'chol' requires a positive definite coefficient matrix",
        ));
    }
    Ok(())
}

pub(super) fn matrix_dims(matrix: &Value) -> (usize, usize) {
    match matrix {
        Value::Tensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::ComplexTensor(tensor) => (tensor.rows, tensor.cols),
        Value::LogicalArray(array) => (logical_rows(array), logical_cols(array)),
        _ => (1, 1),
    }
}

pub(super) fn matrix_shape_value(matrix: &Value) -> BuiltinResult<Value> {
    let (rows, cols) = matrix_dims(matrix);
    let tensor = Tensor::new(vec![rows as f64, cols as f64], vec![1, 2])
        .map_err(|e| internal(format!("decomposition: {e}")))?;
    Ok(Value::Tensor(tensor))
}

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
            for col in 0..tensor.cols {
                for row in 0..tensor.rows {
                    if row != col && values.get(row + col * tensor.rows) != (0.0, 0.0) {
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

pub(super) fn is_triangular_matrix(matrix: &Value) -> bool {
    is_upper_triangular(matrix) || is_lower_triangular(matrix)
}

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
            for col in 0..tensor.cols {
                for row in (col + 1)..tensor.rows {
                    if values.get(row + col * tensor.rows) != (0.0, 0.0) {
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
            for col in 0..tensor.cols {
                for row in 0..col.min(tensor.rows) {
                    if values.get(row + col * tensor.rows) != (0.0, 0.0) {
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

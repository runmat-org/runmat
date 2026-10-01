use super::*;

pub(super) fn real_rhs_matrix(
    tensor: &Tensor,
    rows: usize,
    rhs_cols: usize,
) -> BuiltinResult<DMatrix<f64>> {
    let values = tensor::tensor_values_f64_cow(tensor);
    if is_vector(tensor) && tensor::tensor_element_len(tensor) == rows {
        Ok(DMatrix::from_column_slice(rows, 1, values.as_ref()))
    } else if tensor.rows == rows && tensor.cols == rhs_cols {
        Ok(DMatrix::from_column_slice(rows, rhs_cols, values.as_ref()))
    } else {
        Err(invalid(
            "lscov: B must have one row per observation or be an observation vector",
        ))
    }
}

pub(super) fn complex_matrix(value: &NumericMatrix) -> BuiltinResult<DMatrix<Complex64>> {
    match value {
        NumericMatrix::Real(tensor) => {
            let values = tensor::tensor_values_f64(tensor)
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect::<Vec<_>>();
            Ok(DMatrix::from_column_slice(
                tensor.rows,
                tensor.cols,
                &values,
            ))
        }
        NumericMatrix::Complex(tensor) => Ok(DMatrix::from_column_slice(
            tensor.rows,
            tensor.cols,
            &tensor::complex_tensor_values_complex64(tensor),
        )),
    }
}

pub(super) fn complex_rhs_matrix(
    value: &NumericMatrix,
    rows: usize,
    rhs_cols: usize,
) -> BuiltinResult<DMatrix<Complex64>> {
    let matrix = complex_matrix(value)?;
    if matrix.nrows() == rows && matrix.ncols() == rhs_cols {
        return Ok(matrix);
    }
    let len = numeric_len(value);
    if len == rows && is_numeric_vector(value) {
        return Ok(DMatrix::from_column_slice(rows, 1, matrix.as_slice()));
    }
    Err(invalid(
        "lscov: B must have one row per observation or be an observation vector",
    ))
}

pub(super) fn rhs_columns(value: &NumericMatrix, rows: usize) -> BuiltinResult<usize> {
    match value {
        NumericMatrix::Real(tensor) => {
            rhs_columns_from_shape(tensor.rows, tensor.cols, tensor, rows)
        }
        NumericMatrix::Complex(tensor) => {
            if is_complex_vector(tensor) && tensor::complex_tensor_element_len(tensor) == rows {
                Ok(1)
            } else if tensor.rows == rows {
                Ok(tensor.cols)
            } else {
                Err(invalid(
                    "lscov: B must have one row per observation or be an observation vector",
                ))
            }
        }
    }
}

pub(super) fn rhs_columns_from_shape(
    tensor_rows: usize,
    tensor_cols: usize,
    tensor: &Tensor,
    rows: usize,
) -> BuiltinResult<usize> {
    if is_vector(tensor) && tensor::tensor_element_len(tensor) == rows {
        Ok(1)
    } else if tensor_rows == rows {
        Ok(tensor_cols)
    } else {
        Err(invalid(
            "lscov: B must have one row per observation or be an observation vector",
        ))
    }
}

pub(super) fn matrix_rows(value: &NumericMatrix) -> usize {
    match value {
        NumericMatrix::Real(tensor) => tensor.rows,
        NumericMatrix::Complex(tensor) => tensor.rows,
    }
}

pub(super) fn matrix_cols(value: &NumericMatrix) -> usize {
    match value {
        NumericMatrix::Real(tensor) => tensor.cols,
        NumericMatrix::Complex(tensor) => tensor.cols,
    }
}

pub(super) fn numeric_len(value: &NumericMatrix) -> usize {
    match value {
        NumericMatrix::Real(tensor) => tensor::tensor_element_len(tensor),
        NumericMatrix::Complex(tensor) => tensor::complex_tensor_element_len(tensor),
    }
}

pub(super) fn is_numeric_vector(value: &NumericMatrix) -> bool {
    match value {
        NumericMatrix::Real(tensor) => is_vector(tensor),
        NumericMatrix::Complex(tensor) => is_complex_vector(tensor),
    }
}

pub(super) fn ensure_2d(value: &NumericMatrix, label: &str) -> BuiltinResult<()> {
    let shape_len = match value {
        NumericMatrix::Real(tensor) => tensor.shape.len(),
        NumericMatrix::Complex(tensor) => tensor.shape.len(),
    };
    if shape_len > 2 {
        return Err(invalid(format!("lscov: {label} must be a 2-D matrix")));
    }
    Ok(())
}

pub(super) fn is_vector(tensor: &Tensor) -> bool {
    tensor.shape.len() <= 2 && (tensor.rows == 1 || tensor.cols == 1)
}

pub(super) fn is_complex_vector(tensor: &ComplexTensor) -> bool {
    tensor.shape.len() <= 2 && (tensor.rows == 1 || tensor.cols == 1)
}

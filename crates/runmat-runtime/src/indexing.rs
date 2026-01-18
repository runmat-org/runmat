//! Matrix indexing and slicing operations
//!
//! Implements language-style tensor indexing and access patterns.

use runmat_builtins::{Tensor, Value};
use crate::{build_runtime_error, RuntimeError};

fn indexing_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).build()
}

fn indexing_error_with_identifier(message: impl Into<String>, identifier: &str) -> RuntimeError {
    build_runtime_error(message).with_identifier(identifier).build()
}

/// Get a single element from a tensor (1-based indexing like language)
pub fn matrix_get_element(tensor: &Tensor, row: usize, col: usize) -> Result<f64, RuntimeError> {
    if row == 0 || col == 0 {
        return Err(indexing_error_with_identifier(
            "MATLAB uses 1-based indexing",
            "MATLAB:IndexOutOfBounds",
        ));
    }
    tensor
        .get2(row - 1, col - 1)
        .map_err(|err| indexing_error_with_identifier(err, "MATLAB:IndexOutOfBounds"))
}

/// Set a single element in a tensor (1-based indexing like language)
pub fn matrix_set_element(
    tensor: &mut Tensor,
    row: usize,
    col: usize,
    value: f64,
) -> Result<(), RuntimeError> {
    if row == 0 || col == 0 {
        return Err(indexing_error_with_identifier(
            "The MATLAB language uses 1-based indexing",
            "MATLAB:IndexOutOfBounds",
        ));
    }
    tensor
        .set2(row - 1, col - 1, value)
        .map_err(|err| indexing_error_with_identifier(err, "MATLAB:IndexOutOfBounds"))
}

/// Get a row from a tensor
pub fn matrix_get_row(tensor: &Tensor, row: usize) -> Result<Tensor, RuntimeError> {
    if row == 0 || row > tensor.rows() {
        return Err(indexing_error_with_identifier(
            format!(
                "Row index {} out of bounds for {}x{} tensor",
                row,
                tensor.rows(),
                tensor.cols()
            ),
            "MATLAB:IndexOutOfBounds",
        ));
    }

    // Column-major: row slice picks every element spaced by rows across columns
    let mut row_data = Vec::with_capacity(tensor.cols());
    for c in 0..tensor.cols() {
        row_data.push(tensor.data[(row - 1) + c * tensor.rows()]);
    }
    Tensor::new_2d(row_data, 1, tensor.cols())
        .map_err(|err| indexing_error(err))
}

/// Get a column from a tensor
pub fn matrix_get_col(tensor: &Tensor, col: usize) -> Result<Tensor, RuntimeError> {
    if col == 0 || col > tensor.cols() {
        return Err(indexing_error_with_identifier(
            format!(
                "Column index {} out of bounds for {}x{} tensor",
                col,
                tensor.rows(),
                tensor.cols()
            ),
            "MATLAB:IndexOutOfBounds",
        ));
    }

    let mut col_data = Vec::with_capacity(tensor.rows());
    for row in 0..tensor.rows() {
        col_data.push(tensor.data[row + (col - 1) * tensor.rows()]);
    }
    Tensor::new_2d(col_data, tensor.rows(), 1)
        .map_err(|err| indexing_error(err))
}

/// Array indexing operation (used by all interpreters/compilers)
/// In MATLAB, indexing is 1-based and supports:
/// - Single element: A(i) for vectors, A(i,j) for tensors
/// - Multiple indices: A(i1, i2, ..., iN)
pub fn perform_indexing(base: &Value, indices: &[f64]) -> Result<Value, RuntimeError> {
    match base {
        Value::GpuTensor(h) => {
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                indexing_error("Cannot index value of type GpuTensor without a provider")
            })?;
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }
            // Support scalar indexing cases mirroring Tensor branch
            if indices.len() == 1 {
                let idx = indices[0] as usize;
                let total = h.shape.iter().product();
                if idx < 1 || idx > total {
                    return Err(indexing_error_with_identifier(
                        format!("Index {} out of bounds (1 to {})", idx, total),
                        "MATLAB:IndexOutOfBounds",
                    ));
                }
                let lin0 = idx - 1; // 0-based
                let val = provider
                    .read_scalar(h, lin0)
                    .map_err(|e| indexing_error(format!("gpu index: {e}")))?;
                return Ok(Value::Num(val));
            } else if indices.len() == 2 {
                let row = indices[0] as usize;
                let col = indices[1] as usize;
                let rows = h.shape.first().copied().unwrap_or(1);
                let cols = h.shape.get(1).copied().unwrap_or(1);
                if row < 1 || row > rows || col < 1 || col > cols {
                    return Err(indexing_error_with_identifier(
                        format!(
                            "Index ({row}, {col}) out of bounds for {rows}x{cols} tensor"
                        ),
                        "MATLAB:IndexOutOfBounds",
                    ));
                }
                let lin0 = (row - 1) + (col - 1) * rows;
                let val = provider
                    .read_scalar(h, lin0)
                    .map_err(|e| indexing_error(format!("gpu index: {e}")))?;
                return Ok(Value::Num(val));
            }
            Err(indexing_error_with_identifier(
                format!("Cannot index value of type {base:?}"),
                "MATLAB:SliceNonTensor",
            ))
        }
        Value::Tensor(tensor) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }

            if indices.len() == 1 {
                // Linear indexing (1-based)
                let idx = indices[0] as usize;
                if idx < 1 || idx > tensor.data.len() {
                    return Err(indexing_error_with_identifier(
                        format!("Index {} out of bounds (1 to {})", idx, tensor.data.len()),
                        "MATLAB:IndexOutOfBounds",
                    ));
                }
                Ok(Value::Num(tensor.data[idx - 1])) // Convert to 0-based
            } else if indices.len() == 2 {
                // Row-column indexing (1-based)
                let row = indices[0] as usize;
                let col = indices[1] as usize;

                if row < 1 || row > tensor.rows {
                    return Err(indexing_error_with_identifier(
                        format!("Row index {} out of bounds (1 to {})", row, tensor.rows),
                        "MATLAB:IndexOutOfBounds",
                    ));
                }
                if col < 1 || col > tensor.cols {
                    return Err(indexing_error_with_identifier(
                        format!("Column index {} out of bounds (1 to {})", col, tensor.cols),
                        "MATLAB:IndexOutOfBounds",
                    ));
                }

                let linear_idx = (row - 1) + (col - 1) * tensor.rows; // Convert to 0-based, column-major
                Ok(Value::Num(tensor.data[linear_idx]))
            } else {
                Err(indexing_error(format!(
                    "Tensors support 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::StringArray(sa) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }
            if indices.len() == 1 {
                let idx = indices[0] as usize;
                let total = sa.data.len();
                if idx < 1 || idx > total {
                    return Err(indexing_error_with_identifier(
                        format!("Index {idx} out of bounds (1 to {total})"),
                        "MATLAB:IndexOutOfBounds",
                    ));
                }
                Ok(Value::String(sa.data[idx - 1].clone()))
            } else if indices.len() == 2 {
                let row = indices[0] as usize;
                let col = indices[1] as usize;
                if row < 1 || row > sa.rows || col < 1 || col > sa.cols {
                    return Err(indexing_error_with_identifier(
                        "StringArray subscript out of bounds",
                        "MATLAB:IndexOutOfBounds",
                    ));
                }
                let idx = (row - 1) + (col - 1) * sa.rows;
                Ok(Value::String(sa.data[idx].clone()))
            } else {
                Err(indexing_error(format!(
                    "StringArray supports 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::Num(_) | Value::Int(_) => {
            if indices.len() == 1 && indices[0] == 1.0 {
                // Scalar indexing with A(1) returns the scalar itself
                Ok(base.clone())
            } else {
                Err(indexing_error_with_identifier(
                    "Slicing only supported on tensors",
                    "MATLAB:SliceNonTensor",
                ))
            }
        }
        Value::Cell(ca) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }
            if indices.len() == 1 {
                let idx = indices[0] as usize;
                if idx < 1 || idx > ca.data.len() {
                    return Err(indexing_error_with_identifier(
                        format!("Cell index {} out of bounds (1 to {})", idx, ca.data.len()),
                        "MATLAB:CellIndexOutOfBounds",
                    ));
                }
                Ok((*ca.data[idx - 1]).clone())
            } else if indices.len() == 2 {
                let row = indices[0] as usize;
                let col = indices[1] as usize;
                if row < 1 || row > ca.rows || col < 1 || col > ca.cols {
                    return Err(indexing_error_with_identifier(
                        "Cell subscript out of bounds",
                        "MATLAB:CellSubscriptOutOfBounds",
                    ));
                }
                Ok((*ca.data[(row - 1) * ca.cols + (col - 1)]).clone())
            } else {
                Err(indexing_error(format!(
                    "Cell arrays support 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        _ => Err(indexing_error_with_identifier(
            format!("Cannot index value of type {base:?}"),
            "MATLAB:SliceNonTensor",
        )),
    }
}

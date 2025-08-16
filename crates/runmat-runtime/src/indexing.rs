//! Matrix indexing and slicing operations
//!
//! Implements MATLAB-style matrix indexing and access patterns.

use runmat_builtins::{Tensor, Value};

/// Get a single element from a matrix (1-based indexing like MATLAB)
pub fn matrix_get_element(matrix: &Tensor, row: usize, col: usize) -> Result<f64, String> {
    if row == 0 || col == 0 {
        return Err("MATLAB uses 1-based indexing".to_string());
    }
    matrix.get2(row - 1, col - 1) // Convert to 0-based
}

/// Set a single element in a matrix (1-based indexing like MATLAB)
pub fn matrix_set_element(
    matrix: &mut Tensor,
    row: usize,
    col: usize,
    value: f64,
) -> Result<(), String> {
    if row == 0 || col == 0 {
        return Err("MATLAB uses 1-based indexing".to_string());
    }
    matrix.set2(row - 1, col - 1, value) // Convert to 0-based
}

/// Get a row from a matrix
pub fn matrix_get_row(matrix: &Tensor, row: usize) -> Result<Tensor, String> {
    if row == 0 || row > matrix.rows() {
        return Err(format!(
            "Row index {} out of bounds for {}x{} matrix",
            row, matrix.rows(), matrix.cols()
        ));
    }

    // Column-major: row slice picks every element spaced by rows across columns
    let mut row_data = Vec::with_capacity(matrix.cols());
    for c in 0..matrix.cols() { row_data.push(matrix.data[(row - 1) + c * matrix.rows()]); }
    Tensor::new_2d(row_data, 1, matrix.cols())
}

/// Get a column from a matrix
pub fn matrix_get_col(matrix: &Tensor, col: usize) -> Result<Tensor, String> {
    if col == 0 || col > matrix.cols() {
        return Err(format!(
            "Column index {} out of bounds for {}x{} matrix",
            col, matrix.rows(), matrix.cols()
        ));
    }

    let mut col_data = Vec::with_capacity(matrix.rows());
    for row in 0..matrix.rows() { col_data.push(matrix.data[row + (col - 1) * matrix.rows()]); }
    Tensor::new_2d(col_data, matrix.rows(), 1)
}

/// Array indexing operation (used by all interpreters/compilers)
/// In MATLAB, indexing is 1-based and supports:
/// - Single element: A(i) for vectors, A(i,j) for matrices
/// - Multiple indices: A(i1, i2, ..., iN)
pub fn perform_indexing(base: &Value, indices: &[f64]) -> Result<Value, String> {
    match base {
        Value::Tensor(matrix) => {
            if indices.is_empty() {
                return Err("At least one index is required".to_string());
            }

            if indices.len() == 1 {
                // Linear indexing (1-based)
                let idx = indices[0] as usize;
                if idx < 1 || idx > matrix.data.len() {
                    return Err(format!(
                        "Index {} out of bounds (1 to {})",
                        idx,
                        matrix.data.len()
                    ));
                }
                Ok(Value::Num(matrix.data[idx - 1])) // Convert to 0-based
            } else if indices.len() == 2 {
                // Row-column indexing (1-based)
                let row = indices[0] as usize;
                let col = indices[1] as usize;

                if row < 1 || row > matrix.rows {
                    return Err(format!(
                        "Row index {} out of bounds (1 to {})",
                        row, matrix.rows
                    ));
                }
                if col < 1 || col > matrix.cols {
                    return Err(format!(
                        "Column index {} out of bounds (1 to {})",
                        col, matrix.cols
                    ));
                }

                let linear_idx = (row - 1) + (col - 1) * matrix.rows; // Convert to 0-based, column-major
                Ok(Value::Num(matrix.data[linear_idx]))
            } else {
                Err(format!(
                    "Matrices support 1 or 2 indices, got {}",
                    indices.len()
                ))
            }
        }
        Value::StringArray(sa) => {
            if indices.is_empty() { return Err("At least one index is required".to_string()); }
            if indices.len() == 1 {
                let idx = indices[0] as usize; let total = sa.data.len();
                if idx < 1 || idx > total { return Err(format!("Index {} out of bounds (1 to {})", idx, total)); }
                Ok(Value::String(sa.data[idx - 1].clone()))
            } else if indices.len() == 2 {
                let row = indices[0] as usize; let col = indices[1] as usize;
                if row < 1 || row > sa.rows || col < 1 || col > sa.cols { return Err("StringArray subscript out of bounds".to_string()); }
                let idx = (row - 1) + (col - 1) * sa.rows;
                Ok(Value::String(sa.data[idx].clone()))
            } else {
                Err(format!("StringArray supports 1 or 2 indices, got {}", indices.len()))
            }
        }
        Value::Num(_) | Value::Int(_) => {
            if indices.len() == 1 && indices[0] == 1.0 {
                // Scalar indexing with A(1) returns the scalar itself
                Ok(base.clone())
            } else {
                Err("MATLAB:SliceNonTensor: Slicing only supported on tensors".to_string())
            }
        }
        Value::Cell(ca) => {
            if indices.is_empty() {
                return Err("At least one index is required".to_string());
            }
            if indices.len() == 1 {
                let idx = indices[0] as usize;
                if idx < 1 || idx > ca.data.len() {
                    return Err(format!("Cell index {} out of bounds (1 to {})", idx, ca.data.len()));
                }
                Ok(ca.data[idx - 1].clone())
            } else if indices.len() == 2 {
                let row = indices[0] as usize;
                let col = indices[1] as usize;
                if row < 1 || row > ca.rows || col < 1 || col > ca.cols {
                    return Err("Cell subscript out of bounds".to_string());
                }
                Ok(ca.data[(row - 1) * ca.cols + (col - 1)].clone())
            } else {
                Err(format!("Cell arrays support 1 or 2 indices, got {}", indices.len()))
            }
        }
        _ => Err(format!("Cannot index value of type {base:?}")),
    }
}

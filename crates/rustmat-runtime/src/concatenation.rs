//! Matrix and array concatenation operations
//!
//! This module provides MATLAB-compatible matrix concatenation operations.
//! Supports both horizontal concatenation [A, B] and vertical concatenation [A; B].

use rustmat_builtins::{Matrix, Value};

/// Horizontally concatenate two matrices [A, B]
/// In MATLAB: C = [A, B] creates a matrix with A and B side by side
pub fn hcat_matrices(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows {
        return Err(format!(
            "Cannot horizontally concatenate matrices with different row counts: {} vs {}",
            a.rows, b.rows
        ));
    }

    let new_rows = a.rows;
    let new_cols = a.cols + b.cols;
    let mut new_data = Vec::with_capacity(new_rows * new_cols);

    // Copy row by row: each row gets elements from A then B
    for row in 0..new_rows {
        // Copy elements from matrix A for this row
        for col in 0..a.cols {
            new_data.push(a.data[row * a.cols + col]);
        }
        // Copy elements from matrix B for this row
        for col in 0..b.cols {
            new_data.push(b.data[row * b.cols + col]);
        }
    }

    Matrix::new(new_data, new_rows, new_cols)
}

/// Vertically concatenate two matrices [A; B]
/// In MATLAB: C = [A; B] creates a matrix with A on top and B below
pub fn vcat_matrices(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.cols != b.cols {
        return Err(format!(
            "Cannot vertically concatenate matrices with different column counts: {} vs {}",
            a.cols, b.cols
        ));
    }

    let new_rows = a.rows + b.rows;
    let new_cols = a.cols;
    let mut new_data = Vec::with_capacity(new_rows * new_cols);

    // Copy all elements from matrix A first
    new_data.extend_from_slice(&a.data);
    
    // Then copy all elements from matrix B
    new_data.extend_from_slice(&b.data);

    Matrix::new(new_data, new_rows, new_cols)
}

/// Concatenate values horizontally - handles mixed scalars and matrices
pub fn hcat_values(values: &[Value]) -> Result<Value, String> {
    if values.is_empty() {
        return Ok(Value::Matrix(Matrix::new(vec![], 0, 0)?));
    }

    // Convert all scalars to 1x1 matrices for uniform processing
    let mut matrices = Vec::new();
    let mut _total_cols = 0;
    let mut rows = 0;

    for value in values {
        match value {
            Value::Num(n) => {
                let matrix = Matrix::new(vec![*n], 1, 1)?;
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err("Cannot concatenate scalar with multi-row matrix".to_string());
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Int(i) => {
                let matrix = Matrix::new(vec![*i as f64], 1, 1)?;
                if rows == 0 {
                    rows = 1;
                } else if rows != 1 {
                    return Err("Cannot concatenate scalar with multi-row matrix".to_string());
                }
                _total_cols += 1;
                matrices.push(matrix);
            }
            Value::Matrix(m) => {
                if rows == 0 {
                    rows = m.rows;
                } else if rows != m.rows {
                    return Err(format!(
                        "Cannot concatenate matrices with different row counts: {} vs {}",
                        rows, m.rows
                    ));
                }
                _total_cols += m.cols;
                matrices.push(m.clone());
            }
            _ => return Err(format!("Cannot concatenate value of type {:?}", value)),
        }
    }

    // Now concatenate all matrices horizontally
    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = hcat_matrices(&result, matrix)?;
    }

    Ok(Value::Matrix(result))
}

/// Concatenate values vertically - handles mixed scalars and matrices
pub fn vcat_values(values: &[Value]) -> Result<Value, String> {
    if values.is_empty() {
        return Ok(Value::Matrix(Matrix::new(vec![], 0, 0)?));
    }

    // Convert all scalars to 1x1 matrices for uniform processing
    let mut matrices = Vec::new();
    let mut _total_rows = 0;
    let mut cols = 0;

    for value in values {
        match value {
            Value::Num(n) => {
                let matrix = Matrix::new(vec![*n], 1, 1)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err("Cannot concatenate scalar with multi-column matrix".to_string());
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Int(i) => {
                let matrix = Matrix::new(vec![*i as f64], 1, 1)?;
                if cols == 0 {
                    cols = 1;
                } else if cols != 1 {
                    return Err("Cannot concatenate scalar with multi-column matrix".to_string());
                }
                _total_rows += 1;
                matrices.push(matrix);
            }
            Value::Matrix(m) => {
                if cols == 0 {
                    cols = m.cols;
                } else if cols != m.cols {
                    return Err(format!(
                        "Cannot concatenate matrices with different column counts: {} vs {}",
                        cols, m.cols
                    ));
                }
                _total_rows += m.rows;
                matrices.push(m.clone());
            }
            _ => return Err(format!("Cannot concatenate value of type {:?}", value)),
        }
    }

    // Now concatenate all matrices vertically
    let mut result = matrices[0].clone();
    for matrix in &matrices[1..] {
        result = vcat_matrices(&result, matrix)?;
    }

    Ok(Value::Matrix(result))
}

/// Create a matrix from a 2D array of Values with proper concatenation semantics
/// This handles the case where matrix elements can be variables, not just literals
pub fn create_matrix_from_values(rows: &[Vec<Value>]) -> Result<Value, String> {
    if rows.is_empty() {
        return Ok(Value::Matrix(Matrix::new(vec![], 0, 0)?));
    }

    // First, concatenate each row horizontally
    let mut row_matrices = Vec::new();
    for row in rows {
        let row_result = hcat_values(row)?;
        row_matrices.push(row_result);
    }

    // Then concatenate all rows vertically
    match row_matrices.len() {
        0 => Ok(Value::Matrix(Matrix::new(vec![], 0, 0)?)),
        1 => Ok(row_matrices.into_iter().next().unwrap()),
        _ => vcat_values(&row_matrices),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcat_matrices() {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix::new(vec![5.0, 6.0], 2, 1).unwrap();
        
        let result = hcat_matrices(&a, &b).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert_eq!(result.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vcat_matrices() {
        let a = Matrix::new(vec![1.0, 2.0], 1, 2).unwrap();
        let b = Matrix::new(vec![3.0, 4.0], 1, 2).unwrap();
        
        let result = vcat_matrices(&a, &b).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_hcat_values_scalars() {
        let values = vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)];
        let result = hcat_values(&values).unwrap();
        
        if let Value::Matrix(m) = result {
            assert_eq!(m.rows, 1);
            assert_eq!(m.cols, 3);
            assert_eq!(m.data, vec![1.0, 2.0, 3.0]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn test_vcat_values_scalars() {
        let values = vec![Value::Num(1.0), Value::Num(2.0)];
        let result = vcat_values(&values).unwrap();
        
        if let Value::Matrix(m) = result {
            assert_eq!(m.rows, 2);
            assert_eq!(m.cols, 1);
            assert_eq!(m.data, vec![1.0, 2.0]);
        } else {
            panic!("Expected matrix result");
        }
    }
}
//! Matrix indexing and slicing operations
//! 
//! Implements MATLAB-style matrix indexing and access patterns.

use rustmat_builtins::Matrix;

/// Get a single element from a matrix (1-based indexing like MATLAB)
pub fn matrix_get_element(matrix: &Matrix, row: usize, col: usize) -> Result<f64, String> {
    if row == 0 || col == 0 {
        return Err("MATLAB uses 1-based indexing".to_string());
    }
    matrix.get(row - 1, col - 1)  // Convert to 0-based
}

/// Set a single element in a matrix (1-based indexing like MATLAB)
pub fn matrix_set_element(matrix: &mut Matrix, row: usize, col: usize, value: f64) -> Result<(), String> {
    if row == 0 || col == 0 {
        return Err("MATLAB uses 1-based indexing".to_string());
    }
    matrix.set(row - 1, col - 1, value)  // Convert to 0-based
}

/// Get a row from a matrix
pub fn matrix_get_row(matrix: &Matrix, row: usize) -> Result<Matrix, String> {
    if row == 0 || row > matrix.rows {
        return Err(format!("Row index {} out of bounds for {}x{} matrix", row, matrix.rows, matrix.cols));
    }
    
    let start_idx = (row - 1) * matrix.cols;  // Convert to 0-based
    let row_data = matrix.data[start_idx..start_idx + matrix.cols].to_vec();
    Matrix::new(row_data, 1, matrix.cols)
}

/// Get a column from a matrix
pub fn matrix_get_col(matrix: &Matrix, col: usize) -> Result<Matrix, String> {
    if col == 0 || col > matrix.cols {
        return Err(format!("Column index {} out of bounds for {}x{} matrix", col, matrix.rows, matrix.cols));
    }
    
    let mut col_data = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        col_data.push(matrix.data[row * matrix.cols + (col - 1)]);  // Convert to 0-based
    }
    Matrix::new(col_data, matrix.rows, 1)
} 
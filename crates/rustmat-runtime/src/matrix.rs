//! Matrix operations for MATLAB-compatible arithmetic
//!
//! Implements element-wise and matrix operations following MATLAB semantics.

use rustmat_builtins::Matrix;
use rustmat_macros::runtime_builtin;

/// Matrix addition: C = A + B
pub fn matrix_add(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} + {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Matrix subtraction: C = A - B
pub fn matrix_sub(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} - {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x - y)
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Matrix multiplication: C = A * B
pub fn matrix_mul(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.cols != b.rows {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let rows = a.rows;
    let cols = b.cols;
    let mut data = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            data[i * cols + j] = sum;
        }
    }

    Matrix::new(data, rows, cols)
}

/// Scalar multiplication: C = A * s
pub fn matrix_scalar_mul(a: &Matrix, scalar: f64) -> Matrix {
    let data: Vec<f64> = a.data.iter().map(|x| x * scalar).collect();
    Matrix::new(data, a.rows, a.cols).unwrap() // Always valid
}

/// Matrix transpose: C = A'
pub fn matrix_transpose(a: &Matrix) -> Matrix {
    let mut data = vec![0.0; a.rows * a.cols];
    for i in 0..a.rows {
        for j in 0..a.cols {
            data[j * a.rows + i] = a.data[i * a.cols + j];
        }
    }
    Matrix::new(data, a.cols, a.rows).unwrap() // Always valid
}

/// Create identity matrix
pub fn matrix_eye(n: usize) -> Matrix {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Matrix::new(data, n, n).unwrap() // Always valid
}

// Simple built-in function for testing matrix operations
#[runtime_builtin(name = "matrix_zeros")]
fn matrix_zeros_builtin(rows: i32, cols: i32) -> Result<Matrix, String> {
    if rows < 0 || cols < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Matrix::zeros(rows as usize, cols as usize))
}

#[runtime_builtin(name = "matrix_ones")]
fn matrix_ones_builtin(rows: i32, cols: i32) -> Result<Matrix, String> {
    if rows < 0 || cols < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Matrix::ones(rows as usize, cols as usize))
}

#[runtime_builtin(name = "matrix_eye")]
fn matrix_eye_builtin(n: i32) -> Result<Matrix, String> {
    if n < 0 {
        return Err("Matrix size must be non-negative".to_string());
    }
    Ok(matrix_eye(n as usize))
}

#[runtime_builtin(name = "matrix_transpose")]
fn matrix_transpose_builtin(a: Matrix) -> Result<Matrix, String> {
    Ok(matrix_transpose(&a))
}

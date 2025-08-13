//! Matrix operations for MATLAB-compatible arithmetic
//!
//! Implements element-wise and matrix operations following MATLAB semantics.

use runmat_builtins::Tensor;
use runmat_macros::runtime_builtin;

/// Matrix addition: C = A + B
pub fn matrix_add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
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

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Matrix subtraction: C = A - B
pub fn matrix_sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
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

    Tensor::new_2d(data, a.rows(), a.cols())
}

/// Matrix multiplication: C = A * B
pub fn matrix_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.cols() != b.rows() {
        return Err(format!(
            "Inner matrix dimensions must agree: {}x{} * {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let rows = a.rows();
    let cols = b.cols();
    let mut data = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..a.cols() {
                // Column-major: A(i,k) at i + k*rows, B(k,j) at k + j*rows_b
                sum += a.data[i + k * rows] * b.data[k + j * b.rows()];
            }
            // C(i,j) at i + j*rows
            data[i + j * rows] = sum;
        }
    }

    Tensor::new_2d(data, rows, cols)
}

/// Scalar multiplication: C = A * s
pub fn matrix_scalar_mul(a: &Tensor, scalar: f64) -> Tensor {
    let data: Vec<f64> = a.data.iter().map(|x| x * scalar).collect();
    Tensor::new_2d(data, a.rows(), a.cols()).unwrap() // Always valid
}

/// Matrix transpose: C = A'
pub fn matrix_transpose(a: &Tensor) -> Tensor {
    let mut data = vec![0.0; a.rows() * a.cols()];
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            // dst(j,i) = src(i,j)
            data[j * a.rows() + i] = a.data[i + j * a.rows()];
        }
    }
    Tensor::new_2d(data, a.cols(), a.rows()).unwrap() // Always valid
}

/// Matrix power: C = A^n (for positive integer n)
/// This computes A * A * ... * A (n times) via repeated multiplication
pub fn matrix_power(a: &Tensor, n: i32) -> Result<Tensor, String> {
    if a.rows() != a.cols() {
        return Err(format!(
            "Matrix must be square for matrix power: {}x{}",
            a.rows(), a.cols()
        ));
    }

    if n < 0 {
        return Err("Negative matrix powers not supported yet".to_string());
    }

    if n == 0 {
        // A^0 = I (identity matrix)
        return Ok(matrix_eye(a.rows));
    }

    if n == 1 {
        // A^1 = A
        return Ok(a.clone());
    }

    // Compute A^n via repeated multiplication
    // Use binary exponentiation for efficiency
    let mut result = matrix_eye(a.rows());
    let mut base = a.clone();
    let mut exp = n as u32;

    while exp > 0 {
        if exp % 2 == 1 {
            result = matrix_mul(&result, &base)?;
        }
        base = matrix_mul(&base, &base)?;
        exp /= 2;
    }

    Ok(result)
}

/// Create identity matrix
pub fn matrix_eye(n: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Tensor::new_2d(data, n, n).unwrap() // Always valid
}

// Simple built-in function for testing matrix operations
#[runtime_builtin(name = "matrix_zeros")]
fn matrix_zeros_builtin(rows: i32, cols: i32) -> Result<Tensor, String> {
    if rows < 0 || cols < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Tensor::zeros(vec![rows as usize, cols as usize]))
}

#[runtime_builtin(name = "matrix_ones")]
fn matrix_ones_builtin(rows: i32, cols: i32) -> Result<Tensor, String> {
    if rows < 0 || cols < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Tensor::ones(vec![rows as usize, cols as usize]))
}

#[runtime_builtin(name = "matrix_eye")]
fn matrix_eye_builtin(n: i32) -> Result<Tensor, String> {
    if n < 0 {
        return Err("Matrix size must be non-negative".to_string());
    }
    Ok(matrix_eye(n as usize))
}

#[runtime_builtin(name = "matrix_transpose")]
fn matrix_transpose_builtin(a: Tensor) -> Result<Tensor, String> {
    Ok(matrix_transpose(&a))
}

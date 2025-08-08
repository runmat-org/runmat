//! Array generation functions
//!
//! This module provides MATLAB-compatible array generation functions like linspace, logspace,
//! zeros, ones, eye, etc. These functions are optimized for performance and memory efficiency.

use rustmat_builtins::{Matrix, Value};
use rustmat_macros::runtime_builtin;

/// Generate linearly spaced vector
/// linspace(x1, x2, n) generates n points between x1 and x2 (inclusive)
#[runtime_builtin(name = "linspace")]
fn linspace_builtin(x1: f64, x2: f64, n: i32) -> Result<Matrix, String> {
    if n < 1 {
        return Err("Number of points must be positive".to_string());
    }

    if n == 1 {
        return Matrix::new(vec![x2], 1, 1);
    }

    let n_usize = n as usize;
    let mut data = Vec::with_capacity(n_usize);

    if n == 2 {
        data.push(x1);
        data.push(x2);
    } else {
        let step = (x2 - x1) / ((n - 1) as f64);
        for i in 0..n_usize {
            data.push(x1 + (i as f64) * step);
        }
        // Ensure the last point is exactly x2 to avoid floating point errors
        data[n_usize - 1] = x2;
    }

    Matrix::new(data, 1, n_usize)
}

/// Generate logarithmically spaced vector
/// logspace(a, b, n) generates n points between 10^a and 10^b
#[runtime_builtin(name = "logspace")]
fn logspace_builtin(a: f64, b: f64, n: i32) -> Result<Matrix, String> {
    if n < 1 {
        return Err("Number of points must be positive".to_string());
    }

    let n_usize = n as usize;
    let mut data = Vec::with_capacity(n_usize);

    if n == 1 {
        data.push(10.0_f64.powf(b));
    } else {
        let log_step = (b - a) / ((n - 1) as f64);
        for i in 0..n_usize {
            let log_val = a + (i as f64) * log_step;
            data.push(10.0_f64.powf(log_val));
        }
    }

    Matrix::new(data, 1, n_usize)
}

/// Generate zeros matrix
/// zeros(m, n) creates an m×n matrix of zeros
#[runtime_builtin(name = "zeros")]
fn zeros_builtin(m: i32, n: i32) -> Result<Matrix, String> {
    if m < 0 || n < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Matrix::zeros(m as usize, n as usize))
}

/// Generate ones matrix
/// ones(m, n) creates an m×n matrix of ones
#[runtime_builtin(name = "ones")]
fn ones_builtin(m: i32, n: i32) -> Result<Matrix, String> {
    if m < 0 || n < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }
    Ok(Matrix::ones(m as usize, n as usize))
}

/// Generate identity matrix
/// eye(n) creates an n×n identity matrix
#[runtime_builtin(name = "eye")]
fn eye_builtin(n: i32) -> Result<Matrix, String> {
    if n < 0 {
        return Err("Matrix size must be non-negative".to_string());
    }
    let n_usize = n as usize;
    let mut data = vec![0.0; n_usize * n_usize];

    // Set diagonal elements to 1
    for i in 0..n_usize {
        data[i * n_usize + i] = 1.0;
    }

    Matrix::new(data, n_usize, n_usize)
}

/// Generate random matrix with uniform distribution [0,1)
/// rand(m, n) creates an m×n matrix of random numbers
#[runtime_builtin(name = "rand")]
fn rand_builtin(m: i32, n: i32) -> Result<Matrix, String> {
    if m < 0 || n < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }

    let rows = m as usize;
    let cols = n as usize;
    let total_elements = rows * cols;

    // Use a simple linear congruential generator for reproducible results
    // This is not cryptographically secure but suitable for basic mathematical operations
    static mut SEED: u64 = 1;
    let mut data = Vec::with_capacity(total_elements);

    unsafe {
        for _ in 0..total_elements {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = ((SEED >> 16) & 0x7fff) as f64 / 32768.0;
            data.push(random_val);
        }
    }

    Matrix::new(data, rows, cols)
}

/// Generate matrix filled with specific value
/// fill(value, m, n) creates an m×n matrix filled with value
#[runtime_builtin(name = "fill")]
fn fill_builtin(value: f64, m: i32, n: i32) -> Result<Matrix, String> {
    if m < 0 || n < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }

    let rows = m as usize;
    let cols = n as usize;
    let data = vec![value; rows * cols];

    Matrix::new(data, rows, cols)
}

/// Generate random matrix with normal distribution (mean=0, std=1)
/// randn(m, n) creates an m×n matrix of normally distributed random numbers
#[runtime_builtin(name = "randn")]
fn randn_builtin(m: i32, n: i32) -> Result<Matrix, String> {
    if m < 0 || n < 0 {
        return Err("Matrix dimensions must be non-negative".to_string());
    }

    let rows = m as usize;
    let cols = n as usize;
    let total_elements = rows * cols;

    // Simple approximation of normal distribution using central limit theorem
    // Generate 12 uniform random numbers and sum them, then subtract 6
    // This approximates a normal distribution with mean=0, std=1
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static SEED: OnceLock<Mutex<u64>> = OnceLock::new();
    let seed_mutex = SEED.get_or_init(|| Mutex::new(1));

    let mut data = Vec::with_capacity(total_elements);

    for _ in 0..total_elements {
        let mut seed_guard = seed_mutex
            .lock()
            .map_err(|_| "Failed to acquire RNG lock")?;
        let mut sum = 0.0;

        // Generate 12 uniform random numbers using linear congruential generator
        for _ in 0..12 {
            *seed_guard = seed_guard.wrapping_mul(1103515245).wrapping_add(12345);
            let uniform = ((*seed_guard >> 16) & 0x7fff) as f64 / 32768.0;
            sum += uniform;
        }

        // Apply central limit theorem transformation: N(0,1) ≈ sum(12 uniform) - 6
        let normal_val = sum - 6.0;
        data.push(normal_val);
    }

    Matrix::new(data, rows, cols)
}

/// Get the length of the largest dimension of a matrix
/// length(X) returns the size of the largest dimension of matrix X
#[runtime_builtin(name = "length")]
fn length_builtin(matrix: Matrix) -> Result<f64, String> {
    Ok(matrix.rows.max(matrix.cols) as f64)
}

/// Generate range vector
/// range(start, step, stop) creates a vector from start to stop with given step
#[runtime_builtin(name = "range")]
fn range_builtin(start: f64, step: f64, stop: f64) -> Result<Matrix, String> {
    if step == 0.0 {
        return Err("Step size cannot be zero".to_string());
    }

    if (step > 0.0 && start > stop) || (step < 0.0 && start < stop) {
        // Empty range
        return Matrix::new(vec![], 1, 0);
    }

    let mut data = Vec::new();
    let mut current = start;

    if step > 0.0 {
        while current <= stop + f64::EPSILON {
            data.push(current);
            current += step;
        }
    } else {
        while current >= stop - f64::EPSILON {
            data.push(current);
            current += step;
        }
    }

    let len = data.len();
    Matrix::new(data, 1, len)
}

/// Generate meshgrid for 2D plotting
/// [X, Y] = meshgrid(x, y) creates 2D coordinate arrays
#[runtime_builtin(name = "meshgrid")]
fn meshgrid_builtin(x: Matrix, y: Matrix) -> Result<Matrix, String> {
    // For simplicity, return flattened coordinate matrices
    // In a full implementation, this would return two separate matrices
    if x.rows != 1 && x.cols != 1 {
        return Err("Input x must be a vector".to_string());
    }
    if y.rows != 1 && y.cols != 1 {
        return Err("Input y must be a vector".to_string());
    }

    let x_vec = &x.data;
    let y_vec = &y.data;
    let nx = x_vec.len();
    let ny = y_vec.len();

    // Create X matrix (repeated rows)
    let mut x_data = Vec::with_capacity(nx * ny);
    for _ in 0..ny {
        x_data.extend_from_slice(x_vec);
    }

    Matrix::new(x_data, ny, nx)
}

/// Create a range vector (equivalent to start:end or start:step:end in MATLAB)
pub fn create_range(start: f64, step: Option<f64>, end: f64) -> Result<Value, String> {
    let step = step.unwrap_or(1.0);

    if step == 0.0 {
        return Err("Range step cannot be zero".to_string());
    }

    let mut values = Vec::new();

    if step > 0.0 {
        let mut current = start;
        while current <= end + f64::EPSILON {
            values.push(current);
            current += step;
        }
    } else {
        let mut current = start;
        while current >= end - f64::EPSILON {
            values.push(current);
            current += step;
        }
    }

    if values.is_empty() {
        // Return empty matrix for invalid ranges
        return Ok(Value::Matrix(Matrix::new(vec![], 0, 0)?));
    }

    // Create a row vector (1 x n)
    let cols = values.len();
    let matrix = Matrix::new(values, 1, cols)?;
    Ok(Value::Matrix(matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace() {
        let result = linspace_builtin(0.0, 10.0, 11).unwrap();
        assert_eq!(result.cols, 11);
        assert_eq!(result.rows, 1);
        assert!((result.data[0] - 0.0).abs() < f64::EPSILON);
        assert!((result.data[10] - 10.0).abs() < f64::EPSILON);
        assert!((result.data[5] - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linspace_single_point() {
        let result = linspace_builtin(5.0, 10.0, 1).unwrap();
        assert_eq!(result.cols, 1);
        assert_eq!(result.data[0], 10.0);
    }

    #[test]
    fn test_logspace() {
        let result = logspace_builtin(1.0, 3.0, 3).unwrap();
        assert_eq!(result.cols, 3);
        assert!((result.data[0] - 10.0).abs() < f64::EPSILON);
        assert!((result.data[1] - 100.0).abs() < f64::EPSILON);
        assert!((result.data[2] - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zeros() {
        let result = zeros_builtin(2, 3).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 3);
        assert!(result.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let result = ones_builtin(2, 2).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert!(result.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_eye() {
        let result = eye_builtin(3).unwrap();
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 3);
        // Check diagonal elements
        assert_eq!(result.data[0], 1.0); // (0,0)
        assert_eq!(result.data[4], 1.0); // (1,1)
        assert_eq!(result.data[8], 1.0); // (2,2)
                                         // Check off-diagonal elements
        assert_eq!(result.data[1], 0.0); // (0,1)
        assert_eq!(result.data[3], 0.0); // (1,0)
    }

    #[test]
    fn test_fill() {
        let result = fill_builtin(std::f64::consts::PI, 2, 2).unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert!(result
            .data
            .iter()
            .all(|&x| (x - std::f64::consts::PI).abs() < f64::EPSILON));
    }

    #[test]
    fn test_range() {
        let result = range_builtin(1.0, 1.0, 5.0).unwrap();
        assert_eq!(result.cols, 5);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_range_negative_step() {
        let result = range_builtin(5.0, -1.0, 1.0).unwrap();
        assert_eq!(result.cols, 5);
        assert_eq!(result.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_rand_dimensions() {
        let result = rand_builtin(3, 4).unwrap();
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 4);
        assert_eq!(result.data.len(), 12);
        // Check that values are in [0, 1)
        assert!(result.data.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_error_cases() {
        assert!(linspace_builtin(0.0, 1.0, -1).is_err());
        assert!(zeros_builtin(-1, 5).is_err());
        assert!(eye_builtin(-1).is_err());
        assert!(range_builtin(1.0, 0.0, 5.0).is_err());
    }
}

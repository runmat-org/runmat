//! Array generation functions
//!
//! This module provides array generation functions like linspace, logspace,
//! zeros, ones, eye, etc. These functions are optimized for performance and memory efficiency.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

/// Generate meshgrid for 2D plotting
/// [X, Y] = meshgrid(x, y) creates 2D coordinate arrays
#[runtime_builtin(name = "meshgrid")]
fn meshgrid_builtin(x: Tensor, y: Tensor) -> Result<Tensor, String> {
    // For simplicity, return flattened coordinate matrices
    // In a full implementation, this would return two separate matrices
    if x.rows() != 1 && x.cols() != 1 {
        return Err("Input x must be a vector".to_string());
    }
    if y.rows() != 1 && y.cols() != 1 {
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

    Tensor::new_2d(x_data, ny, nx)
}

/// Create a range vector (equivalent to start:end or start:step:end)
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
        return Ok(Value::Tensor(Tensor::new(vec![], vec![0, 0])?));
    }

    // Create a row vector (1 x n)
    let cols = values.len();
    let matrix = Tensor::new_2d(values, 1, cols)?;
    Ok(Value::Tensor(matrix))
}
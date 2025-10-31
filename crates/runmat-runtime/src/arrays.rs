//! Array generation functions
//!
//! This module provides array generation functions like linspace, logspace,
//! zeros, ones, eye, etc. These functions are optimized for performance and memory efficiency.

use runmat_builtins::{Tensor, Value};

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

//! Comparison operations for MATLAB-compatible logic
//!
//! Implements comparison operators returning logical matrices/values.

use rustmat_builtins::Matrix;
use rustmat_macros::runtime_builtin;

/// Element-wise greater than comparison
pub fn matrix_gt(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} > {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Element-wise greater than or equal comparison
pub fn matrix_ge(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} >= {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x >= y { 1.0 } else { 0.0 })
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Element-wise less than comparison
pub fn matrix_lt(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} < {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Element-wise less than or equal comparison
pub fn matrix_le(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} <= {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| if x <= y { 1.0 } else { 0.0 })
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Element-wise equality comparison
pub fn matrix_eq(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} == {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| {
            if (x - y).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

/// Element-wise inequality comparison
pub fn matrix_ne(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(format!(
            "Matrix dimensions must agree: {}x{} != {}x{}",
            a.rows, a.cols, b.rows, b.cols
        ));
    }

    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| {
            if (x - y).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    Matrix::new(data, a.rows, a.cols)
}

// Built-in comparison functions
#[runtime_builtin(name = "gt")]
fn gt_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a > b { 1.0 } else { 0.0 })
}

#[runtime_builtin(name = "ge")]
fn ge_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a >= b { 1.0 } else { 0.0 })
}

#[runtime_builtin(name = "lt")]
fn lt_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a < b { 1.0 } else { 0.0 })
}

#[runtime_builtin(name = "le")]
fn le_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if a <= b { 1.0 } else { 0.0 })
}

#[runtime_builtin(name = "eq")]
fn eq_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if (a - b).abs() < f64::EPSILON {
        1.0
    } else {
        0.0
    })
}

#[runtime_builtin(name = "ne")]
fn ne_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(if (a - b).abs() >= f64::EPSILON {
        1.0
    } else {
        0.0
    })
}

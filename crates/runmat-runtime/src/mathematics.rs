//! Mathematical functions
//!
//! This module provides comprehensive mathematical functions including trigonometric,
//! logarithmic, exponential, hyperbolic, and other mathematical operations.
//! All functions are optimized for performance and handle both scalars and matrices.

use runmat_builtins::Tensor;

// Logarithmic and exponential functions

#[runmat_macros::runtime_builtin(name = "ln")]
fn ln_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for ln".to_string())
    } else {
        Ok(x.ln())
    }
}

#[runmat_macros::runtime_builtin(name = "log10")]
fn log10_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for log10".to_string())
    } else {
        Ok(x.log10())
    }
}

#[runmat_macros::runtime_builtin(name = "exp2")]
fn exp2_builtin(x: f64) -> Result<f64, String> {
    Ok(2.0_f64.powf(x))
}

#[runmat_macros::runtime_builtin(name = "exp10")]
fn exp10_builtin(x: f64) -> Result<f64, String> {
    Ok(10.0_f64.powf(x))
}

#[runmat_macros::runtime_builtin(name = "pow")]
fn pow_builtin(base: f64, exponent: f64) -> Result<f64, String> {
    Ok(base.powf(exponent))
}

// Basic math functions

#[runmat_macros::runtime_builtin(name = "factorial")]
fn factorial_builtin(n: i32) -> Result<f64, String> {
    if n < 0 {
        return Err("Factorial undefined for negative numbers".to_string());
    }

    if n > 170 {
        return Err("Factorial overflow for n > 170".to_string());
    }

    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }

    Ok(result)
}

// Statistical functions

#[cfg(test)]
#[allow(dead_code)]
fn sum_builtin(matrix: Tensor) -> Result<f64, String> {
    Ok(matrix.data.iter().sum())
}

#[cfg(test)]
#[allow(dead_code)]
fn mean_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.is_empty() {
        return Err("Cannot compute mean of empty matrix".to_string());
    }
    Ok(matrix.data.iter().sum::<f64>() / matrix.data.len() as f64)
}

#[runmat_macros::runtime_builtin(name = "std")]
fn std_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute standard deviation".to_string());
    }

    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (matrix.data.len() - 1) as f64;

    Ok(variance.sqrt())
}

#[runmat_macros::runtime_builtin(name = "var")]
fn var_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute variance".to_string());
    }

    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (matrix.data.len() - 1) as f64;

    Ok(variance)
}

// Unit tests for mathematics live under crates/runmat-runtime/tests/mathematics.rs

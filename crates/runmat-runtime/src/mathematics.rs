//! Mathematical functions
//!
//! This module provides comprehensive mathematical functions including trigonometric,
//! logarithmic, exponential, hyperbolic, and other mathematical operations.
//! All functions are optimized for performance and handle both scalars and matrices.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

// Logarithmic and exponential functions

#[runtime_builtin(name = "ln")]
fn ln_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for ln".to_string())
    } else {
        Ok(x.ln())
    }
}

#[runtime_builtin(name = "log10")]
fn log10_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Input must be positive for log10".to_string())
    } else {
        Ok(x.log10())
    }
}

#[runtime_builtin(name = "exp2")]
fn exp2_builtin(x: f64) -> Result<f64, String> {
    Ok(2.0_f64.powf(x))
}

#[runtime_builtin(name = "exp10")]
fn exp10_builtin(x: f64) -> Result<f64, String> {
    Ok(10.0_f64.powf(x))
}

#[runtime_builtin(name = "pow")]
fn pow_builtin(base: f64, exponent: f64) -> Result<f64, String> {
    Ok(base.powf(exponent))
}

// Basic math functions
#[runtime_builtin(name = "abs", accel = "unary")]
fn abs_runtime_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.unary_abs(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("abs: unsupported for gpuArray".to_string())
        }
        Value::Num(n) => Ok(Value::Num(n.abs())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().abs())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.abs()).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, t.rows(), t.cols())?))
        }
        Value::Complex(re, im) => Ok(Value::Num((re * re + im * im).sqrt())),
        Value::ComplexTensor(ct) => {
            let data: Vec<f64> = ct
                .data
                .iter()
                .map(|(re, im)| (re * re + im * im).sqrt())
                .collect();
            Ok(Value::Tensor(Tensor::new_2d(data, ct.rows, ct.cols)?))
        }
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let (rows, cols) = match la.shape.len() {
                0 => (0, 0),
                1 => (1, la.shape[0]),
                _ => (la.shape[0], la.shape[1]),
            };
            Ok(Value::Tensor(Tensor::new_2d(data, rows, cols)?))
        }
        other => Err(format!("abs: unsupported input {other:?}")),
    }
}

#[runtime_builtin(name = "trunc")]
fn trunc_builtin(x: f64) -> Result<f64, String> {
    Ok(x.trunc())
}

#[runtime_builtin(name = "fract")]
fn fract_builtin(x: f64) -> Result<f64, String> {
    Ok(x.fract())
}

#[runtime_builtin(name = "sign")]
fn sign_builtin(x: f64) -> Result<f64, String> {
    if x > 0.0 {
        Ok(1.0)
    } else if x < 0.0 {
        Ok(-1.0)
    } else {
        Ok(0.0)
    }
}

// Complex number support (basic)

#[runtime_builtin(name = "real")]
fn real_builtin(x: f64) -> Result<f64, String> {
    Ok(x) // For real numbers, real part is the number itself
}

#[runtime_builtin(name = "imag")]
fn imag_builtin(_x: f64) -> Result<f64, String> {
    Ok(0.0) // For real numbers, imaginary part is always 0
}

#[runtime_builtin(name = "angle")]
fn angle_builtin(x: f64) -> Result<f64, String> {
    if x >= 0.0 {
        Ok(0.0)
    } else {
        Ok(std::f64::consts::PI)
    }
}

#[runtime_builtin(name = "factorial")]
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

#[runtime_builtin(name = "std")]
fn std_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute standard deviation".to_string());
    }

    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (matrix.data.len() - 1) as f64;

    Ok(variance.sqrt())
}

#[runtime_builtin(name = "var")]
fn var_builtin(matrix: Tensor) -> Result<f64, String> {
    if matrix.data.len() < 2 {
        return Err("Need at least 2 elements to compute variance".to_string());
    }

    let mean = matrix.data.iter().sum::<f64>() / matrix.data.len() as f64;
    let variance = matrix.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (matrix.data.len() - 1) as f64;

    Ok(variance)
}

#[runtime_builtin(name = "sqrt", accel = "unary")]
fn sqrt_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            if let Some(p) = runmat_accelerate_api::provider() {
                if let Ok(hc) = p.unary_sqrt(&h) {
                    return Ok(Value::GpuTensor(hc));
                }
            }
            Err("sqrt: unsupported for gpuArray".to_string())
        }
        Value::Num(n) => Ok(Value::Num(n.sqrt())),
        Value::Int(i) => Ok(Value::Num(i.to_f64().sqrt())),
        Value::Tensor(t) => {
            let data: Vec<f64> = t.data.iter().map(|&v| v.sqrt()).collect();
            Ok(Value::Tensor(Tensor::new_2d(data, t.rows(), t.cols())?))
        }
        other => Err(format!("sqrt: unsupported input {other:?}")),
    }
}

// Unit tests for mathematics live under crates/runmat-runtime/tests/mathematics.rs

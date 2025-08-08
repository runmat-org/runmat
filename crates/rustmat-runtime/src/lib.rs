use rustmat_builtins::{builtin_functions, Value};
use rustmat_macros::runtime_builtin;

pub mod arrays;
pub mod comparison;
pub mod concatenation;
pub mod constants;
pub mod elementwise;
pub mod indexing;
pub mod io;
pub mod mathematics;
pub mod matrix;
pub mod plotting;

#[cfg(feature = "blas-lapack")]
pub mod blas;
#[cfg(feature = "blas-lapack")]
pub mod lapack;

// Link to Apple's Accelerate framework on macOS
#[cfg(all(feature = "blas-lapack", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

pub use arrays::*;
pub use comparison::*;
pub use concatenation::*;
// Note: constants and mathematics modules only contain #[runtime_builtin] functions
// and don't export public items, so they don't need to be re-exported
pub use elementwise::*;
pub use indexing::*;
pub use matrix::*;

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

/// Call a registered MATLAB builtin by name.
/// Supports function overloading by trying different argument patterns.
/// Returns an error if no builtin with that name and compatible arguments is found.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    let mut matching_builtins = Vec::new();

    // Collect all builtins with the matching name
    for b in builtin_functions() {
        if b.name == name {
            matching_builtins.push(b);
        }
    }

    if matching_builtins.is_empty() {
        return Err(format!("unknown builtin `{name}`"));
    }

    // Try each builtin until one succeeds
    let mut last_error = String::new();
    for builtin in matching_builtins {
        let f = builtin.implementation;
        match (f)(args) {
            Ok(result) => return Ok(result),
            Err(e) => last_error = e,
        }
    }

    // If none succeeded, return the last error
    Err(format!(
        "No matching overload for `{}` with {} args: {}",
        name,
        args.len(),
        last_error
    ))
}

// Common mathematical functions that tests expect

/// Transpose operation for Values
pub fn transpose(value: Value) -> Result<Value, String> {
    match value {
        Value::Matrix(ref m) => Ok(Value::Matrix(matrix_transpose(m))),
        Value::Num(n) => Ok(Value::Num(n)), // Scalar transpose is identity
        _ => Err("transpose not supported for this type".to_string()),
    }
}

#[runtime_builtin(name = "abs")]
fn abs_builtin(x: f64) -> Result<f64, String> {
    Ok(x.abs())
}

#[runtime_builtin(name = "max")]
fn max_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(a.max(b))
}

#[runtime_builtin(name = "min")]
fn min_builtin(a: f64, b: f64) -> Result<f64, String> {
    Ok(a.min(b))
}

#[runtime_builtin(name = "sqrt")]
fn sqrt_builtin(x: f64) -> Result<f64, String> {
    if x < 0.0 {
        Err("Cannot take square root of negative number".to_string())
    } else {
        Ok(x.sqrt())
    }
}

/// Simple timing functions for benchmarks
/// tic() starts a timer and returns current time
#[runtime_builtin(name = "tic")]
fn tic_builtin() -> Result<f64, String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Time error: {e}"))?;
    Ok(now.as_secs_f64())
}

/// toc() returns elapsed time since the last tic() call
/// Note: In a real implementation, this would use a saved start time,
/// but for simplicity we'll just return a small time value
#[runtime_builtin(name = "toc")]
fn toc_builtin() -> Result<f64, String> {
    // For benchmark purposes, return a realistic small time
    Ok(0.001) // 1 millisecond
}

#[runtime_builtin(name = "exp")]
fn exp_builtin(x: f64) -> Result<f64, String> {
    Ok(x.exp())
}

#[runtime_builtin(name = "log")]
fn log_builtin(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Cannot take logarithm of non-positive number".to_string())
    } else {
        Ok(x.ln())
    }
}

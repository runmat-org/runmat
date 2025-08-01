use rustmat_builtins::{builtins, Value, BuiltinFn};
use rustmat_macros::runtime_builtin;

pub mod matrix;
pub mod comparison;
pub mod indexing;

#[cfg(feature = "blas-lapack")]
pub mod blas;
#[cfg(feature = "blas-lapack")]
pub mod lapack;

// Link to Apple's Accelerate framework on macOS
#[cfg(all(feature = "blas-lapack", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

pub use matrix::*;
pub use comparison::*;
pub use indexing::*;

#[cfg(feature = "blas-lapack")]
pub use blas::*;
#[cfg(feature = "blas-lapack")]
pub use lapack::*;

/// Call a registered MATLAB builtin by name.
/// Returns an error if no builtin with that name is found.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    for b in builtins() {
        if b.name == name {
            let f: BuiltinFn = b.func;
            return (f)(args);
        }
    }
    Err(format!("unknown builtin `{name}`"))
}

// Common mathematical functions that tests expect

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

#[runtime_builtin(name = "sin")]
fn sin_builtin(x: f64) -> Result<f64, String> {
    Ok(x.sin())
}

#[runtime_builtin(name = "cos")]
fn cos_builtin(x: f64) -> Result<f64, String> {
    Ok(x.cos())
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
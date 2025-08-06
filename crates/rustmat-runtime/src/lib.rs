use rustmat_builtins::{builtins, BuiltinFn, Value};
use rustmat_macros::runtime_builtin;

pub mod arrays;
pub mod comparison;
pub mod concatenation;
pub mod constants;
pub mod elementwise;
pub mod indexing;
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

#[allow(unused_imports)]
pub use arrays::*;
pub use comparison::*;
#[allow(unused_imports)]
pub use concatenation::*;
#[allow(unused_imports)]
pub use constants::*;
pub use elementwise::*;
pub use indexing::*;
#[allow(unused_imports)]
pub use mathematics::*;
pub use matrix::*;
// Note: plotting functions are registered automatically via runtime_builtin macro

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
    for b in builtins() {
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
        let f: BuiltinFn = builtin.func;
        match (f)(args) {
            Ok(result) => return Ok(result),
            Err(e) => last_error = e,
        }
    }
    
    // If none succeeded, return the last error
    Err(format!("No matching overload for `{}` with {} args: {}", name, args.len(), last_error))
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

// sin and cos are now defined in the mathematics module to handle both scalars and matrices

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

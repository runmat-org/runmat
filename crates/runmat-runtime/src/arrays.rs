//! Array generation functions
//!
//! This module provides array generation functions like linspace, logspace,
//! zeros, ones, eye, etc. These functions are optimized for performance and memory efficiency.

use runmat_builtins::Value;
use runmat_control_flow::RuntimeControlFlow;

/// Create a range vector (equivalent to start:end or start:step:end)
pub fn create_range(start: f64, step: Option<f64>, end: f64) -> Result<Value, RuntimeControlFlow> {
    // Delegate to new builtins to ensure unified semantics
    match step {
        Some(s) => crate::call_builtin(
            "colon",
            &[Value::Num(start), Value::Num(s), Value::Num(end)],
        ),
        None => crate::call_builtin("colon", &[Value::Num(start), Value::Num(end)]),
    }
}

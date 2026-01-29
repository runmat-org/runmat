//! Array generation functions
//!
//! This module provides array generation functions like linspace, logspace,
//! zeros, ones, eye, etc. These functions are optimized for performance and memory efficiency.

use crate::RuntimeError;
use runmat_builtins::Value;

/// Create a range vector (equivalent to start:end or start:step:end)
pub async fn create_range(start: f64, step: Option<f64>, end: f64) -> Result<Value, RuntimeError> {
    // Delegate to new builtins to ensure unified semantics
    match step {
        Some(s) => {
            crate::call_builtin_async(
                "colon",
                &[Value::Num(start), Value::Num(s), Value::Num(end)],
            )
            .await
        }
        None => crate::call_builtin_async("colon", &[Value::Num(start), Value::Num(end)]).await,
    }
}

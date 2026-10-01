//! Descriptive-statistics family boundary. Each implementation owns its own
//! metadata and numeric algorithm; only shared admission stays here.

pub(crate) mod flagged;
pub(crate) mod tabulation;
#[cfg(test)]
mod tests;

pub use flagged::{kurtosis, mad, skewness};
pub use tabulation::tabulate;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_value::Value;

fn descriptive_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn gathered(value: Value, name: &str) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

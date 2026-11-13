//! Checked conversion of a floating structural length to a host index.

use runmat_value::Value;

use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult};

pub(crate) fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

pub(crate) fn parse_nonnegative_usize(value: &Value, name: &str) -> BuiltinResult<usize> {
    let parsed = match value {
        Value::Num(n) => nonnegative_platform_usize(*n),
        Value::Int(i) => i.try_to_usize(),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage.value_at(0).and_then(|value| value.try_to_usize())
            } else {
                nonnegative_platform_usize(tensor_utils::tensor_value_f64(tensor, 0))
            }
        }
        _ => None,
    };
    parsed.ok_or_else(|| {
        build_runtime_error(format!("{name}: expected a nonnegative integer scalar"))
            .with_builtin(name)
            .build()
    })
}

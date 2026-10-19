//! Character-row construction from an exact scalar length.

use runmat_builtins::{
    BLANKS_DESCRIPTOR, BLANKS_ERROR_TOO_MANY_OUTPUTS, BLANKS_GPU_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntValue, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::numeric_length::nonnegative_platform_usize;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

#[runtime_builtin(
    name = "blanks",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::blanks"
)]
pub(crate) async fn blanks_builtin(n: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &BLANKS_DESCRIPTOR,
        &BLANKS_ERROR_TOO_MANY_OUTPUTS,
        "blanks",
    )?;
    if matches!(n, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BLANKS_GPU_INPUT_EXTENSION,
            "blanks",
        )?;
    }
    let n = gather_if_needed_async(&n)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "blanks"))?;
    let n = parse_blanks_length(&n)?;
    let mut spaces = String::new();
    spaces
        .try_reserve_exact(n)
        .map_err(|_| blanks_error("blanks: requested character array is too large"))?;
    spaces.extend(std::iter::repeat_n(' ', n));
    Ok(Value::CharArray(CharArray::new_row(&spaces)))
}

fn parse_blanks_length(value: &Value) -> BuiltinResult<usize> {
    let parsed = match value {
        Value::Num(n) if n.is_finite() && n.fract() == 0.0 && *n <= 0.0 => Some(0),
        Value::Num(n) => nonnegative_platform_usize(*n),
        Value::Int(value) => zero_clamped_integer_usize(value),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                storage
                    .value_at(0)
                    .and_then(|value| zero_clamped_integer_usize(&value))
            } else {
                let value = tensor_utils::tensor_value_f64(tensor, 0);
                if value.is_finite() && value.fract() == 0.0 && value <= 0.0 {
                    Some(0)
                } else {
                    nonnegative_platform_usize(value)
                }
            }
        }
        _ => None,
    };
    parsed.ok_or_else(|| blanks_error("blanks: expected an integer-valued numeric scalar"))
}

fn blanks_error(message: &'static str) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("blanks").build()
}

fn zero_clamped_integer_usize(value: &IntValue) -> Option<usize> {
    match value {
        IntValue::I8(value) => Some(if *value <= 0 { 0 } else { *value as usize }),
        IntValue::I16(value) => Some(if *value <= 0 { 0 } else { *value as usize }),
        IntValue::I32(value) => {
            if *value <= 0 {
                Some(0)
            } else {
                usize::try_from(*value).ok()
            }
        }
        IntValue::I64(value) => {
            if *value <= 0 {
                Some(0)
            } else {
                usize::try_from(*value).ok()
            }
        }
        IntValue::U8(value) => Some(*value as usize),
        IntValue::U16(value) => Some(*value as usize),
        IntValue::U32(value) => usize::try_from(*value).ok(),
        IntValue::U64(value) => usize::try_from(*value).ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_excess_outputs_before_gather() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = futures::executor::block_on(blanks_builtin(Value::Num(3.0)))
            .expect_err("fixed output bound");
        assert_eq!(error.identifier(), BLANKS_ERROR_TOO_MANY_OUTPUTS.identifier);
    }
}

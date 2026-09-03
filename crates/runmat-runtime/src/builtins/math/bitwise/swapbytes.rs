//! Native byte-order reversal for real numeric values.

use runmat_builtins::{SWAPBYTES_ERROR_INVALID_INPUT, SWAPBYTES_EXPLICIT_GPU_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericStorage, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "swapbytes";

#[runtime_builtin(
    name = "swapbytes",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::bitwise::swapbytes"
)]
async fn swapbytes_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SWAPBYTES_EXPLICIT_GPU_EXTENSION,
            NAME,
        )?;
    }
    let gathered = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(|error| invalid(error.message()))?;
    match gathered {
        Value::Num(value) => Ok(Value::Num(f64::from_bits(value.to_bits().swap_bytes()))),
        Value::Int(value) => Ok(Value::Int(swap_integer(value))),
        Value::Tensor(tensor) => swap_tensor(tensor),
        other => Err(invalid(format!("unsupported input {other:?}"))),
    }
}

fn swap_integer(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value),
        IntValue::I16(value) => IntValue::I16(value.swap_bytes()),
        IntValue::I32(value) => IntValue::I32(value.swap_bytes()),
        IntValue::I64(value) => IntValue::I64(value.swap_bytes()),
        IntValue::U8(value) => IntValue::U8(value),
        IntValue::U16(value) => IntValue::U16(value.swap_bytes()),
        IntValue::U32(value) => IntValue::U32(value.swap_bytes()),
        IntValue::U64(value) => IntValue::U64(value.swap_bytes()),
    }
}

fn swap_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(invalid)?;
    let swapped = match storage {
        NumericStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|value| f64::from_bits(value.to_bits().swap_bytes()))
                .collect(),
        ),
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| f32::from_bits(value.to_bits().swap_bytes()))
                .collect(),
        ),
        NumericStorage::I8(values) => NumericStorage::I8(values),
        NumericStorage::I16(values) => {
            NumericStorage::I16(values.into_iter().map(i16::swap_bytes).collect())
        }
        NumericStorage::I32(values) => {
            NumericStorage::I32(values.into_iter().map(i32::swap_bytes).collect())
        }
        NumericStorage::I64(values) => {
            NumericStorage::I64(values.into_iter().map(i64::swap_bytes).collect())
        }
        NumericStorage::U8(values) => NumericStorage::U8(values),
        NumericStorage::U16(values) => {
            NumericStorage::U16(values.into_iter().map(u16::swap_bytes).collect())
        }
        NumericStorage::U32(values) => {
            NumericStorage::U32(values.into_iter().map(u32::swap_bytes).collect())
        }
        NumericStorage::U64(values) => {
            NumericStorage::U64(values.into_iter().map(u64::swap_bytes).collect())
        }
    };
    Tensor::from_numeric_storage(swapped, shape)
        .map(Value::Tensor)
        .map_err(invalid)
}

fn invalid(detail: impl std::fmt::Display) -> RuntimeError {
    let message = format!("{}: {detail}", SWAPBYTES_ERROR_INVALID_INPUT.message);
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = SWAPBYTES_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests;

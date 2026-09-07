use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::{IntegerStorage, NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

use super::super::{builtin_error, host, BUILTIN_NAME};

pub(super) async fn gather_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            Ok(Value::Tensor(tensor))
        }
        other => Ok(other),
    }
}

pub(super) async fn value_to_real_tensor(value: &Value) -> BuiltinResult<Option<Tensor>> {
    match value {
        Value::Tensor(tensor) => Ok(Some(tensor.clone())),
        Value::Num(number) => Ok(Some(
            Tensor::new(vec![*number], vec![1, 1])
                .map_err(|error| builtin_error(format!("power: {error}")))?,
        )),
        Value::Int(integer) => Ok(Some(
            Tensor::new_integer(IntegerStorage::from_scalar(integer.clone()), vec![1, 1])
                .map_err(|error| builtin_error(format!("power: {error}")))?,
        )),
        Value::Bool(boolean) => Ok(Some(
            Tensor::new(vec![if *boolean { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|error| builtin_error(format!("power: {error}")))?,
        )),
        Value::LogicalArray(logical) => Ok(Some(
            tensor::logical_to_tensor(logical)
                .map_err(|error| builtin_error(format!("power: {error}")))?,
        )),
        Value::CharArray(chars) => Ok(Some(host::char_array_to_tensor(chars)?)),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            Ok(Some(tensor))
        }
        _ => Ok(None),
    }
}

pub(super) fn is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

pub(super) fn is_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

pub(super) fn tensor_matches_handle(tensor: &Tensor, handle: &GpuTensorHandle) -> bool {
    if tensor.integer_storage().is_some()
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
    {
        return false;
    }

    match tensor.numeric_dtype() {
        NumericDType::F64 => {
            runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F64)
        }
        NumericDType::F32 => {
            runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32)
        }
        NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => false,
    }
}

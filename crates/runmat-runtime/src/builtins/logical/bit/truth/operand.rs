use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, SharedShape, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct LogicalBuffer {
    pub(super) data: Vec<u8>,
    pub(super) shape: SharedShape,
}

pub(super) async fn from_value(
    builtin: &'static str,
    value: Value,
    invalid: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<LogicalBuffer> {
    admitted_shape(builtin, &value, invalid)?;
    match value {
        Value::LogicalArray(array) => from_logical(array),
        Value::Bool(value) => Ok(scalar(value)),
        Value::Num(value) => Ok(scalar(value != 0.0)),
        Value::Int(value) => Ok(scalar(!value.is_zero())),
        Value::Complex(real, imaginary) => Ok(scalar(real != 0.0 || imaginary != 0.0)),
        Value::Tensor(tensor) => Ok(from_tensor(tensor)),
        Value::ComplexTensor(tensor) => Ok(from_complex_tensor(tensor)),
        Value::CharArray(array) => Ok(from_char_array(array)),
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|error| error_with_detail(builtin, invalid, error))?;
            Box::pin(from_value(builtin, gathered, invalid)).await
        }
        other => Err(error_with_detail(
            builtin,
            invalid,
            format!("unsupported input type {other:?}"),
        )),
    }
}

/// Shared operand admission without conversion, gather or provider effects.
/// Conversion and preparation use this same class/shape authority.
pub(super) fn admitted_shape<'a>(
    builtin: &'static str,
    value: &'a Value,
    invalid: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<&'a [usize]> {
    match value {
        Value::Bool(_) | Value::Num(_) | Value::Int(_) | Value::Complex(_, _) => Ok(&[1, 1]),
        Value::LogicalArray(array) => Ok(&array.shape),
        Value::Tensor(array) => Ok(array.shape()),
        Value::ComplexTensor(array) => Ok(array.shape()),
        Value::CharArray(array) => Ok(&array.shape),
        Value::GpuTensor(handle) => Ok(handle.shape()),
        other => Err(error_with_detail(
            builtin,
            invalid,
            format!("unsupported input type {other:?}"),
        )),
    }
}

pub(super) fn into_value(
    builtin: &'static str,
    data: Vec<u8>,
    shape: SharedShape,
    invalid: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    if data.len() == 1 && crate::builtins::common::tensor::element_count(&shape) == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(|error| error_with_detail(builtin, invalid, error))
}

pub(super) fn is_complex(value: &Value) -> bool {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        }
        _ => false,
    }
}

pub(super) fn error_with_detail(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{builtin}: {detail}")).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn scalar(value: bool) -> LogicalBuffer {
    LogicalBuffer {
        data: vec![u8::from(value)],
        shape: vec![1, 1].into(),
    }
}

fn from_logical(array: LogicalArray) -> BuiltinResult<LogicalBuffer> {
    let LogicalArray { data, shape } = array;
    Ok(LogicalBuffer {
        data: data.into_vec(),
        shape,
    })
}

fn from_tensor(tensor: Tensor) -> LogicalBuffer {
    let shape = tensor.shared_shape();
    let data = (0..tensor.len())
        .map(|index| {
            u8::from(
                !tensor
                    .numeric_value_at(index)
                    .expect("validated tensor storage")
                    .is_zero(),
            )
        })
        .collect();
    LogicalBuffer { data, shape }
}

fn from_complex_tensor(tensor: ComplexTensor) -> LogicalBuffer {
    let shape = tensor.shared_shape();
    let data = (0..tensor.len())
        .map(|index| {
            let (real, imaginary) = tensor
                .numeric_value_at(index)
                .expect("validated complex tensor storage");
            u8::from(!real.is_zero() || !imaginary.is_zero())
        })
        .collect();
    LogicalBuffer { data, shape }
}

fn from_char_array(array: CharArray) -> LogicalBuffer {
    let data = array.to_column_major();
    let shape = array.shape.into();
    LogicalBuffer {
        data: data
            .into_iter()
            .map(|value| u8::from(value.raw() != 0))
            .collect(),
        shape,
    }
}

//! MATLAB-compatible `shiftdim` with native-class and GPU-residency preservation.

use crate::aggregate::structure::operations as struct_ops;

use crate::builtins::array::shape::permute::{
    permute_char_array, permute_complex_tensor, permute_generic, permute_logical_array,
    permute_string_array, permute_tensor,
};
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericScalar, StringArray, Value,
};

#[path = "shiftdim/admission.rs"]
mod admission;

const NAME: &str = "shiftdim";

fn shiftdim_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn effective_shape(shape: &[usize]) -> Vec<usize> {
    let mut shape = match shape {
        [] => vec![1, 1],
        [length] => vec![1, *length],
        _ => shape.to_vec(),
    };
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn canonical_shape(mut shape: Vec<usize>) -> Vec<usize> {
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    if shape.is_empty() {
        vec![1, 1]
    } else if shape.len() == 1 {
        shape.push(1);
        shape
    } else {
        shape
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ShiftPlan {
    RemoveLeading {
        shape: Vec<usize>,
        removed: usize,
    },
    Prepend {
        shape: Vec<usize>,
    },
    Rotate {
        order: Vec<usize>,
        permuted_shape: Vec<usize>,
        shape: Vec<usize>,
    },
}

fn make_plan(shape: &[usize], shift: Option<isize>) -> Result<ShiftPlan, RuntimeError> {
    let shape = effective_shape(shape);
    match shift {
        None => {
            let removed = shape
                .iter()
                .position(|dimension| *dimension != 1)
                .unwrap_or(0);
            let output = if removed == 0 {
                shape
            } else {
                canonical_shape(shape[removed..].to_vec())
            };
            Ok(ShiftPlan::RemoveLeading {
                shape: output,
                removed,
            })
        }
        Some(n) if n < 0 => {
            let count = n
                .checked_abs()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    shiftdim_error(
                        &SHIFTDIM_ERROR_INVALID_SHIFT,
                        "shiftdim: negative n is too large to represent",
                    )
                })?;
            let capacity = shape.len().checked_add(count).ok_or_else(|| {
                shiftdim_error(
                    &SHIFTDIM_ERROR_INVALID_SHIFT,
                    "shiftdim: shifted rank exceeds platform limits",
                )
            })?;
            let mut output = Vec::new();
            output.try_reserve_exact(capacity).map_err(|_| {
                shiftdim_error(
                    &SHIFTDIM_ERROR_INVALID_SHIFT,
                    "shiftdim: shifted rank exceeds available memory",
                )
            })?;
            output.resize(count, 1);
            output.extend(shape);
            Ok(ShiftPlan::Prepend { shape: output })
        }
        Some(n) => {
            let amount = usize::try_from(n).map_err(|_| {
                shiftdim_error(
                    &SHIFTDIM_ERROR_INVALID_SHIFT,
                    SHIFTDIM_ERROR_INVALID_SHIFT.message,
                )
            })? % shape.len();
            if amount == 0 {
                return Ok(ShiftPlan::RemoveLeading { shape, removed: 0 });
            }
            let order: Vec<usize> = (0..shape.len())
                .map(|index| {
                    let rotated = if index >= shape.len() - amount {
                        index - (shape.len() - amount)
                    } else {
                        index + amount
                    };
                    rotated + 1
                })
                .collect();
            let permuted_shape: Vec<usize> = order
                .iter()
                .map(|dimension| shape[*dimension - 1])
                .collect();
            let output = canonical_shape(permuted_shape.clone());
            Ok(ShiftPlan::Rotate {
                order,
                permuted_shape,
                shape: output,
            })
        }
    }
}

#[runtime_builtin(
    name = "shiftdim",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::shiftdim"
)]
async fn shiftdim_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(shiftdim_error(
            &SHIFTDIM_ERROR_TOO_MANY_INPUTS,
            SHIFTDIM_ERROR_TOO_MANY_INPUTS.message,
        ));
    }
    let explicit_shift = if let Some(value) = rest.first() {
        Some(parse_shift(value)?)
    } else {
        None
    };
    let requested = crate::output_count::current_output_count();
    if matches!(requested, Some(count) if count > 2)
        || (explicit_shift.is_some() && matches!(requested, Some(count) if count > 1))
    {
        return Err(shiftdim_error(
            &SHIFTDIM_ERROR_TOO_MANY_OUTPUTS,
            SHIFTDIM_ERROR_TOO_MANY_OUTPUTS.message,
        ));
    }
    if explicit_shift.is_some_and(|shift| shift < 0) {
        match &value {
            Value::SparseTensor(_) => {
                return Err(shiftdim_error(
                    &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
                    "shiftdim: sparse arrays currently support only their existing 2-D shape",
                ));
            }
            Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
            | Value::String(_)
            | Value::Struct(_) => return finish_shifted(value, 0, requested),
            _ => {}
        }
    }
    let admitted = match (
        explicit_shift.filter(|shift| *shift < 0),
        crate::context::legacy::active(),
    ) {
        (Some(shift), Some(runtime)) => match &value {
            Value::Tensor(tensor) => {
                admission::NegativeShiftShape::prepare(&runtime, tensor.shape(), shift)?
                    .map(|prepared| (runtime, prepared))
            }
            Value::ComplexTensor(tensor) => {
                admission::NegativeShiftShape::prepare(&runtime, tensor.shape(), shift)?
                    .map(|prepared| (runtime, prepared))
            }
            Value::LogicalArray(array) => {
                admission::NegativeShiftShape::prepare(&runtime, &array.shape, shift)?
                    .map(|prepared| (runtime, prepared))
            }
            _ => None,
        },
        _ => None,
    };
    let (shifted, removed) = if let Some((runtime, prepared)) = admitted {
        (prepared.publish(value, &runtime)?, 0)
    } else {
        let shape = value_shape(&value)?;
        let plan = make_plan(&shape, explicit_shift)?;
        let removed = match &plan {
            ShiftPlan::RemoveLeading { removed, .. } => *removed,
            _ => 0,
        };
        (apply_plan(value, &plan).await?, removed)
    };
    finish_shifted(shifted, removed, requested)
}

fn finish_shifted(
    shifted: Value,
    removed: usize,
    requested: Option<usize>,
) -> crate::BuiltinResult<Value> {
    match requested {
        None => Ok(shifted),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![shifted])),
        Some(2) => Ok(Value::output_list(vec![
            shifted,
            Value::Num(removed as f64),
        ])),
        Some(_) => unreachable!("output count validated above"),
    }
}

fn parse_shift(value: &Value) -> Result<isize, RuntimeError> {
    let scalar = match value {
        Value::Int(value) => {
            return value.try_to_isize().ok_or_else(|| {
                shiftdim_error(
                    &SHIFTDIM_ERROR_INVALID_SHIFT,
                    "shiftdim: n is outside the supported signed shift range",
                )
            })
        }
        Value::Num(value) => NumericScalar::F64(*value),
        Value::Bool(value) => NumericScalar::U8(u8::from(*value)),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            tensor.numeric_value_at(0).ok_or_else(|| {
                shiftdim_error(
                    &SHIFTDIM_ERROR_INVALID_SHIFT,
                    SHIFTDIM_ERROR_INVALID_SHIFT.message,
                )
            })?
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            NumericScalar::U8(logical.data[0])
        }
        _ => {
            return Err(shiftdim_error(
                &SHIFTDIM_ERROR_INVALID_SHIFT,
                SHIFTDIM_ERROR_INVALID_SHIFT.message,
            ))
        }
    };
    numeric_scalar_to_isize(scalar).ok_or_else(|| {
        shiftdim_error(
            &SHIFTDIM_ERROR_INVALID_SHIFT,
            SHIFTDIM_ERROR_INVALID_SHIFT.message,
        )
    })
}

fn numeric_scalar_to_isize(value: NumericScalar) -> Option<isize> {
    match value {
        NumericScalar::F64(value) => floating_to_isize(value),
        NumericScalar::F32(value) => floating_to_isize(f64::from(value)),
        other => other.into_int_value()?.try_to_isize(),
    }
}

fn floating_to_isize(value: f64) -> Option<isize> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < isize::MIN as f64
        || value >= (isize::MAX as u128 + 1) as f64
    {
        None
    } else {
        Some(value as isize)
    }
}

fn value_shape(value: &Value) -> Result<Vec<usize>, RuntimeError> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape().to_vec()),
        Value::ComplexTensor(tensor) => Ok(tensor.shape().to_vec()),
        Value::LogicalArray(array) => Ok(array.shape.to_vec()),
        Value::StringArray(array) => Ok(array.shape().to_vec()),
        Value::CharArray(array) => Ok(array.shape().to_vec()),
        Value::Cell(array) => Ok(array.shape().to_vec()),
        Value::StructArray(array) => Ok(array.shape().to_vec()),
        Value::GpuTensor(handle) => Ok(handle.shape().to_vec()),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::Struct(_) => Ok(vec![1, 1]),
        Value::SparseTensor(array) => Ok(vec![array.rows, array.cols]),
        other => Err(shiftdim_error(
            &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: unsupported input type {other:?}"),
        )),
    }
}

async fn apply_plan(value: Value, plan: &ShiftPlan) -> crate::BuiltinResult<Value> {
    match plan {
        ShiftPlan::RemoveLeading { shape, .. } | ShiftPlan::Prepend { shape } => {
            reshape_value(value, shape)
        }
        ShiftPlan::Rotate {
            order,
            permuted_shape,
            shape,
        } => {
            let raw_shape = value_shape(&value)?;
            let source_shape = effective_shape(&raw_shape);
            let value = if raw_shape == source_shape {
                value
            } else {
                reshape_value(value, &source_shape)?
            };
            let value = permute_value(value, order).await?;
            if shape == permuted_shape {
                Ok(value)
            } else {
                reshape_value(value, shape)
            }
        }
    }
}

async fn permute_value(value: Value, order: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => permute_tensor(NAME, tensor, order).map(Value::Tensor),
        Value::ComplexTensor(tensor) => {
            permute_complex_tensor(NAME, tensor, order).map(Value::ComplexTensor)
        }
        Value::LogicalArray(array) => {
            permute_logical_array(NAME, array, order).map(Value::LogicalArray)
        }
        Value::StringArray(array) => {
            permute_string_array(NAME, array, order).map(Value::StringArray)
        }
        Value::CharArray(array) => permute_char_array(NAME, array, order).map(Value::CharArray),
        Value::Cell(array) => {
            let (data, shape) =
                permute_generic(NAME, &array.to_column_major(), &array.shape, order)?;
            CellArray::from_column_major(data, shape)
                .map(Value::Cell)
                .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error))
        }
        Value::StructArray(array) => struct_ops::permute(array, order)
            .map(Value::StructArray)
            .map_err(|error| {
                struct_ops::remap(error, |message| {
                    shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, message)
                })
            }),
        Value::GpuTensor(handle) => permute_gpu(handle, order).await,
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::Struct(_) => Ok(value),
        Value::SparseTensor(_) => Err(shiftdim_error(
            &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
            "shiftdim: sparse arrays currently support only unshifted 2-D shapes",
        )),
        other => Err(shiftdim_error(
            &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: unsupported input type {other:?}"),
        )),
    }
}

fn reshape_value(value: Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => tensor
            .reshape(shape.to_vec())
            .map(Value::Tensor)
            .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error)),
        Value::ComplexTensor(tensor) => {
            ComplexTensor::from_complex_storage(tensor.into_complex_storage(), shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error))
        }
        Value::LogicalArray(array) => LogicalArray::from_host_buffer(array.data, shape.to_vec())
            .map(Value::LogicalArray)
            .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error)),
        Value::StringArray(array) => {
            StringArray::from_shared_storage(array.into_parts().0, shape.to_vec().into())
                .map(Value::StringArray)
                .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error))
        }
        Value::Cell(array) => CellArray::from_column_major(array.to_column_major(), shape.to_vec())
            .map(Value::Cell)
            .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error)),
        Value::StructArray(array) => struct_ops::reshape(array, shape.to_vec())
            .map(Value::StructArray)
            .map_err(|error| {
                struct_ops::remap(error, |message| {
                    shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, message)
                })
            }),
        Value::CharArray(array) => reshape_char(array, shape).map(Value::CharArray),
        Value::GpuTensor(handle) => reshape_gpu(handle, shape),
        Value::SparseTensor(array) if shape == [array.rows, array.cols] => {
            Ok(Value::SparseTensor(array))
        }
        Value::SparseTensor(_) => Err(shiftdim_error(
            &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
            "shiftdim: sparse arrays currently support only their existing 2-D shape",
        )),
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::String(_)
        | Value::Struct(_)
            if shape.iter().all(|dimension| *dimension == 1) =>
        {
            Ok(value)
        }
        other => Err(shiftdim_error(
            &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
            format!("shiftdim: input representation cannot express shape {shape:?}: {other:?}"),
        )),
    }
}

fn reshape_char(array: CharArray, shape: &[usize]) -> crate::BuiltinResult<CharArray> {
    CharArray::from_column_major(array.to_column_major(), shape.to_vec())
        .map_err(|error| shiftdim_error(&SHIFTDIM_ERROR_UNSUPPORTED_INPUT, error))
}

fn wrap_gpu(handle: GpuTensorHandle, complex: bool, logical: bool) -> crate::BuiltinResult<Value> {
    if complex {
        gpu_helpers::complex_gpu_value(handle)
    } else if logical {
        gpu_helpers::logical_gpu_value(handle)
    } else {
        gpu_helpers::resident_gpu_value(handle)
    }
}

async fn permute_gpu(handle: GpuTensorHandle, order: &[usize]) -> crate::BuiltinResult<Value> {
    let complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    if complex && runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let host = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
        let Value::ComplexTensor(tensor) = host else {
            return Err(shiftdim_error(
                &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
                "shiftdim: paired complex integer GPU input did not gather to exact complex storage",
            ));
        };
        let permuted = permute_complex_tensor(NAME, tensor, order).map(Value::ComplexTensor)?;
        return gpu_helpers::restore_class_preserving_value(&handle, permuted, NAME);
    }
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        shiftdim_error(
            &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
            "shiftdim: no acceleration provider owns the GPU input",
        )
    })?;
    let order: Vec<usize> = order.iter().map(|dimension| dimension - 1).collect();
    let mut output = match provider.permute(&handle, &order) {
        Ok(Some(output)) => output,
        Ok(None) => {
            let mut error = shiftdim_error(
                &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
                "shiftdim: provider permutation is unavailable",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return Err(error);
        }
        Err(source) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                shiftdim_error(
                    &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
                    format!("shiftdim: GPU permutation failed: {source}"),
                ),
                source,
            ))
        }
    };
    if !gpu_helpers::permutation_output_matches(&handle, &output, provider, &order) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            shiftdim_error(
                &SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
                "shiftdim: invalid provider permutation output",
            ),
        ));
    }
    output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
    wrap_gpu(output, complex, logical)
}

fn reshape_gpu(handle: GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let complex =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    let output = super::gpu_reshape::reshape(handle, shape, NAME)?;
    wrap_gpu(output, complex, logical)
}

#[cfg(test)]
#[path = "shiftdim/tests.rs"]
mod tests;

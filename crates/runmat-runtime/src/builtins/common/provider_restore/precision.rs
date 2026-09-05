use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{Tensor, Value};

use crate::{build_runtime_error, BuiltinResult};

pub(crate) fn align_floating_value_precision(
    value: Value,
    prototype: &GpuTensorHandle,
    builtin: &str,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(prototype).is_some()
        || runmat_accelerate_api::handle_is_logical(prototype)
    {
        return Ok(value);
    }
    let dtype = match runmat_accelerate_api::handle_precision(prototype) {
        Some(runmat_accelerate_api::ProviderPrecision::F32) => runmat_value::NumericDType::F32,
        _ => runmat_value::NumericDType::F64,
    };
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            let shape = tensor.shape.clone();
            let values = tensor.materialize_f64();
            let tensor = if dtype == runmat_value::NumericDType::F32 {
                Tensor::from_f32(
                    values.into_iter().map(|value| value as f32).collect(),
                    shape,
                )
            } else {
                Tensor::new(values, shape)
            }
            .map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
            Ok(Value::Tensor(tensor))
        }
        Value::ComplexTensor(tensor) => {
            let tensor = runmat_value::ComplexTensor::from_f64_values_with_dtype(
                tensor.materialize_f64(),
                tensor.shape.clone(),
                dtype,
            )
            .map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
            Ok(Value::ComplexTensor(tensor))
        }
        Value::Num(value) if dtype == runmat_value::NumericDType::F32 => {
            Tensor::from_f32(vec![value as f32], vec![1, 1])
                .map(Value::Tensor)
                .map_err(|error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_builtin(builtin)
                        .build()
                })
        }
        Value::Complex(re, im) if dtype == runmat_value::NumericDType::F32 => {
            runmat_value::ComplexTensor::from_f32(vec![(re as f32, im as f32)], vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(|error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_builtin(builtin)
                        .build()
                })
        }
        other => Ok(other),
    }
}

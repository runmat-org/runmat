use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinExtensionDescriptor, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value,
};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::{build_runtime_error, BuiltinResult};

pub(crate) fn ensure_input_extensions(
    value: &Value,
    builtin: &str,
    integer: &'static BuiltinExtensionDescriptor,
    logical: &'static BuiltinExtensionDescriptor,
    character: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    let is_integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some()
        );
    if is_integer {
        crate::compatibility::ensure_builtin_extension_enabled(integer, builtin)?;
    }
    let is_logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        );
    if is_logical {
        crate::compatibility::ensure_builtin_extension_enabled(logical, builtin)?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(character, builtin)?;
    }
    Ok(())
}

pub(crate) fn reject_excess_outputs(builtin: &str) -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(
            build_runtime_error(format!("{builtin}: too many output arguments"))
                .with_builtin(builtin)
                .with_identifier(format!("RunMat:{builtin}:TooManyOutputs"))
                .build(),
        );
    }
    Ok(())
}

pub(crate) async fn gather_compute_restore<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Tensor) -> BuiltinResult<Value>,
{
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .build()
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = compute(tensor)?;
    upload_value(provider, output, builtin)
}

pub(crate) fn upload_value(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
) -> BuiltinResult<Value> {
    let handle = match value {
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
            gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                build_runtime_error(format!(
                    "{builtin}: failed to restore result to input provider: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?
        }
        Value::Tensor(tensor) => {
            gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                build_runtime_error(format!(
                    "{builtin}: failed to restore result to input provider: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
            gpu_helpers::upload_complex_tensor(provider, &tensor)?
        }
        Value::ComplexTensor(tensor) => gpu_helpers::upload_complex_tensor(provider, &tensor)?,
        other => {
            return Err(build_runtime_error(format!(
                "{builtin}: cannot restore unsupported result {other:?} to provider"
            ))
            .with_builtin(builtin)
            .build())
        }
    };
    Ok(gpu_helpers::resident_gpu_value(handle))
}

pub(crate) fn map_real_tensor<F64, F32>(
    tensor: Tensor,
    builtin: &str,
    map_f64: F64,
    map_f32: F32,
) -> BuiltinResult<Tensor>
where
    F64: Fn(f64) -> f64,
    F32: Fn(f32) -> f32,
{
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(map_f64).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(map_f32).collect())
        }
        integer => NumericStorage::F64(
            integer
                .into_integer_storage()
                .expect("inverse function integer boundary")
                .to_f64_vec()
                .into_iter()
                .map(map_f64)
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(output, shape).map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })
}

pub(crate) fn map_complex_tensor<F64, F32>(
    tensor: ComplexTensor,
    builtin: &str,
    map_f64: F64,
    map_f32: F32,
) -> BuiltinResult<ComplexTensor>
where
    F64: Fn((f64, f64)) -> (f64, f64),
    F32: Fn((f32, f32)) -> (f32, f32),
{
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => {
            ComplexStorage::F64(values.into_iter().map(map_f64).collect())
        }
        ComplexStorage::F32(values) => {
            ComplexStorage::F32(values.into_iter().map(map_f32).collect())
        }
        ComplexStorage::Integer(_) => {
            return Err(build_runtime_error(format!(
                "{builtin}: typed complex integer input is not supported"
            ))
            .with_builtin(builtin)
            .build())
        }
    };
    ComplexTensor::from_complex_storage(storage, shape).map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })
}

pub(crate) fn map_real_tensor_promoting<F64, F32>(
    tensor: Tensor,
    builtin: &str,
    map_f64: F64,
    map_f32: F32,
) -> BuiltinResult<Value>
where
    F64: Fn(f64) -> (f64, f64),
    F32: Fn(f32) -> (f32, f32),
{
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(|error| {
        build_runtime_error(format!("{builtin}: {error}"))
            .with_builtin(builtin)
            .build()
    })?;
    match storage {
        NumericStorage::F64(values) => map_promoting_f64(values, shape, builtin, &map_f64),
        NumericStorage::F32(values) => map_promoting_f32(values, shape, builtin, &map_f32),
        integer => map_promoting_f64(
            integer
                .into_integer_storage()
                .expect("inverse function integer boundary")
                .to_f64_vec(),
            shape,
            builtin,
            &map_f64,
        ),
    }
}

fn map_promoting_f64<F>(
    values: Vec<f64>,
    shape: Vec<usize>,
    builtin: &str,
    map: &F,
) -> BuiltinResult<Value>
where
    F: Fn(f64) -> (f64, f64),
{
    if values.is_empty() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F64(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        return Ok(crate::builtins::common::tensor::tensor_into_value(tensor));
    }
    let mapped = values.into_iter().map(map).collect::<Vec<_>>();
    if mapped.iter().any(|(_, imag)| *imag != 0.0) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(mapped), shape)
            .map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = mapped.into_iter().map(|(real, _)| real).collect();
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F64(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(crate::builtins::common::tensor::tensor_into_value(tensor))
    }
}

fn map_promoting_f32<F>(
    values: Vec<f32>,
    shape: Vec<usize>,
    builtin: &str,
    map: &F,
) -> BuiltinResult<Value>
where
    F: Fn(f32) -> (f32, f32),
{
    if values.is_empty() {
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        return Ok(crate::builtins::common::tensor::tensor_into_value(tensor));
    }
    let mapped = values.into_iter().map(map).collect::<Vec<_>>();
    if mapped.iter().any(|(_, imag)| *imag != 0.0) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(mapped), shape)
            .map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = mapped.into_iter().map(|(real, _)| real).collect();
        let tensor =
            Tensor::from_numeric_storage(NumericStorage::F32(values), shape).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
        Ok(crate::builtins::common::tensor::tensor_into_value(tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerStorage, NumericDType};

    #[test]
    fn real_boundary_promotes_all_eight_integer_classes_to_double() {
        let inputs = [
            IntegerStorage::I8(vec![-1, 2]),
            IntegerStorage::I16(vec![-1, 2]),
            IntegerStorage::I32(vec![-1, 2]),
            IntegerStorage::I64(vec![-1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];

        for input in inputs {
            let tensor = Tensor::new_integer(input, vec![2, 1]).expect("integer tensor");
            let output = map_real_tensor(tensor, "inverse-test", |value| value, |value| value)
                .expect("integer-to-double boundary");
            assert_eq!(output.numeric_dtype(), NumericDType::F64);
            assert_eq!(output.shape, vec![2, 1]);
        }
    }
}

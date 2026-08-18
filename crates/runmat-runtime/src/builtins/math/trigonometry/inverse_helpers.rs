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
    upload_value_like(provider, output, builtin, &handle)
}

pub(crate) fn upload_value_like(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    upload_value_like_protected(provider, value, builtin, prototype, &[])
}

pub(crate) fn upload_value_like_protected(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
    prototype: &GpuTensorHandle,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let output = upload_value_protected(provider, value, builtin, protected)?;
    let Value::GpuTensor(mut handle) = output else {
        unreachable!("upload_value always returns a resident value")
    };
    if handle.device_id != prototype.device_id {
        free_unless_protected(provider, &handle, protected);
        return Err(build_runtime_error(format!(
            "{builtin}: provider restored the result on the wrong device"
        ))
        .with_builtin(builtin)
        .build());
    }
    if let Some(provenance) = runmat_accelerate_api::handle_provenance(prototype) {
        runmat_accelerate_api::set_handle_provenance(&mut handle, provenance);
    }
    Ok(Value::GpuTensor(handle))
}

pub(crate) fn upload_value(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
) -> BuiltinResult<Value> {
    upload_value_protected(provider, value, builtin, &[])
}

pub(crate) fn upload_value_protected(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let (expected_shape, expected_storage, expected_integer_type, expected_precision) = match &value
    {
        Value::Num(_) => (
            vec![1, 1],
            runmat_accelerate_api::GpuTensorStorage::Real,
            None,
            Some(runmat_accelerate_api::ProviderPrecision::F64),
        ),
        Value::Tensor(tensor) => (
            tensor.shape.clone(),
            runmat_accelerate_api::GpuTensorStorage::Real,
            tensor.integer_storage().map(integer_storage_type),
            if tensor.integer_storage().is_some() {
                None
            } else if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            } else {
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            },
        ),
        Value::Complex(_, _) => (
            vec![1, 1],
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            None,
            Some(runmat_accelerate_api::ProviderPrecision::F64),
        ),
        Value::ComplexTensor(tensor) => (
            tensor.shape.clone(),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            None,
            Some(
                if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
                    runmat_accelerate_api::ProviderPrecision::F32
                } else {
                    runmat_accelerate_api::ProviderPrecision::F64
                },
            ),
        ),
        other => {
            return Err(build_runtime_error(format!(
                "{builtin}: cannot restore unsupported result {other:?} to provider"
            ))
            .with_builtin(builtin)
            .build())
        }
    };
    if expected_precision.is_some_and(|precision| provider.precision() != precision) {
        return Err(build_runtime_error(format!(
            "{builtin}: input provider cannot restore the requested result precision"
        ))
        .with_builtin(builtin)
        .build());
    }
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
            upload_complex_without_precision_override(provider, &tensor, builtin)?
        }
        Value::ComplexTensor(tensor) => {
            upload_complex_without_precision_override(provider, &tensor, builtin)?
        }
        other => unreachable!("validated restore value {other:?}"),
    };
    let valid = handle.shape == expected_shape
        && runmat_accelerate_api::handle_storage(&handle) == expected_storage
        && runmat_accelerate_api::handle_integer_type(&handle) == expected_integer_type
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && (expected_integer_type.is_some()
            || runmat_accelerate_api::handle_precision(&handle) == expected_precision)
        && runmat_accelerate_api::provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !valid {
        free_unless_protected(provider, &handle, protected);
        return Err(build_runtime_error(format!(
            "{builtin}: provider returned an incompatible restored result"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn free_unless_protected(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    protected: &[GpuTensorHandle],
) {
    if protected.iter().any(|candidate| {
        candidate.device_id == handle.device_id && candidate.buffer_id == handle.buffer_id
    }) {
        return;
    }
    let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(provider);
    let _ = owner.free(handle);
}

fn upload_complex_without_precision_override(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
    builtin: &str,
) -> BuiltinResult<GpuTensorHandle> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(format!(
            "{builtin}: typed complex integer GPU buffers are not supported"
        ))
        .with_builtin(builtin)
        .build());
    }
    let mut interleaved = Vec::with_capacity(tensor.len().saturating_mul(2));
    for &(real, imag) in tensor.materialize_f64().iter() {
        interleaved.push(real);
        interleaved.push(imag);
    }
    let handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &interleaved,
            shape: &tensor.shape,
        })
        .map_err(|error| {
            build_runtime_error(format!(
                "{builtin}: failed to restore result to input provider: {error}"
            ))
            .with_builtin(builtin)
            .build()
        })?;
    runmat_accelerate_api::set_handle_logical(&handle, false);
    runmat_accelerate_api::set_handle_storage(
        &handle,
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
    );
    Ok(handle)
}

fn integer_storage_type(
    storage: &runmat_builtins::IntegerStorage,
) -> runmat_accelerate_api::IntegerElementType {
    use runmat_accelerate_api::IntegerElementType;
    use runmat_builtins::IntegerStorage;
    match storage {
        IntegerStorage::I8(_) => IntegerElementType::I8,
        IntegerStorage::I16(_) => IntegerElementType::I16,
        IntegerStorage::I32(_) => IntegerElementType::I32,
        IntegerStorage::I64(_) => IntegerElementType::I64,
        IntegerStorage::U8(_) => IntegerElementType::U8,
        IntegerStorage::U16(_) => IntegerElementType::U16,
        IntegerStorage::U32(_) => IntegerElementType::U32,
        IntegerStorage::U64(_) => IntegerElementType::U64,
    }
}

pub(crate) fn ensure_integer_exact_f64(value: &Value, builtin: &str) -> BuiltinResult<()> {
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(build_runtime_error(format!(
            "{builtin}: integer input must be exactly representable as double"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:InvalidInput"))
        .build())
    }
}

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
        Some(runmat_accelerate_api::ProviderPrecision::F32) => runmat_builtins::NumericDType::F32,
        _ => runmat_builtins::NumericDType::F64,
    };
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_none() => {
            let shape = tensor.shape.clone();
            let values = tensor.materialize_f64();
            let tensor = if dtype == runmat_builtins::NumericDType::F32 {
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
            let tensor = runmat_builtins::ComplexTensor::from_f64_values_with_dtype(
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
        Value::Num(value) if dtype == runmat_builtins::NumericDType::F32 => {
            Tensor::from_f32(vec![value as f32], vec![1, 1])
                .map(Value::Tensor)
                .map_err(|error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_builtin(builtin)
                        .build()
                })
        }
        Value::Complex(re, im) if dtype == runmat_builtins::NumericDType::F32 => {
            runmat_builtins::ComplexTensor::from_f32(vec![(re as f32, im as f32)], vec![1, 1])
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
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, NumericDType};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct F32OnlyProvider {
        upload_calls: AtomicUsize,
    }

    impl runmat_accelerate_api::AccelProvider for F32OnlyProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<GpuTensorHandle> {
            self.upload_calls.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("unexpected upload"))
        }

        fn download<'a>(
            &'a self,
            _handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("download unsupported")) })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "f32-only-test".to_string()
        }

        fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
            runmat_accelerate_api::ProviderPrecision::F32
        }
    }

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

    #[test]
    fn restore_preserves_non_f32_exact_double_without_metadata_relabeling() {
        test_support::with_test_provider(|provider| {
            let value = 1.0000000000000002_f64;
            let output =
                upload_value(provider, Value::Num(value), "restore-test").expect("double restore");
            let Value::GpuTensor(handle) = output else {
                panic!("expected resident output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            let downloaded = block_on(provider.download(&handle)).expect("download");
            assert_eq!(downloaded.data, vec![value]);
        });
    }

    #[test]
    fn restore_rejects_double_before_upload_on_f32_only_provider() {
        let provider = F32OnlyProvider {
            upload_calls: AtomicUsize::new(0),
        };
        let error = upload_value(&provider, Value::Num(1.0000000000000002), "restore-test")
            .expect_err("f32-only provider cannot supply a double result");
        assert!(error.message().contains("requested result precision"));
        assert_eq!(provider.upload_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn restore_rejects_single_before_upload_on_f64_only_provider() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::from_f32(vec![1.0000001], vec![1, 1]).expect("single");
            let error = upload_value(provider, Value::Tensor(tensor), "restore-test")
                .expect_err("f64-only provider cannot supply a single result");
            assert!(error.message().contains("requested result precision"));
        });
    }
}

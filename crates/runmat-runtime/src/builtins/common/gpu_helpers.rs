use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostIntegerDataOwned, HostIntegerDataView,
    HostIntegerTensorView, HostTensorView, IntegerElementType, ProviderPrecision,
};
use runmat_builtins::{
    ComplexStorage, ComplexTensor, IntegerStorage, LogicalArray, NumericDType, Tensor, Value,
};

use crate::build_runtime_error;
use crate::builtins::common::tensor;

/// Download a GPU tensor handle to host memory, returning a dense `Tensor`.
///
/// This helper routes through the dispatcher so residency hooks and provider
/// semantics stay consistent with the rest of the runtime.
pub async fn gather_tensor_async(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> crate::BuiltinResult<Tensor> {
    // Ensure the correct provider is active for WGPU-backed handles when tests run in parallel.
    // This mirrors the guard used in test_support::gather.
    #[cfg(all(test, feature = "wgpu"))]
    {
        let active_owner = runmat_accelerate_api::provider()
            .is_some_and(|provider| provider.device_id() == handle.device_id);
        if handle.device_id != 0 && !active_owner {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let value = Value::GpuTensor(handle.clone());
    let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
    match gathered {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| build_runtime_error(format!("gather: {e}")).build()),
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(data, la.shape.clone())
                .map_err(|e| build_runtime_error(format!("gather: {e}")).build())
        }
        other => {
            Err(build_runtime_error(format!("gather: unexpected value kind {other:?}")).build())
        }
    }
}

/// Gather an arbitrary value, returning a host-side `Value`.
pub async fn gather_value_async(value: &Value) -> crate::BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(value).await
}

/// Download a handle through its owner without changing the handle's residency or metadata.
pub async fn download_value_preserving_residency_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> crate::BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        let host = provider.download_integer(handle).await.map_err(|error| {
            build_runtime_error(format!("gpu download: {error}"))
                .with_identifier("RunMat:gpu:DownloadFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
        let storage = match host.data {
            HostIntegerDataOwned::I8(values) => IntegerStorage::I8(values),
            HostIntegerDataOwned::I16(values) => IntegerStorage::I16(values),
            HostIntegerDataOwned::I32(values) => IntegerStorage::I32(values),
            HostIntegerDataOwned::I64(values) => IntegerStorage::I64(values),
            HostIntegerDataOwned::U8(values) => IntegerStorage::U8(values),
            HostIntegerDataOwned::U16(values) => IntegerStorage::U16(values),
            HostIntegerDataOwned::U32(values) => IntegerStorage::U32(values),
            HostIntegerDataOwned::U64(values) => IntegerStorage::U64(values),
        };
        return Tensor::new_integer(storage, host.shape)
            .map(Value::Tensor)
            .map_err(|error| {
                build_runtime_error(format!("gpu download: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            });
    }

    let host = crate::dispatcher::download_handle_async(provider, handle)
        .await
        .map_err(|error| {
            build_runtime_error(format!("gpu download: {error}"))
                .with_identifier("RunMat:gpu:DownloadFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
    if runmat_accelerate_api::handle_is_logical(handle) {
        let bits = host
            .data
            .into_iter()
            .map(|value| u8::from(value != 0.0))
            .collect();
        return LogicalArray::new(bits, host.shape)
            .map(Value::LogicalArray)
            .map_err(|error| {
                build_runtime_error(format!("gpu download: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            });
    }

    let precision =
        runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
    if host.storage == GpuTensorStorage::ComplexInterleaved {
        if host.data.len() % 2 != 0 {
            return Err(build_runtime_error(
                "gpu download: complex-interleaved buffer has odd scalar length",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let pairs = host.data.chunks_exact(2).map(|pair| (pair[0], pair[1]));
        let storage = match precision {
            ProviderPrecision::F32 => {
                ComplexStorage::F32(pairs.map(|(re, im)| (re as f32, im as f32)).collect())
            }
            ProviderPrecision::F64 => ComplexStorage::F64(pairs.collect()),
        };
        return ComplexTensor::from_complex_storage(storage, host.shape)
            .map(Value::ComplexTensor)
            .map_err(|error| {
                build_runtime_error(format!("gpu download: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            });
    }

    let dtype = match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    };
    Tensor::new_with_dtype(host.data, host.shape, dtype)
        .map(Value::Tensor)
        .map_err(|error| {
            build_runtime_error(format!("gpu download: {error}"))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })
}

/// Upload a host complex tensor as an interleaved GPU buffer and record complex
/// storage metadata on the returned handle.
pub fn upload_complex_tensor(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
) -> crate::BuiltinResult<GpuTensorHandle> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(
            "typed complex integer GPU buffers are not supported by the active acceleration provider",
        )
        .build());
    }

    let mut interleaved = Vec::with_capacity(tensor.materialize_f64().len() * 2);
    for &(re, im) in &tensor.materialize_f64() {
        interleaved.push(re);
        interleaved.push(im);
    }
    let view = HostTensorView {
        data: &interleaved,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| build_runtime_error(format!("gpu upload: {e}")).build())?;
    runmat_accelerate_api::set_handle_logical(&handle, false);
    runmat_accelerate_api::set_handle_storage(&handle, GpuTensorStorage::ComplexInterleaved);
    runmat_accelerate_api::set_handle_precision(&handle, provider.precision());
    Ok(handle)
}

/// Upload a host tensor while retaining its exact typed-integer backing store.
pub fn upload_tensor(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
) -> Result<GpuTensorHandle, String> {
    if let Some(storage) = tensor.integer_storage() {
        let data = match storage {
            IntegerStorage::I8(values) => HostIntegerDataView::I8(values),
            IntegerStorage::I16(values) => HostIntegerDataView::I16(values),
            IntegerStorage::I32(values) => HostIntegerDataView::I32(values),
            IntegerStorage::I64(values) => HostIntegerDataView::I64(values),
            IntegerStorage::U8(values) => HostIntegerDataView::U8(values),
            IntegerStorage::U16(values) => HostIntegerDataView::U16(values),
            IntegerStorage::U32(values) => HostIntegerDataView::U32(values),
            IntegerStorage::U64(values) => HostIntegerDataView::U64(values),
        };
        provider
            .upload_integer(&HostIntegerTensorView {
                data,
                shape: &tensor.shape,
            })
            .map_err(|error| error.to_string())
    } else {
        let data = tensor::tensor_values_f64_cow(tensor);
        provider
            .upload(&HostTensorView {
                data: data.as_ref(),
                shape: &tensor.shape,
            })
            .map_err(|error| error.to_string())
    }
}

/// Upload a finite integral scalar in the native integer class of `prototype`.
/// Returns `None` when preserving MATLAB's typed-integer scalar semantics would
/// require the host extended-precision path instead.
pub fn upload_exact_integer_scalar_like(
    provider: &dyn AccelProvider,
    prototype: &GpuTensorHandle,
    scalar: f64,
) -> Option<GpuTensorHandle> {
    if !scalar.is_finite() || scalar.fract() != 0.0 {
        return None;
    }
    let element_type = runmat_accelerate_api::handle_integer_type(prototype)?;
    let shape = [1usize, 1usize];
    macro_rules! upload {
        ($value:expr, $variant:ident) => {{
            let values = [$value];
            provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::$variant(&values),
                    shape: &shape,
                })
                .ok()
        }};
    }
    match element_type {
        IntegerElementType::I8 if scalar >= i8::MIN as f64 && scalar <= i8::MAX as f64 => {
            upload!(scalar as i8, I8)
        }
        IntegerElementType::I16 if scalar >= i16::MIN as f64 && scalar <= i16::MAX as f64 => {
            upload!(scalar as i16, I16)
        }
        IntegerElementType::I32 if scalar >= i32::MIN as f64 && scalar <= i32::MAX as f64 => {
            upload!(scalar as i32, I32)
        }
        IntegerElementType::I64
            if scalar >= i64::MIN as f64 && scalar < 9_223_372_036_854_775_808.0 =>
        {
            upload!(scalar as i64, I64)
        }
        IntegerElementType::U8 if scalar >= 0.0 && scalar <= u8::MAX as f64 => {
            upload!(scalar as u8, U8)
        }
        IntegerElementType::U16 if scalar >= 0.0 && scalar <= u16::MAX as f64 => {
            upload!(scalar as u16, U16)
        }
        IntegerElementType::U32 if scalar >= 0.0 && scalar <= u32::MAX as f64 => {
            upload!(scalar as u32, U32)
        }
        IntegerElementType::U64 if (0.0..18_446_744_073_709_551_616.0).contains(&scalar) => {
            upload!(scalar as u64, U64)
        }
        _ => None,
    }
}

/// Wrap a GPU tensor handle, marking it as resident for downstream fusion-aware
/// consumers and tests.
pub fn resident_gpu_value(handle: GpuTensorHandle) -> Value {
    runmat_accelerate_api::mark_residency(&handle);
    Value::GpuTensor(handle)
}

/// Wrap a GPU tensor handle as a logical gpuArray value, recording metadata so that
/// predicates like `islogical` can inspect the handle without downloading it.
pub fn logical_gpu_value(handle: GpuTensorHandle) -> Value {
    runmat_accelerate_api::set_handle_logical(&handle, true);
    resident_gpu_value(handle)
}

#[cfg(test)]
mod preserving_download_tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    #[test]
    fn owner_download_preserves_integer_source_residency_and_metadata() {
        test_support::with_test_provider(|provider| {
            let source = Tensor::new_integer(IntegerStorage::U64(vec![3, 4]), vec![1, 2])
                .expect("integer source");
            let handle = upload_tensor(provider, &source).expect("upload integer source");
            let metadata = (
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::handle_precision(&handle),
                runmat_accelerate_api::handle_integer_type(&handle),
                runmat_accelerate_api::handle_is_logical(&handle),
            );
            let downloaded = block_on(download_value_preserving_residency_async(provider, &handle))
                .expect("non-destructive owner download");
            let Value::Tensor(downloaded) = downloaded else {
                panic!("expected integer tensor")
            };
            assert_eq!(
                downloaded.into_numeric_storage().unwrap(),
                runmat_builtins::NumericStorage::U64(vec![3, 4])
            );
            assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
            assert_eq!(runmat_accelerate_api::handle_storage(&handle), metadata.0);
            assert_eq!(runmat_accelerate_api::handle_precision(&handle), metadata.1);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                metadata.2
            );
            assert_eq!(
                runmat_accelerate_api::handle_is_logical(&handle),
                metadata.3
            );
        });
    }

    #[test]
    fn preserving_download_contract_errors_are_terminal_to_dispatcher_retry() {
        let error =
            build_runtime_error("gpu download: complex-interleaved buffer has odd scalar length")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build();
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
}

/// Wrap a GPU tensor handle as a complex gpuArray value.
pub fn complex_gpu_value(handle: GpuTensorHandle) -> Value {
    runmat_accelerate_api::set_handle_logical(&handle, false);
    runmat_accelerate_api::set_handle_storage(&handle, GpuTensorStorage::ComplexInterleaved);
    resident_gpu_value(handle)
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};

    #[test]
    fn exact_integer_scalar_upload_preserves_64_bit_class_and_admission_rules() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let shape = [1usize, 1usize];
        let signed = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[0]),
                shape: &shape,
            })
            .expect("upload int64 prototype");
        let unsigned = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0]),
                shape: &shape,
            })
            .expect("upload uint64 prototype");
        let signed_scalar = upload_exact_integer_scalar_like(provider, &signed, -7.0)
            .expect("representable int64 scalar");
        let unsigned_scalar = upload_exact_integer_scalar_like(provider, &unsigned, 7.0)
            .expect("representable uint64 scalar");
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&signed_scalar),
            Some(IntegerElementType::I64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&unsigned_scalar),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            block_on(provider.download_integer(&signed_scalar))
                .expect("download int64")
                .data,
            HostIntegerDataOwned::I64(vec![-7])
        );
        assert_eq!(
            block_on(provider.download_integer(&unsigned_scalar))
                .expect("download uint64")
                .data,
            HostIntegerDataOwned::U64(vec![7])
        );
        assert!(upload_exact_integer_scalar_like(provider, &signed, 1.5).is_none());
        assert!(upload_exact_integer_scalar_like(provider, &unsigned, -1.0).is_none());
        for handle in [&signed, &unsigned, &signed_scalar, &unsigned_scalar] {
            provider.free(handle).expect("free integer scalar handle");
        }
    }
}

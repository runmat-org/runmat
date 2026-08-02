use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostIntegerDataView, HostIntegerTensorView,
    HostTensorView, IntegerElementType,
};
use runmat_builtins::{ComplexTensor, IntegerStorage, Tensor, Value};

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
        if handle.device_id != 0 {
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

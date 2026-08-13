use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostIntegerDataOwned, HostIntegerDataView,
    HostIntegerTensorView, HostTensorView, IntegerElementType, ProviderPrecision,
};
use runmat_builtins::{
    ComplexStorage, ComplexTensor, IntegerStorage, LogicalArray, NumericDType, Tensor, Value,
};

use crate::build_runtime_error;
use crate::builtins::common::tensor;

/// Resolve the provider that actually owns `handle`.
///
/// `provider_for_handle` retains a legacy active-provider fallback for older
/// callers. Handle operations must additionally prove that the provider's
/// device namespace matches the durable handle identity before touching it.
pub fn exact_provider_for_handle(handle: &GpuTensorHandle) -> Option<&'static dyn AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|provider| provider.device_id() == handle.device_id)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuHandleMetadataSnapshot {
    storage: GpuTensorStorage,
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
    logical: bool,
    transpose: Option<runmat_accelerate_api::TransposeInfo>,
    class_name: Option<String>,
    provenance: Option<runmat_accelerate_api::GpuHandleProvenance>,
}

pub fn snapshot_handle_metadata(handle: &GpuTensorHandle) -> GpuHandleMetadataSnapshot {
    GpuHandleMetadataSnapshot {
        storage: runmat_accelerate_api::handle_storage(handle),
        precision: runmat_accelerate_api::handle_precision(handle),
        integer: runmat_accelerate_api::handle_integer_type(handle),
        logical: runmat_accelerate_api::handle_is_logical(handle),
        transpose: runmat_accelerate_api::handle_transpose_info(handle),
        class_name: runmat_accelerate_api::handle_class_name(handle),
        provenance: runmat_accelerate_api::handle_provenance(handle),
    }
}

pub fn restore_handle_metadata(handle: &GpuTensorHandle, snapshot: &GpuHandleMetadataSnapshot) {
    runmat_accelerate_api::set_handle_storage(handle, snapshot.storage);
    match snapshot.precision {
        Some(precision) => runmat_accelerate_api::set_handle_precision(handle, precision),
        None => runmat_accelerate_api::clear_handle_precision(handle),
    }
    match snapshot.integer {
        Some(integer) => runmat_accelerate_api::set_handle_integer_type(handle, integer),
        None => runmat_accelerate_api::clear_handle_integer_type(handle),
    }
    runmat_accelerate_api::set_handle_logical(handle, snapshot.logical);
    match snapshot.transpose {
        Some(info) => {
            runmat_accelerate_api::record_handle_transpose(handle, info.base_rows, info.base_cols)
        }
        None => runmat_accelerate_api::clear_handle_transpose(handle),
    }
    match snapshot.class_name.as_deref() {
        Some(class_name) => runmat_accelerate_api::set_handle_class_name(handle, class_name),
        None => runmat_accelerate_api::clear_handle_class_name(handle),
    }
    match snapshot.provenance {
        Some(provenance) => runmat_accelerate_api::set_handle_provenance(handle, provenance),
        None => runmat_accelerate_api::clear_handle_provenance(handle),
    }
    runmat_accelerate_api::mark_residency(handle);
}

struct HandleMetadataRestoreGuard<'a> {
    handle: &'a GpuTensorHandle,
    snapshot: GpuHandleMetadataSnapshot,
}

impl<'a> HandleMetadataRestoreGuard<'a> {
    fn new(handle: &'a GpuTensorHandle) -> Self {
        Self {
            handle,
            snapshot: snapshot_handle_metadata(handle),
        }
    }
}

impl Drop for HandleMetadataRestoreGuard<'_> {
    fn drop(&mut self) {
        restore_handle_metadata(self.handle, &self.snapshot);
    }
}

pub fn expected_gpu_class_name(
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
    logical: bool,
) -> Option<&'static str> {
    if logical {
        return Some("logical");
    }
    if let Some(integer) = integer {
        return Some(match integer {
            IntegerElementType::I8 => "int8",
            IntegerElementType::I16 => "int16",
            IntegerElementType::I32 => "int32",
            IntegerElementType::I64 => "int64",
            IntegerElementType::U8 => "uint8",
            IntegerElementType::U16 => "uint16",
            IntegerElementType::U32 => "uint32",
            IntegerElementType::U64 => "uint64",
        });
    }
    precision.map(|precision| match precision {
        ProviderPrecision::F32 => "single",
        ProviderPrecision::F64 => "double",
    })
}

pub fn gpu_class_metadata_matches(
    handle: &GpuTensorHandle,
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
    logical: bool,
) -> bool {
    let expected = expected_gpu_class_name(precision, integer, logical);
    runmat_accelerate_api::handle_class_name(handle)
        .as_deref()
        .is_none_or(|actual| expected == Some(actual))
}

pub fn same_gpu_handle(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    left.device_id == right.device_id && left.buffer_id == right.buffer_id
}

pub fn free_unprotected_exact_owner(handle: &GpuTensorHandle, protected: &[&GpuTensorHandle]) {
    if protected
        .iter()
        .any(|protected| same_gpu_handle(handle, protected))
    {
        return;
    }
    if let Some(owner) = exact_provider_for_handle(handle) {
        if owner.free(handle).is_ok() {
            runmat_accelerate_api::clear_handle_metadata(handle);
        }
    }
}

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
    let source_guard = HandleMetadataRestoreGuard::new(handle);
    if provider.device_id() != handle.device_id {
        return Err(
            build_runtime_error("gpu download: provider does not own the input handle")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        );
    }
    let precision = source_guard.snapshot.precision;
    let integer = source_guard.snapshot.integer;
    let logical = source_guard.snapshot.logical;
    let expected_class = expected_gpu_class_name(precision, integer, logical);
    if source_guard
        .snapshot
        .class_name
        .as_deref()
        .is_some_and(|actual| expected_class != Some(actual))
    {
        return Err(build_runtime_error(
            "gpu download: class metadata contradicts the physical payload metadata",
        )
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    if integer.is_none() && precision != Some(provider.precision()) {
        return Err(build_runtime_error(
            "gpu download: floating precision metadata contradicts the owning provider",
        )
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    if let Some(expected_type) = integer {
        let host = provider.download_integer(handle).await.map_err(|error| {
            build_runtime_error(format!("gpu download: {error}"))
                .with_identifier("RunMat:gpu:DownloadFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
        if host.shape != handle.shape || host.data.element_type() != expected_type {
            return Err(provider_payload_mismatch(
                handle,
                &host.shape,
                format!(
                    "integer class {:?}, expected {:?}",
                    host.data.element_type(),
                    expected_type
                ),
            ));
        }
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
    let expected_storage = source_guard.snapshot.storage;
    if host.shape != handle.shape || host.storage != expected_storage {
        return Err(provider_payload_mismatch(
            handle,
            &host.shape,
            format!(
                "storage {:?}, expected {:?}",
                host.storage, expected_storage
            ),
        ));
    }
    if logical {
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

    let precision = precision.unwrap_or_else(|| provider.precision());
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

fn provider_payload_mismatch(
    handle: &GpuTensorHandle,
    actual_shape: &[usize],
    detail: String,
) -> crate::RuntimeError {
    build_runtime_error(format!(
        "gpu download: provider payload mismatch ({detail}; shape {actual_shape:?}, expected {:?})",
        handle.shape
    ))
    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
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

/// Restore a class-preserving host value to the provider that owns `source`.
///
/// If the owner cannot physically represent the floating class, the host value
/// is returned instead of relabelling it. Integer and logical storage retain
/// their exact metadata independently of the provider's floating precision.
pub fn restore_class_preserving_value(
    source: &GpuTensorHandle,
    value: Value,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let provider = exact_provider_for_handle(source).ok_or_else(|| {
        build_runtime_error(format!(
            "{builtin}: no acceleration provider owns the input handle"
        ))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    })?;
    let source_guard = HandleMetadataRestoreGuard::new(source);

    let (output, expected_shape, expected_storage, expected_precision, expected_integer, logical) =
        match &value {
            Value::Tensor(tensor) => {
                let expected_integer = tensor.integer_storage().map(|storage| match storage {
                    IntegerStorage::I8(_) => IntegerElementType::I8,
                    IntegerStorage::I16(_) => IntegerElementType::I16,
                    IntegerStorage::I32(_) => IntegerElementType::I32,
                    IntegerStorage::I64(_) => IntegerElementType::I64,
                    IntegerStorage::U8(_) => IntegerElementType::U8,
                    IntegerStorage::U16(_) => IntegerElementType::U16,
                    IntegerStorage::U32(_) => IntegerElementType::U32,
                    IntegerStorage::U64(_) => IntegerElementType::U64,
                });
                let expected_precision = if expected_integer.is_some() {
                    None
                } else {
                    Some(match tensor.numeric_dtype() {
                        NumericDType::F32 => ProviderPrecision::F32,
                        _ => ProviderPrecision::F64,
                    })
                };
                if expected_precision.is_some_and(|precision| provider.precision() != precision) {
                    return Ok(value);
                }
                let output = upload_tensor(provider, tensor).map_err(|error| {
                    build_runtime_error(format!("{builtin}: failed to restore GPU result: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
                (
                    output,
                    tensor.shape.clone(),
                    GpuTensorStorage::Real,
                    expected_precision,
                    expected_integer,
                    false,
                )
            }
            Value::LogicalArray(array) => {
                let tensor = Tensor::new(
                    array.data.iter().map(|bit| f64::from(*bit != 0)).collect(),
                    array.shape.clone(),
                )
                .map_err(|error| {
                    build_runtime_error(format!("{builtin}: invalid logical result: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
                let output = upload_tensor(provider, &tensor).map_err(|error| {
                    build_runtime_error(format!("{builtin}: failed to restore GPU result: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
                runmat_accelerate_api::set_handle_logical(&output, true);
                (
                    output,
                    array.shape.clone(),
                    GpuTensorStorage::Real,
                    Some(provider.precision()),
                    None,
                    true,
                )
            }
            Value::ComplexTensor(tensor) => {
                if tensor.integer_storage().is_some() {
                    return Ok(value);
                }
                let expected_precision = match tensor.numeric_dtype() {
                    NumericDType::F32 => ProviderPrecision::F32,
                    _ => ProviderPrecision::F64,
                };
                if provider.precision() != expected_precision {
                    return Ok(value);
                }
                let output = upload_complex_tensor(provider, tensor).map_err(|error| {
                    build_runtime_error(format!("{builtin}: failed to restore GPU result: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
                (
                    output,
                    tensor.shape.clone(),
                    GpuTensorStorage::ComplexInterleaved,
                    Some(expected_precision),
                    None,
                    false,
                )
            }
            _ => return Ok(value),
        };

    let aliases_source = same_gpu_handle(&output, source);
    if aliases_source {
        return Err(build_runtime_error(format!(
            "{builtin}: provider aliased the protected input while restoring the result"
        ))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }

    let valid = output.shape == expected_shape
        && output.device_id == provider.device_id()
        && exact_provider_for_handle(&output).is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == expected_storage
        && expected_precision.is_none_or(|precision| {
            runmat_accelerate_api::handle_precision(&output) == Some(precision)
        })
        && runmat_accelerate_api::handle_integer_type(&output) == expected_integer
        && runmat_accelerate_api::handle_is_logical(&output) == logical
        && gpu_class_metadata_matches(&output, expected_precision, expected_integer, logical);
    if !valid {
        free_unprotected_exact_owner(&output, &[source]);
        return Err(build_runtime_error(format!(
            "{builtin}: provider returned an invalid restored result"
        ))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    runmat_accelerate_api::set_handle_provenance(
        &output,
        source_guard
            .snapshot
            .provenance
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
    runmat_accelerate_api::mark_residency(&output);
    Ok(Value::GpuTensor(output))
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

    struct MalformedDownloadProvider;

    impl runmat_accelerate_api::AccelProvider for MalformedDownloadProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("unused")
        }

        fn download<'a>(
            &'a self,
            _handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async {
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data: vec![1.0, 2.0],
                    shape: vec![2, 1],
                    storage: GpuTensorStorage::Real,
                })
            })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "malformed download test provider".into()
        }
    }

    struct MutatingDownloadProvider;

    impl runmat_accelerate_api::AccelProvider for MutatingDownloadProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("unused")
        }

        fn download<'a>(
            &'a self,
            handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async move {
                runmat_accelerate_api::set_handle_logical(handle, true);
                runmat_accelerate_api::set_handle_class_name(handle, "logical");
                Ok(runmat_accelerate_api::HostTensorOwned {
                    data: vec![1.0],
                    shape: handle.shape.clone(),
                    storage: GpuTensorStorage::Real,
                })
            })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "mutating download test provider".into()
        }
    }

    #[test]
    fn owner_download_preserves_integer_source_residency_and_metadata() {
        test_support::with_test_provider(|provider| {
            let source = Tensor::new_integer(IntegerStorage::U64(vec![3, 4]), vec![1, 2])
                .expect("integer source");
            let handle = upload_tensor(provider, &source).expect("upload integer source");
            runmat_accelerate::ensure_residency_hooks();
            runmat_accelerate_api::mark_residency(&handle);
            runmat_accelerate_api::record_handle_transpose(&handle, 2, 1);
            let snapshot = snapshot_handle_metadata(&handle);
            runmat_accelerate_api::clear_residency(&handle);
            runmat_accelerate_api::clear_handle_transpose(&handle);
            restore_handle_metadata(&handle, &snapshot);
            assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
            assert_eq!(
                runmat_accelerate_api::handle_transpose_info(&handle),
                Some(runmat_accelerate_api::TransposeInfo {
                    base_rows: 2,
                    base_cols: 1,
                })
            );
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

    #[test]
    fn preserving_download_restores_metadata_mutated_by_provider() {
        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: u64::MAX - 8,
        };
        runmat_accelerate_api::set_handle_storage(&handle, GpuTensorStorage::Real);
        runmat_accelerate_api::set_handle_precision(&handle, ProviderPrecision::F64);
        runmat_accelerate_api::set_handle_logical(&handle, false);
        runmat_accelerate_api::set_handle_class_name(&handle, "double");
        let result = block_on(download_value_preserving_residency_async(
            &MutatingDownloadProvider,
            &handle,
        ))
        .expect("valid payload should download");
        assert!(matches!(result, Value::Tensor(_)));
        assert!(!runmat_accelerate_api::handle_is_logical(&handle));
        assert_eq!(
            runmat_accelerate_api::handle_class_name(&handle).as_deref(),
            Some("double")
        );
    }

    #[test]
    fn preserving_download_rejects_provider_payload_shape_mismatch() {
        let handle = GpuTensorHandle {
            shape: vec![1, 2],
            device_id: 0,
            buffer_id: u64::MAX - 7,
        };
        let error = block_on(download_value_preserving_residency_async(
            &MalformedDownloadProvider,
            &handle,
        ))
        .expect_err("provider payload metadata must match the source handle");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }

    #[test]
    fn preserving_download_rejects_missing_or_contradictory_floating_precision() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let handle = upload_tensor(provider, &tensor).expect("upload");
            runmat_accelerate_api::clear_handle_precision(&handle);
            let missing = block_on(download_value_preserving_residency_async(provider, &handle))
                .expect_err("missing physical precision must reject");
            assert_eq!(
                missing.identifier(),
                Some("RunMat:gpu:ProviderPayloadMismatch")
            );
            let contradictory = match provider.precision() {
                ProviderPrecision::F32 => ProviderPrecision::F64,
                ProviderPrecision::F64 => ProviderPrecision::F32,
            };
            runmat_accelerate_api::set_handle_precision(&handle, contradictory);
            let mismatch = block_on(download_value_preserving_residency_async(provider, &handle))
                .expect_err("contradictory physical precision must reject");
            assert_eq!(
                mismatch.identifier(),
                Some("RunMat:gpu:ProviderPayloadMismatch")
            );
        });
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

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, IntegerElementType, NumericElementType,
    ProviderPrecision,
};
use runmat_builtins::{
    ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray,
    NumericDType, NumericStorage, Tensor, Value,
};

use crate::build_runtime_error;

/// Resolve the provider that actually owns `handle`.
///
/// `provider_for_handle` retains a legacy active-provider fallback for older
/// callers. Handle operations must additionally prove that the provider's
/// device namespace matches the durable handle identity before touching it.
pub fn exact_provider_for_handle(handle: &GpuTensorHandle) -> Option<&'static dyn AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|provider| provider.device_id() == handle.device_id)
}

/// Select one owning provider for a set of resident inputs, with explicit
/// gpuArray provenance taking precedence over automatic residency.
pub fn select_resident_output_source(
    handles: impl IntoIterator<Item = GpuTensorHandle>,
    builtin: &str,
) -> crate::BuiltinResult<Option<GpuTensorHandle>> {
    let mut selected: Option<(GpuTensorHandle, &'static dyn AccelProvider)> = None;
    for handle in handles {
        let owner = exact_provider_for_handle(&handle).ok_or_else(|| {
            build_runtime_error(format!(
                "{builtin}: no acceleration provider owns a resident input"
            ))
            .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        })?;
        if let Some((current, current_owner)) = &selected {
            if current.device_id != handle.device_id || !std::ptr::eq(*current_owner, owner) {
                return Err(build_runtime_error(format!(
                    "{builtin}: resident inputs must share one owning provider"
                ))
                .with_identifier("RunMat:gpu:MixedProviders")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build());
            }
            if !runmat_accelerate_api::handle_is_explicit(current)
                && runmat_accelerate_api::handle_is_explicit(&handle)
            {
                selected = Some((handle, owner));
            }
        } else {
            selected = Some((handle, owner));
        }
    }
    Ok(selected.map(|(handle, _)| handle))
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

fn integer_element_type(storage: &IntegerStorage) -> IntegerElementType {
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

pub(crate) fn expected_handle_numeric_element_type(
    handle: &GpuTensorHandle,
) -> Result<NumericElementType, String> {
    let precision = runmat_accelerate_api::handle_precision(handle);
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let logical = runmat_accelerate_api::handle_is_logical(handle);
    let expected_class = expected_gpu_class_name(precision, integer, logical);
    if runmat_accelerate_api::handle_class_name(handle)
        .as_deref()
        .is_some_and(|actual| expected_class != Some(actual))
    {
        return Err("class metadata contradicts the physical payload metadata".into());
    }
    if logical && integer.is_some() {
        return Err("logical metadata cannot annotate native integer storage".into());
    }
    match (precision, integer) {
        (Some(ProviderPrecision::F64), None) => Ok(NumericElementType::F64),
        (Some(ProviderPrecision::F32), None) => Ok(NumericElementType::F32),
        (None, Some(integer)) => Ok(integer.into()),
        (Some(_), Some(_)) => {
            Err("floating precision and integer class metadata are both present".into())
        }
        (None, None) => Err("physical numeric element metadata is missing".into()),
    }
}

fn deinterleave<T: Copy>(values: &[T]) -> (Vec<T>, Vec<T>) {
    let mut real = Vec::with_capacity(values.len() / 2);
    let mut imag = Vec::with_capacity(values.len() / 2);
    for pair in values.chunks_exact(2) {
        real.push(pair[0]);
        imag.push(pair[1]);
    }
    (real, imag)
}

pub(crate) fn value_from_numeric_download(
    host: HostNumericTensorOwned,
    logical: bool,
) -> Result<Value, String> {
    host.validate().map_err(|error| error.to_string())?;
    if logical {
        if host.storage != GpuTensorStorage::Real {
            return Err("logical payload must use real storage".into());
        }
        let bits = match host.data {
            HostNumericDataOwned::F64(values) => values
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            HostNumericDataOwned::F32(values) => values
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            HostNumericDataOwned::I8(_)
            | HostNumericDataOwned::I16(_)
            | HostNumericDataOwned::I32(_)
            | HostNumericDataOwned::I64(_)
            | HostNumericDataOwned::U8(_)
            | HostNumericDataOwned::U16(_)
            | HostNumericDataOwned::U32(_)
            | HostNumericDataOwned::U64(_) => {
                return Err("logical payload cannot use native integer storage".into())
            }
        };
        return LogicalArray::new(bits, host.shape).map(Value::LogicalArray);
    }

    if host.storage == GpuTensorStorage::Real {
        let storage = match host.data {
            HostNumericDataOwned::F64(values) => NumericStorage::F64(values),
            HostNumericDataOwned::F32(values) => NumericStorage::F32(values),
            HostNumericDataOwned::I8(values) => NumericStorage::I8(values),
            HostNumericDataOwned::I16(values) => NumericStorage::I16(values),
            HostNumericDataOwned::I32(values) => NumericStorage::I32(values),
            HostNumericDataOwned::I64(values) => NumericStorage::I64(values),
            HostNumericDataOwned::U8(values) => NumericStorage::U8(values),
            HostNumericDataOwned::U16(values) => NumericStorage::U16(values),
            HostNumericDataOwned::U32(values) => NumericStorage::U32(values),
            HostNumericDataOwned::U64(values) => NumericStorage::U64(values),
        };
        return Tensor::from_numeric_storage(storage, host.shape).map(Value::Tensor);
    }

    macro_rules! integer_complex {
        ($values:expr, $variant:ident) => {{
            let (real, imag) = deinterleave(&$values);
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::$variant(real),
                IntegerStorage::$variant(imag),
            )?)
        }};
    }
    let storage = match host.data {
        HostNumericDataOwned::F64(values) => {
            let (real, imag) = deinterleave(&values);
            ComplexStorage::F64(real.into_iter().zip(imag).collect())
        }
        HostNumericDataOwned::F32(values) => {
            let (real, imag) = deinterleave(&values);
            ComplexStorage::F32(real.into_iter().zip(imag).collect())
        }
        HostNumericDataOwned::I8(values) => integer_complex!(values, I8),
        HostNumericDataOwned::I16(values) => integer_complex!(values, I16),
        HostNumericDataOwned::I32(values) => integer_complex!(values, I32),
        HostNumericDataOwned::I64(values) => integer_complex!(values, I64),
        HostNumericDataOwned::U8(values) => integer_complex!(values, U8),
        HostNumericDataOwned::U16(values) => integer_complex!(values, U16),
        HostNumericDataOwned::U32(values) => integer_complex!(values, U32),
        HostNumericDataOwned::U64(values) => integer_complex!(values, U64),
    };
    ComplexTensor::from_complex_storage(storage, host.shape).map(Value::ComplexTensor)
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
    let expected_element = expected_handle_numeric_element_type(handle).map_err(|error| {
        build_runtime_error(format!("gpu download: {error}"))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let host = provider.download_numeric(handle).await.map_err(|error| {
        build_runtime_error(format!("gpu download: {error}"))
            .with_identifier("RunMat:gpu:DownloadFailed")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    if host.shape != handle.shape
        || host.storage != source_guard.snapshot.storage
        || host.data.element_type() != expected_element
    {
        return Err(provider_payload_mismatch(
            handle,
            &host.shape,
            format!(
                "{:?} {:?}, expected {:?} {:?}",
                host.data.element_type(),
                host.storage,
                expected_element,
                source_guard.snapshot.storage
            ),
        ));
    }
    value_from_numeric_download(host, source_guard.snapshot.logical).map_err(|error| {
        build_runtime_error(format!("gpu download: {error}"))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
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

fn interleave<T: Copy>(real: &[T], imag: &[T]) -> Vec<T> {
    let mut values = Vec::with_capacity(real.len().saturating_mul(2));
    for (&real, &imag) in real.iter().zip(imag) {
        values.push(real);
        values.push(imag);
    }
    values
}

fn interleaved_complex_data(tensor: &ComplexTensor) -> Result<HostNumericDataOwned, String> {
    macro_rules! integer_interleaved {
        ($real:expr, $imag:expr, $variant:ident) => {
            HostNumericDataOwned::$variant(interleave($real, $imag))
        };
    }
    match tensor.complex_storage() {
        ComplexStorage::F64(values) => Ok(HostNumericDataOwned::F64(
            values
                .iter()
                .flat_map(|&(real, imag)| [real, imag])
                .collect(),
        )),
        ComplexStorage::F32(values) => Ok(HostNumericDataOwned::F32(
            values
                .iter()
                .flat_map(|&(real, imag)| [real, imag])
                .collect(),
        )),
        ComplexStorage::Integer(storage) => match (&storage.real, &storage.imag) {
            (IntegerStorage::I8(real), IntegerStorage::I8(imag)) => {
                Ok(integer_interleaved!(real, imag, I8))
            }
            (IntegerStorage::I16(real), IntegerStorage::I16(imag)) => {
                Ok(integer_interleaved!(real, imag, I16))
            }
            (IntegerStorage::I32(real), IntegerStorage::I32(imag)) => {
                Ok(integer_interleaved!(real, imag, I32))
            }
            (IntegerStorage::I64(real), IntegerStorage::I64(imag)) => {
                Ok(integer_interleaved!(real, imag, I64))
            }
            (IntegerStorage::U8(real), IntegerStorage::U8(imag)) => {
                Ok(integer_interleaved!(real, imag, U8))
            }
            (IntegerStorage::U16(real), IntegerStorage::U16(imag)) => {
                Ok(integer_interleaved!(real, imag, U16))
            }
            (IntegerStorage::U32(real), IntegerStorage::U32(imag)) => {
                Ok(integer_interleaved!(real, imag, U32))
            }
            (IntegerStorage::U64(real), IntegerStorage::U64(imag)) => {
                Ok(integer_interleaved!(real, imag, U64))
            }
            _ => Err("complex integer components must have matching classes".into()),
        },
    }
}

/// Upload a host complex tensor through the shared native numeric transfer contract.
pub fn upload_complex_tensor(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
) -> crate::BuiltinResult<GpuTensorHandle> {
    let data = interleaved_complex_data(tensor)
        .map_err(|error| build_runtime_error(format!("gpu upload: {error}")).build())?;
    provider
        .upload_numeric(&HostNumericTensorView {
            data: data.as_view(),
            shape: &tensor.shape,
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .map_err(|error| build_runtime_error(format!("gpu upload: {error}")).build())
}

fn tensor_numeric_view(tensor: &Tensor) -> HostNumericDataView<'_> {
    if let Some(values) = tensor.as_f64_slice() {
        return HostNumericDataView::F64(values);
    }
    if let Some(values) = tensor.as_f32_slice() {
        return HostNumericDataView::F32(values);
    }
    match tensor
        .integer_storage()
        .expect("non-floating tensor has integer storage")
    {
        IntegerStorage::I8(values) => HostNumericDataView::I8(values),
        IntegerStorage::I16(values) => HostNumericDataView::I16(values),
        IntegerStorage::I32(values) => HostNumericDataView::I32(values),
        IntegerStorage::I64(values) => HostNumericDataView::I64(values),
        IntegerStorage::U8(values) => HostNumericDataView::U8(values),
        IntegerStorage::U16(values) => HostNumericDataView::U16(values),
        IntegerStorage::U32(values) => HostNumericDataView::U32(values),
        IntegerStorage::U64(values) => HostNumericDataView::U64(values),
    }
}

/// Upload a host tensor through the shared native numeric transfer contract.
pub fn upload_tensor(
    provider: &dyn AccelProvider,
    tensor: &Tensor,
) -> Result<GpuTensorHandle, String> {
    provider
        .upload_numeric(&HostNumericTensorView {
            data: tensor_numeric_view(tensor),
            shape: &tensor.shape,
            storage: GpuTensorStorage::Real,
        })
        .map_err(|error| error.to_string())
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
                let expected_integer = tensor.integer_storage().map(integer_element_type);
                let expected_precision = if expected_integer.is_some() {
                    None
                } else {
                    Some(match tensor.numeric_dtype() {
                        NumericDType::F32 => ProviderPrecision::F32,
                        _ => ProviderPrecision::F64,
                    })
                };
                let output = match upload_tensor(provider, tensor) {
                    Ok(output) => output,
                    Err(_) if expected_precision != Some(provider.precision()) => return Ok(value),
                    Err(error) => {
                        return Err(build_runtime_error(format!(
                            "{builtin}: failed to restore GPU result: {error}"
                        ))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build())
                    }
                };
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
                let storage = match provider.precision() {
                    ProviderPrecision::F32 => NumericStorage::F32(
                        array.data.iter().map(|bit| f32::from(*bit != 0)).collect(),
                    ),
                    ProviderPrecision::F64 => NumericStorage::F64(
                        array.data.iter().map(|bit| f64::from(*bit != 0)).collect(),
                    ),
                };
                let tensor = Tensor::from_numeric_storage(storage, array.shape.clone()).map_err(
                    |error| {
                        build_runtime_error(format!("{builtin}: invalid logical result: {error}"))
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build()
                    },
                )?;
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
                let expected_integer = tensor
                    .integer_storage()
                    .map(|storage| integer_element_type(&storage.real));
                let expected_precision =
                    expected_integer
                        .is_none()
                        .then(|| match tensor.numeric_dtype() {
                            NumericDType::F32 => ProviderPrecision::F32,
                            _ => ProviderPrecision::F64,
                        });
                let output = match upload_complex_tensor(provider, tensor) {
                    Ok(output) => output,
                    Err(_)
                        if expected_integer.is_some()
                            || expected_precision != Some(provider.precision()) =>
                    {
                        return Ok(value)
                    }
                    Err(error) => {
                        return Err(build_runtime_error(format!(
                            "{builtin}: failed to restore GPU result: {error}"
                        ))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build())
                    }
                };
                (
                    output,
                    tensor.shape.clone(),
                    GpuTensorStorage::ComplexInterleaved,
                    expected_precision,
                    expected_integer,
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
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::$variant(&values),
                    shape: &shape,
                    storage: GpuTensorStorage::Real,
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
    fn central_runtime_transfer_round_trips_every_native_real_and_complex_class() {
        test_support::with_test_provider(|provider| {
            let real_cases = vec![
                NumericStorage::F64(vec![-1.25, 2.5]),
                NumericStorage::F32(vec![-1.25, 2.5]),
                NumericStorage::I8(vec![i8::MIN, i8::MAX]),
                NumericStorage::I16(vec![i16::MIN, i16::MAX]),
                NumericStorage::I32(vec![i32::MIN, i32::MAX]),
                NumericStorage::I64(vec![i64::MIN, i64::MAX]),
                NumericStorage::U8(vec![0, u8::MAX]),
                NumericStorage::U16(vec![0, u16::MAX]),
                NumericStorage::U32(vec![0, u32::MAX]),
                NumericStorage::U64(vec![1_u64 << 63, u64::MAX]),
            ];
            for storage in real_cases {
                let expected = storage.clone();
                let tensor = Tensor::from_numeric_storage(storage, vec![1, 2]).unwrap();
                let handle = upload_tensor(provider, &tensor).expect("native real upload");
                let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                    &Value::GpuTensor(handle.clone()),
                ))
                .expect("native real gather");
                let Value::Tensor(gathered) = gathered else {
                    panic!("real transfer reconstructed the wrong value kind")
                };
                assert_eq!(gathered.into_numeric_storage().unwrap(), expected);
                provider.free(&handle).unwrap();
                runmat_accelerate_api::clear_handle_metadata(&handle);
            }

            macro_rules! integer_complex_case {
                ($variant:ident, $real:expr, $imag:expr) => {
                    ComplexStorage::Integer(
                        IntegerComplexStorage::new(
                            IntegerStorage::$variant($real),
                            IntegerStorage::$variant($imag),
                        )
                        .unwrap(),
                    )
                };
            }
            let complex_cases = vec![
                ComplexStorage::F64(vec![(-1.25, 3.5), (2.5, -4.75)]),
                ComplexStorage::F32(vec![(-1.25, 3.5), (2.5, -4.75)]),
                integer_complex_case!(I8, vec![i8::MIN, i8::MAX], vec![1, -1]),
                integer_complex_case!(I16, vec![i16::MIN, i16::MAX], vec![1, -1]),
                integer_complex_case!(I32, vec![i32::MIN, i32::MAX], vec![1, -1]),
                integer_complex_case!(I64, vec![i64::MIN, i64::MAX], vec![1, -1]),
                integer_complex_case!(U8, vec![0, u8::MAX], vec![1, 2]),
                integer_complex_case!(U16, vec![0, u16::MAX], vec![1, 2]),
                integer_complex_case!(U32, vec![0, u32::MAX], vec![1, 2]),
                integer_complex_case!(U64, vec![1_u64 << 63, u64::MAX], vec![1, 2]),
            ];
            for storage in complex_cases {
                let expected = storage.clone();
                let tensor = ComplexTensor::from_complex_storage(storage, vec![1, 2]).unwrap();
                let handle =
                    upload_complex_tensor(provider, &tensor).expect("native complex upload");
                let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                    &Value::GpuTensor(handle.clone()),
                ))
                .expect("native complex gather");
                let Value::ComplexTensor(gathered) = gathered else {
                    panic!("complex transfer reconstructed the wrong value kind")
                };
                assert_eq!(gathered.into_complex_storage(), expected);
                provider.free(&handle).unwrap();
                runmat_accelerate_api::clear_handle_metadata(&handle);
            }
        });
    }

    #[test]
    fn class_preserving_restore_uses_shared_single_and_complex_integer_storage() {
        test_support::with_test_provider(|provider| {
            let source =
                upload_tensor(provider, &Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()).unwrap();
            runmat_accelerate_api::mark_handle_automatic(&source);

            let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).unwrap();
            let restored =
                restore_class_preserving_value(&source, Value::Tensor(single.clone()), "test")
                    .expect("native single restore");
            let Value::GpuTensor(single_handle) = restored else {
                panic!("shared provider should preserve native single residency")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&single_handle),
                Some(ProviderPrecision::F32)
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(single_handle.clone()),
            ))
            .unwrap();
            assert_eq!(gathered, Value::Tensor(single));

            let complex = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                    IntegerStorage::U64(vec![3, 4]),
                )
                .unwrap(),
                vec![1, 2],
            )
            .unwrap();
            let restored = restore_class_preserving_value(
                &source,
                Value::ComplexTensor(complex.clone()),
                "test",
            )
            .expect("complex integer restore");
            let Value::GpuTensor(complex_handle) = restored else {
                panic!("shared provider should preserve complex integer residency")
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&complex_handle),
                Some(IntegerElementType::U64)
            );
            assert_eq!(
                runmat_accelerate_api::handle_storage(&complex_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(complex_handle.clone()),
            ))
            .unwrap();
            assert_eq!(gathered, Value::ComplexTensor(complex));

            let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
            let restored = restore_class_preserving_value(
                &source,
                Value::LogicalArray(logical.clone()),
                "test",
            )
            .expect("logical restore");
            let Value::GpuTensor(logical_handle) = restored else {
                panic!("logical restoration should retain residency")
            };
            assert!(runmat_accelerate_api::handle_is_logical(&logical_handle));
            assert_eq!(
                runmat_accelerate_api::handle_class_name(&logical_handle).as_deref(),
                Some("logical")
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(logical_handle.clone()),
            ))
            .unwrap();
            assert_eq!(gathered, Value::LogicalArray(logical));

            for handle in [&single_handle, &complex_handle, &logical_handle, &source] {
                provider.free(handle).unwrap();
                runmat_accelerate_api::clear_handle_metadata(handle);
            }
        });
    }

    #[test]
    fn class_preserving_logical_restore_uses_owner_physical_precision() {
        test_support::with_f32_test_provider(|provider| {
            let source = upload_tensor(
                provider,
                &Tensor::from_f32(vec![0.0, 0.0], vec![1, 2]).unwrap(),
            )
            .unwrap();
            runmat_accelerate_api::mark_handle_automatic(&source);
            let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
            let restored = restore_class_preserving_value(
                &source,
                Value::LogicalArray(logical.clone()),
                "test",
            )
            .expect("logical restore");
            let Value::GpuTensor(handle) = restored else {
                panic!("logical restoration should retain residency")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(ProviderPrecision::F32)
            );
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(handle.clone()),
            ))
            .unwrap();
            assert_eq!(gathered, Value::LogicalArray(logical));
            for handle in [&handle, &source] {
                provider.free(handle).unwrap();
                runmat_accelerate_api::clear_handle_metadata(handle);
            }
        });
    }

    #[test]
    fn resident_output_source_prefers_explicit_intent_independent_of_order() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let automatic = upload_tensor(provider, &tensor).unwrap();
            let explicit = upload_tensor(provider, &tensor).unwrap();
            runmat_accelerate_api::mark_handle_automatic(&automatic);
            runmat_accelerate_api::mark_handle_explicit(&explicit);
            for handles in [
                vec![automatic.clone(), explicit.clone()],
                vec![explicit.clone(), automatic.clone()],
            ] {
                let selected = select_resident_output_source(handles, "test")
                    .unwrap()
                    .expect("resident source");
                assert!(same_gpu_handle(&selected, &explicit));
                assert!(runmat_accelerate_api::handle_is_explicit(&selected));
            }
            runmat_accelerate_api::clear_handle_metadata(&automatic);
            runmat_accelerate_api::clear_handle_metadata(&explicit);
        });
    }

    #[test]
    fn resident_output_source_rejects_a_stale_or_wrong_owner() {
        test_support::with_test_provider(|_| {
            let stale = GpuTensorHandle {
                shape: vec![1, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX - 426,
            };
            let error = select_resident_output_source([stale], "test")
                .expect_err("unowned handle must reject");
            assert_eq!(
                error.identifier(),
                Some("RunMat:gpu:ProviderOwnershipMismatch")
            );
            assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        });
    }

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

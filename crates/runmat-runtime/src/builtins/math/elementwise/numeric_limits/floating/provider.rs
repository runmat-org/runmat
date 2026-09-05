use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::FloatingLimitKind;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

pub(super) fn like(
    prototype: &GpuTensorHandle,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let precision = runmat_accelerate_api::handle_precision(prototype)
        .ok_or_else(|| super::super::errors::invalid_floating_prototype(builtin))?;
    let storage = runmat_accelerate_api::handle_storage(prototype);
    if !matches!(
        storage,
        GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
    ) || runmat_accelerate_api::handle_integer_type(prototype).is_some()
        || runmat_accelerate_api::handle_is_logical(prototype)
        || !gpu_helpers::gpu_class_metadata_matches(prototype, Some(precision), None, false)
    {
        return Err(super::super::errors::class(
            builtin,
            "floating-point gpuArray prototype has contradictory class metadata",
        ));
    }
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        super::super::errors::class(
            builtin,
            "floating-point gpuArray prototype has no registered provider",
        )
    })?;
    let shape = vec![1, 1];
    let output = match (precision, storage) {
        (ProviderPrecision::F64, GpuTensorStorage::Real) => gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![super::host::f64_value(kind)], shape.clone())
                .map_err(|error| super::super::errors::internal(builtin, error))?,
        ),
        (ProviderPrecision::F32, GpuTensorStorage::Real) => gpu_helpers::upload_tensor(
            provider,
            &Tensor::from_f32(vec![super::host::f32_value(kind)], shape.clone())
                .map_err(|error| super::super::errors::internal(builtin, error))?,
        ),
        (ProviderPrecision::F64, GpuTensorStorage::ComplexInterleaved) => {
            gpu_helpers::upload_complex_tensor(
                provider,
                &ComplexTensor::new(vec![(super::host::f64_value(kind), 0.0)], shape.clone())
                    .map_err(|error| super::super::errors::internal(builtin, error))?,
            )
            .map_err(|error| error.message().to_string())
        }
        (ProviderPrecision::F32, GpuTensorStorage::ComplexInterleaved) => {
            gpu_helpers::upload_complex_tensor(
                provider,
                &ComplexTensor::from_f32(vec![(super::host::f32_value(kind), 0.0)], shape.clone())
                    .map_err(|error| super::super::errors::internal(builtin, error))?,
            )
            .map_err(|error| error.message().to_string())
        }
    }
    .map_err(|error| {
        super::super::errors::internal(builtin, format!("GPU limit creation failed: {error}"))
    })?;
    validate(
        provider, prototype, output, &shape, storage, precision, builtin,
    )
}

fn validate(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    storage: GpuTensorStorage,
    precision: ProviderPrecision,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let valid = output.shape == shape
        && output.device_id == prototype.device_id
        && !gpu_helpers::same_gpu_handle(prototype, &output)
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == storage
        && runmat_accelerate_api::handle_precision(&output) == Some(precision)
        && gpu_helpers::gpu_class_metadata_matches(&output, Some(precision), None, false);
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&output, &[prototype]);
        return Err(super::super::errors::internal(
            builtin,
            "GPU limit creation returned an invalid provider result",
        ));
    }
    let mut output = output;
    let provenance = runmat_accelerate_api::handle_provenance(prototype)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(gpu_helpers::resident_gpu_value(output))
}

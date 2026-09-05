use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::IntegerLimitKind;
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType, Tensor, Value,
};

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

pub(super) fn like(
    prototype: &GpuTensorHandle,
    kind: IntegerLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let Some(element_type) = runmat_accelerate_api::handle_integer_type(prototype) else {
        return Err(super::super::errors::invalid_integer_prototype(builtin));
    };
    let storage_kind = runmat_accelerate_api::handle_storage(prototype);
    if !matches!(
        storage_kind,
        GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
    ) || runmat_accelerate_api::handle_precision(prototype).is_some()
        || runmat_accelerate_api::handle_is_logical(prototype)
        || !gpu_helpers::gpu_class_metadata_matches(prototype, None, Some(element_type), false)
    {
        return Err(super::super::errors::class(
            builtin,
            "integer gpuArray prototype has contradictory class metadata",
        ));
    }
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        super::super::errors::class(
            builtin,
            "integer gpuArray prototype has no registered provider",
        )
    })?;
    let dtype = NumericDType::from(runmat_types::IntegerClass::from(element_type));
    let storage = IntegerStorage::from_scalar(super::value::scalar(dtype, kind));
    let shape = vec![1usize, 1usize];
    let provenance = runmat_accelerate_api::handle_provenance(prototype)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let metadata = gpu_helpers::snapshot_handle_metadata(prototype);
    let output = match storage_kind {
        GpuTensorStorage::Real => {
            let tensor = Tensor::new_integer(storage, shape.clone())
                .map_err(|error| super::super::errors::internal(builtin, error))?;
            gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| error.to_string())
        }
        GpuTensorStorage::ComplexInterleaved => {
            let imaginary = storage.zeros_like(1);
            let tensor = ComplexTensor::new_integer(
                IntegerComplexStorage::new(storage, imaginary)
                    .map_err(|error| super::super::errors::internal(builtin, error))?,
                shape.clone(),
            )
            .map_err(|error| super::super::errors::internal(builtin, error))?;
            gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|error| error.message().to_string())
        }
    };
    gpu_helpers::restore_handle_metadata(prototype, &metadata);
    let output = output.map_err(|error| {
        super::super::errors::class(builtin, format!("GPU limit creation failed: {error}"))
    })?;
    validate(
        provider,
        prototype,
        output,
        ExpectedOutput {
            shape: &shape,
            storage: storage_kind,
            element_type,
            provenance,
            builtin,
        },
    )
}

struct ExpectedOutput<'a> {
    shape: &'a [usize],
    storage: GpuTensorStorage,
    element_type: runmat_accelerate_api::IntegerElementType,
    provenance: runmat_accelerate_api::GpuHandleProvenance,
    builtin: &'static str,
}

fn validate(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    output: GpuTensorHandle,
    expected: ExpectedOutput<'_>,
) -> BuiltinResult<Value> {
    let valid = output.shape == expected.shape
        && output.device_id == prototype.device_id
        && !gpu_helpers::same_gpu_handle(prototype, &output)
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == expected.storage
        && runmat_accelerate_api::handle_integer_type(&output) == Some(expected.element_type)
        && runmat_accelerate_api::handle_precision(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output)
        && gpu_helpers::gpu_class_metadata_matches(
            &output,
            None,
            Some(expected.element_type),
            false,
        );
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&output, &[prototype]);
        return Err(super::super::errors::class(
            expected.builtin,
            "GPU limit creation returned an invalid provider result",
        ));
    }
    let mut output = output;
    runmat_accelerate_api::set_handle_provenance(&mut output, expected.provenance);
    Ok(gpu_helpers::resident_gpu_value(output))
}

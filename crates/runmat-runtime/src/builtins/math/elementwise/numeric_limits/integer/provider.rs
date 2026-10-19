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
    let storage_kind = prototype
        .descriptor()
        .storage
        .ok_or_else(|| super::super::errors::invalid_integer_prototype(builtin))?;
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
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(prototype);
    let output = match storage_kind {
        GpuTensorStorage::Real => {
            let tensor = Tensor::new_integer(storage, shape.clone())
                .map_err(|error| super::super::errors::internal(builtin, error))?;
            gpu_helpers::upload_tensor_protecting(provider, &tensor, &[prototype])
        }
        GpuTensorStorage::ComplexInterleaved => {
            let imaginary = storage.zeros_like(1);
            let tensor = ComplexTensor::new_integer(
                IntegerComplexStorage::new(storage, imaginary)
                    .map_err(|error| super::super::errors::internal(builtin, error))?,
                shape.clone(),
            )
            .map_err(|error| super::super::errors::internal(builtin, error))?;
            gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &[prototype])
        }
    }?;
    validate(
        provider,
        prototype,
        output,
        ExpectedOutput {
            shape: &shape,
            storage: storage_kind,
            element_type,
            builtin,
        },
    )
}

struct ExpectedOutput<'a> {
    shape: &'a [usize],
    storage: GpuTensorStorage,
    element_type: runmat_accelerate_api::IntegerElementType,
    builtin: &'static str,
}

fn validate(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    output: GpuTensorHandle,
    expected: ExpectedOutput<'_>,
) -> BuiltinResult<Value> {
    let valid = output.shape() == expected.shape
        && output.device_id() == prototype.device_id()
        && !prototype.shares_allocation_with(&output)
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && output.descriptor().storage == Some(expected.storage)
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
        return Err(gpu_helpers::reject_gpu_output(
            output,
            super::super::errors::class(
                expected.builtin,
                "GPU limit creation returned an invalid provider result",
            ),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [prototype])?;
    gpu_helpers::resident_gpu_value(output)
}

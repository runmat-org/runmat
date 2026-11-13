//! Owner-aware provider input preparation, Unsupported host continuation,
//! and validation of the one admitted output before receipt publication.

use super::{
    apply_imfilter_host, fallback_accumulation, gpu_helpers, imfilter_error_with_detail,
    imfilter_map_error, tensor, BuiltinResult, GpuTensorHandle, GpuTensorStorage, ImageOutputClass,
    ImfilterOptions, ProviderPrecision, Tensor, Value, IMFILTER_BUILTIN, IMFILTER_ERROR_INTERNAL,
    IMFILTER_ERROR_INVALID_INPUT,
};

pub(super) fn valid_host_only_restore(
    value: &Value,
    image: &GpuTensorHandle,
    resident_kernel: Option<&GpuTensorHandle>,
    expected_shape: &[usize],
    expected_class: ImageOutputClass,
    owner: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let Value::GpuTensor(output) = value else {
        return false;
    };
    output.shape() == expected_shape
        && output.device_id() == image.device_id()
        && output.allocation_lease().validate_live().is_ok()
        && !output.shares_allocation_with(image)
        && resident_kernel.is_none_or(|kernel| !output.shares_allocation_with(kernel))
        && runmat_accelerate_api::handle_is_explicit(output)
        && ImageOutputClass::from_gpu_handle(output).ok() == Some(expected_class)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|actual| runmat_accelerate_api::same_provider_instance(actual, owner))
}

pub(super) fn upload_kernel(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: &Tensor,
    image: &GpuTensorHandle,
    resident_kernel: Option<&GpuTensorHandle>,
    precision: ProviderPrecision,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView, NumericUploadError};
    let values = tensor.as_f64_slice().ok_or_else(|| {
        imfilter_error_with_detail(&IMFILTER_ERROR_INVALID_INPUT, "H must be double")
    })?;
    // Preserve the existing kernel adaptation to the admitted execution
    // precision, but declare the actual upload payload instead of relabeling.
    let single;
    let data = if precision == ProviderPrecision::F32 {
        single = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
        HostNumericDataView::F32(&single)
    } else {
        HostNumericDataView::F64(values)
    };
    let protected = std::iter::once(image)
        .chain(resident_kernel)
        .collect::<Vec<_>>();
    match runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &HostNumericTensorView {
            data,
            shape: tensor.shape(),
            storage: GpuTensorStorage::Real,
        },
        &protected,
    ) {
        Ok(output) => Ok(Some(output)),
        Err(NumericUploadError::Provider(error)) if error.is_unsupported() => Ok(None),
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            imfilter_error_with_detail(
                &IMFILTER_ERROR_INTERNAL,
                format!("failed to upload H to A's provider: {error}"),
            ),
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        )),
    }
}

pub(super) fn preflight_resident_input(handle: &GpuTensorHandle, name: &str) -> BuiltinResult<()> {
    handle
        .allocation_lease()
        .validate_live()
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                imfilter_error_with_detail(
                    &IMFILTER_ERROR_INVALID_INPUT,
                    format!("{name} allocation is unavailable"),
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(source),
            )
        })?;
    if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
        let mut error = imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            format!("no exact provider owns {name}"),
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(())
}

pub(super) async fn imfilter_gpu_host_fallback(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    image_handle: &GpuTensorHandle,
    kernel: &Tensor,
    options: &ImfilterOptions,
    resident_kernel: Option<&GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(image_handle);
    let download =
        gpu_helpers::download_value_preserving_residency_async(provider, image_handle).await;
    let image = download.map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_INPUT))?;
    let output_class = ImageOutputClass::from_gpu_handle(image_handle)?;
    let image_tensor = tensor::value_into_tensor_for(IMFILTER_BUILTIN, image)
        .map_err(|err| imfilter_error_with_detail(&IMFILTER_ERROR_INVALID_INPUT, err))?;
    let accumulation = fallback_accumulation(
        output_class,
        runmat_accelerate_api::handle_is_explicit(image_handle),
    );
    let result = apply_imfilter_host(
        &image_tensor,
        kernel,
        options,
        output_class,
        accumulation,
        IMFILTER_BUILTIN,
    )
    .map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_INPUT))?;
    if !runmat_accelerate_api::handle_is_explicit(image_handle) {
        return Ok(result);
    }
    let protected = resident_kernel.into_iter().collect::<Vec<_>>();
    let restored = gpu_helpers::restore_class_preserving_value_protecting(
        image_handle,
        result,
        IMFILTER_BUILTIN,
        &protected,
    )?;
    if matches!(restored, Value::GpuTensor(_)) {
        Ok(restored)
    } else {
        Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INTERNAL,
            "explicit gpuArray result could not be restored to its owning provider",
        ))
    }
}

pub(super) fn validate_conditional_restore(
    result: BuiltinResult<Value>,
    selected: Option<&mut crate::context::provider_operation::SelectedProviderOperation>,
    image: &GpuTensorHandle,
    kernel: Option<&GpuTensorHandle>,
    expected_shape: &[usize],
    precision: ProviderPrecision,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Value> {
    match result {
        Ok(Value::GpuTensor(output))
            if selected.is_some_and(|selection| {
                !valid_provider_output(&output, image, kernel, expected_shape, precision, provider)
                    || !selection.admits_conditional_provider_upload(&output)
            }) =>
        {
            let mut error = imfilter_error_with_detail(
                &IMFILTER_ERROR_INTERNAL,
                "conditional restore returned a result outside its admitted physical route",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(gpu_helpers::reject_gpu_output(output, error))
        }
        other => other,
    }
}

pub(super) fn valid_provider_output(
    output: &GpuTensorHandle,
    image: &GpuTensorHandle,
    kernel: Option<&GpuTensorHandle>,
    expected_shape: &[usize],
    expected_precision: ProviderPrecision,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape() == expected_shape
        && output.device_id() == image.device_id()
        && !output.shares_allocation_with(image)
        && !kernel.is_some_and(|kernel| output.shares_allocation_with(kernel))
        && output.allocation_lease().validate_live().is_ok()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, Some(expected_precision), None, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{GpuHandleProvenance, ThreadProviderGuard};
    use runmat_value::NumericDType;

    #[test]
    fn explicit_restore_validator_rejects_alias_class_shape_and_foreign_owner() {
        let owner: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let foreign: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
        let _selected = ThreadProviderGuard::set(Some(owner));
        let byte =
            Tensor::new_with_dtype(vec![7.0], vec![1, 1], NumericDType::U8).expect("uint8 fixture");
        let image = gpu_helpers::upload_tensor(owner, &byte)
            .expect("original A")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit original A");
        let valid = gpu_helpers::upload_tensor(owner, &byte)
            .expect("new output")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit new output");
        assert!(valid_host_only_restore(
            &Value::GpuTensor(valid.clone()),
            &image,
            None,
            &[1, 1],
            ImageOutputClass::Integer(NumericDType::U8),
            owner,
        ));
        assert!(!valid_host_only_restore(
            &Value::GpuTensor(image.clone()),
            &image,
            None,
            &[1, 1],
            ImageOutputClass::Integer(NumericDType::U8),
            owner,
        ));
        assert!(!valid_host_only_restore(
            &Value::GpuTensor(valid.clone()),
            &image,
            None,
            &[1, 1],
            ImageOutputClass::Integer(NumericDType::U16),
            owner,
        ));
        assert!(!valid_host_only_restore(
            &Value::GpuTensor(valid.clone()),
            &image,
            None,
            &[2, 1],
            ImageOutputClass::Integer(NumericDType::U8),
            owner,
        ));
        let foreign_output = gpu_helpers::upload_tensor(foreign, &byte)
            .expect("foreign output")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit foreign output");
        assert!(!valid_host_only_restore(
            &Value::GpuTensor(foreign_output),
            &image,
            None,
            &[1, 1],
            ImageOutputClass::Integer(NumericDType::U8),
            owner,
        ));
        assert!(image.allocation_lease().validate_live().is_ok());
        assert!(valid.allocation_lease().validate_live().is_ok());
    }
}

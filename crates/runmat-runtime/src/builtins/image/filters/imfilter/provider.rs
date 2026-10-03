use super::kernel::fallback_accumulation;
use super::*;

pub(super) async fn imfilter_gpu(
    image_handle: GpuTensorHandle,
    kernel_value: Value,
    options: ImfilterOptions,
) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        let kernel_is_wgpu = matches!(kernel_value, Value::GpuTensor(ref h) if h.device_id() != 0);
        if kernel_is_wgpu || image_handle.device_id() != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    preflight_resident_input(&image_handle, "A")?;
    if let Value::GpuTensor(kernel) = &kernel_value {
        preflight_resident_input(kernel, "H")?;
    }
    let output_class = ImageOutputClass::from_gpu_handle(&image_handle)?;
    let provider = gpu_helpers::exact_provider_for_handle(&image_handle).ok_or_else(|| {
        imfilter_error_with_detail(&IMFILTER_ERROR_INTERNAL, "no acceleration provider owns A")
    })?;
    let (kernel_tensor, resident_kernel) = match kernel_value {
        Value::GpuTensor(handle) => (download_kernel_tensor(&handle).await?, Some(handle)),
        other => (host_kernel_tensor(other)?, None),
    };
    let expected_shape = build_imfilter_plan(
        image_handle.shape(),
        &kernel_tensor,
        &options,
        IMFILTER_BUILTIN,
    )?
    .final_shape;

    // Native providers currently expose floating imfilter kernels. Integer and
    // logical images take an exact owner-aware host path, then explicit inputs
    // are restored to their owner with their original class metadata.
    let Some(expected_precision) = output_class.provider_precision() else {
        return imfilter_gpu_host_fallback(
            provider,
            &image_handle,
            &kernel_tensor,
            &options,
            resident_kernel.as_ref(),
        )
        .await;
    };
    let kernel_precision = match floating_accumulation(
        expected_precision,
        runmat_accelerate_api::handle_is_explicit(&image_handle),
    ) {
        FilterAccumulation::HostDouble => ProviderPrecision::F64,
        FilterAccumulation::GpuSingle => ProviderPrecision::F32,
    };
    // Retain both originals through upload, execution, fallback and suspension.
    let _image_residency = runmat_accelerate_api::ResidencyRestoreGuard::new(&image_handle);
    let _kernel_residency = resident_kernel
        .as_ref()
        .map(runmat_accelerate_api::ResidencyRestoreGuard::new);
    let same_owner_kernel = resident_kernel.as_ref().filter(|handle| {
        gpu_helpers::exact_provider_for_handle(handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            && runmat_accelerate_api::handle_precision(handle) == Some(kernel_precision)
    });
    let kernel_handle = match same_owner_kernel {
        Some(handle) => handle.clone(),
        None => match upload_kernel(
            provider,
            &kernel_tensor,
            &image_handle,
            resident_kernel.as_ref(),
            kernel_precision,
        )? {
            Some(uploaded) => uploaded,
            None => {
                return imfilter_gpu_host_fallback(
                    provider,
                    &image_handle,
                    &kernel_tensor,
                    &options,
                    resident_kernel.as_ref(),
                )
                .await
            }
        },
    };
    let provider_result = provider
        .imfilter(&image_handle, &kernel_handle, &options)
        .await;

    match provider_result {
        Ok(mut output) => {
            if !valid_provider_output(
                &output,
                &image_handle,
                &kernel_handle,
                &expected_shape,
                expected_precision,
                provider,
            ) || resident_kernel
                .as_ref()
                .is_some_and(|original| output.shares_allocation_with(original))
            {
                let mut error = imfilter_error_with_detail(
                    &IMFILTER_ERROR_INTERNAL,
                    "provider returned an invalid imfilter result",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            gpu_helpers::propagate_output_provenance(&mut output, [&image_handle])?;
            gpu_helpers::resident_gpu_value(output)
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            imfilter_gpu_host_fallback(
                provider,
                &image_handle,
                &kernel_tensor,
                &options,
                resident_kernel.as_ref(),
            )
            .await
        }
        Err(error) => Err(kernel_error::with_provider_source(
            imfilter_error_with_detail(
                &IMFILTER_ERROR_INTERNAL,
                format!("provider execution failed: {error}"),
            ),
            error.into(),
        )),
    }
}

fn upload_kernel(
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
            shape: &tensor.shape,
            storage: GpuTensorStorage::Real,
        },
        &protected,
    ) {
        Ok(output) => Ok(Some(output)),
        Err(NumericUploadError::Provider(error))
            if gpu_helpers::provider_hook_is_unsupported(&error) =>
        {
            Ok(None)
        }
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            imfilter_error_with_detail(
                &IMFILTER_ERROR_INTERNAL,
                format!("failed to upload H to A's provider: {error}"),
            ),
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        )),
    }
}

fn preflight_resident_input(handle: &GpuTensorHandle, name: &str) -> BuiltinResult<()> {
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
    if gpu_helpers::exact_provider_for_handle(handle).is_none() {
        let mut error = imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            format!("no exact provider owns {name}"),
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(())
}

async fn imfilter_gpu_host_fallback(
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

fn valid_provider_output(
    output: &GpuTensorHandle,
    image: &GpuTensorHandle,
    kernel: &GpuTensorHandle,
    expected_shape: &[usize],
    expected_precision: ProviderPrecision,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape() == expected_shape
        && output.device_id() == image.device_id()
        && !output.shares_allocation_with(image)
        && !output.shares_allocation_with(kernel)
        && output.allocation_lease().validate_live().is_ok()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, Some(expected_precision), None, false)
}

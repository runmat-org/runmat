use super::super::placement;
use super::kernel::fallback_accumulation;
use super::*;
mod route;
use route::{select_provider_route, valid_public_kernel};
mod publication;
use publication::{
    imfilter_gpu_host_fallback, preflight_resident_input, upload_kernel, valid_host_only_restore,
    valid_provider_output, validate_conditional_restore,
};

pub(super) async fn imfilter_gpu(
    runtime: Option<&crate::context::RuntimeContext>,
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
    let host_only =
        output_class.provider_precision().is_none() && valid_public_kernel(&kernel_value);
    let (host_sink, host_upload) = if host_only {
        let kernel_shape =
            imfilter_value_shape(&kernel_value).expect("admitted public H has numeric geometry");
        let readbacks = std::iter::once(image_handle.clone())
            .chain(match &kernel_value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if runmat_accelerate_api::handle_is_explicit(&image_handle) {
            let element_type = if output_class == ImageOutputClass::Logical {
                runmat_accelerate_api::ProviderElementType::Logical
            } else {
                runmat_accelerate_api::ProviderElementType::from(
                    runmat_accelerate_api::handle_integer_type(&image_handle)
                        .expect("admitted nonfloating A has integer storage"),
                )
            };
            (
                None,
                placement::select_host_only_upload(
                    runtime,
                    IMFILTER_BUILTIN,
                    &image_handle,
                    &kernel_shape,
                    &options,
                    &readbacks,
                    element_type,
                )?,
            )
        } else {
            (
                placement::select_host_only_sink(
                    runtime,
                    IMFILTER_BUILTIN,
                    &image_handle,
                    &kernel_shape,
                    &options,
                    &readbacks,
                    u64::try_from(image_handle.physical().element_type().element_size())
                        .expect("element byte width fits u64"),
                )?,
                None,
            )
        }
    } else {
        (None, None)
    };
    let (mut selected, planned_query) = select_provider_route(
        runtime,
        &image_handle,
        &kernel_value,
        &options,
        output_class,
        provider,
    )?;
    let (kernel_tensor, resident_kernel) = match kernel_value {
        Value::GpuTensor(handle) => {
            let downloaded = runmat_accelerate_api::observation::observe_provider_future(
                host_upload
                    .as_ref()
                    .map(|receipt| receipt.recorder())
                    .or_else(|| selected.as_ref().map(|selection| selection.recorder())),
                download_kernel_tensor(&handle),
            )
            .await?;
            (downloaded, Some(handle))
        }
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
        let result = runmat_accelerate_api::observation::observe_provider_future(
            host_upload.as_ref().map(|receipt| receipt.recorder()),
            imfilter_gpu_host_fallback(
                provider,
                &image_handle,
                &kernel_tensor,
                &options,
                resident_kernel.as_ref(),
            ),
        )
        .await;
        let result = if runmat_accelerate_api::handle_is_explicit(&image_handle)
            && result.as_ref().is_ok_and(|value| {
                !valid_host_only_restore(
                    value,
                    &image_handle,
                    resident_kernel.as_ref(),
                    &expected_shape,
                    output_class,
                    provider,
                ) || host_upload.as_ref().is_some_and(|receipt| match value {
                    Value::GpuTensor(output) => !receipt.admits_published_outputs(&[output]),
                    _ => true,
                })
            }) {
            let mut error = imfilter_error_with_detail(
                &IMFILTER_ERROR_INTERNAL,
                "host-filter result did not match admitted owner, class, shape or physical allocation",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(error)
        } else {
            result
        };
        if let Some(receipt) = host_sink {
            receipt.finish(result.is_ok());
        }
        if let Some(receipt) = host_upload {
            receipt.finish(if result.is_ok() {
                Ok(true)
            } else {
                Err(runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed)
            });
        }
        return result;
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
        None => match runmat_accelerate_api::observation::with_provider_observation(
            selected.as_ref().map(|selection| selection.recorder()),
            || {
                upload_kernel(
                    provider,
                    &kernel_tensor,
                    &image_handle,
                    resident_kernel.as_ref(),
                    kernel_precision,
                )
            },
        )? {
            Some(uploaded) => uploaded,
            None => {
                let explicit = runmat_accelerate_api::handle_is_explicit(&image_handle);
                if explicit {
                    if let Some(selection) = selected.as_mut() {
                        selection.begin_preparation_provider_upload();
                    }
                }
                let result = runmat_accelerate_api::observation::observe_provider_future(
                    selected.as_ref().map(|selection| selection.recorder()),
                    imfilter_gpu_host_fallback(
                        provider,
                        &image_handle,
                        &kernel_tensor,
                        &options,
                        resident_kernel.as_ref(),
                    ),
                )
                .await;
                let result = validate_conditional_restore(
                    result,
                    selected.as_ref(),
                    &image_handle,
                    resident_kernel.as_ref(),
                    &expected_shape,
                    expected_precision,
                    provider,
                );
                if let Some(selection) = selected {
                    match (&result, explicit) {
                        (Ok(Value::GpuTensor(_)), true) => {
                            selection.finish_preparation_provider_upload()
                        }
                        (Ok(_), false) => selection.finish_preparation_host_fallback(),
                        (Ok(_), true) => selection.finish(Err(
                            runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed,
                        )),
                        (Err(_), _) => selection.finish(Err(
                            runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                        )),
                    }
                }
                return result;
            }
        },
    };
    if planned_query.as_ref().is_some_and(|(query, bytes)| {
        !placement::prepared_hook_matches_realized(
            query,
            &image_handle,
            &kernel_handle,
            &options,
            *bytes,
        )
    }) {
        if let Some(selection) = selected {
            selection.finish(Err(
                runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed,
            ));
        }
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INTERNAL,
            "prepared H did not match the admitted provider representation",
        ));
    }
    let provider_result = runmat_accelerate_api::observation::observe_provider_future(
        selected.as_ref().map(|selection| selection.recorder()),
        provider.imfilter(&image_handle, &kernel_handle, &options),
    )
    .await;

    match provider_result {
        Ok(mut output) => {
            if !valid_provider_output(
                &output,
                &image_handle,
                Some(&kernel_handle),
                &expected_shape,
                expected_precision,
                provider,
            ) || resident_kernel
                .as_ref()
                .is_some_and(|original| output.shares_allocation_with(original))
            {
                if let Some(selection) = selected {
                    selection.finish(Err(runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed));
                }
                let mut error = imfilter_error_with_detail(
                    &IMFILTER_ERROR_INTERNAL,
                    "provider returned an invalid imfilter result",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            let result = gpu_helpers::propagate_output_provenance(&mut output, [&image_handle])
                .and_then(|()| gpu_helpers::resident_gpu_value(output));
            if let Some(selection) = selected {
                selection.finish(if result.is_ok() { Ok(true) } else { Err(runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed) });
            }
            result
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            let explicit = runmat_accelerate_api::handle_is_explicit(&image_handle);
            if explicit {
                if let Some(selection) = selected.as_mut() {
                    selection.begin_conditional_provider_upload();
                }
            }
            let result = runmat_accelerate_api::observation::observe_provider_future(
                selected.as_ref().map(|selection| selection.recorder()),
                imfilter_gpu_host_fallback(
                    provider,
                    &image_handle,
                    &kernel_tensor,
                    &options,
                    resident_kernel.as_ref(),
                ),
            )
            .await;
            let result = validate_conditional_restore(
                result,
                selected.as_ref(),
                &image_handle,
                Some(&kernel_handle),
                &expected_shape,
                expected_precision,
                provider,
            );
            if let Some(selection) = selected {
                match (&result, explicit) {
                    (Ok(Value::GpuTensor(_)), true) => {
                        selection.finish_conditional_provider_upload()
                    }
                    (Ok(_), false) => selection.finish_conditional_host_fallback(),
                    (Ok(_), true) => selection.finish(Err(
                        runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed,
                    )),
                    (Err(_), _) => selection.finish(Err(
                        runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                    )),
                }
            }
            result
        }
        Err(error) => {
            if let Some(selection) = selected {
                selection.finish(Err(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                ));
            }
            Err(kernel_error::with_provider_source(
                imfilter_error_with_detail(
                    &IMFILTER_ERROR_INTERNAL,
                    format!("provider execution failed: {error}"),
                ),
                error.into(),
            ))
        }
    }
}

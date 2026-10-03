use super::super::placement;
use super::kernel::fallback_accumulation;
use super::*;

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
    let host_sink = if output_class.provider_precision().is_none()
        && !runmat_accelerate_api::handle_is_explicit(&image_handle)
        && valid_public_kernel(&kernel_value)
    {
        let kernel_shape =
            imfilter_value_shape(&kernel_value).expect("admitted public H has numeric geometry");
        let readbacks = std::iter::once(image_handle.clone())
            .chain(match &kernel_value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        placement::select_host_only_sink(
            runtime,
            IMFILTER_BUILTIN,
            &image_handle,
            &kernel_shape,
            &options,
            &readbacks,
            u64::try_from(image_handle.physical().element_type().element_size())
                .expect("element byte width fits u64"),
        )?
    } else {
        None
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
                selected.as_ref().map(|selection| selection.recorder()),
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
        let result = imfilter_gpu_host_fallback(
            provider,
            &image_handle,
            &kernel_tensor,
            &options,
            resident_kernel.as_ref(),
        )
        .await;
        if let Some(receipt) = host_sink {
            receipt.finish(result.is_ok());
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

fn projected_public_kernel(
    value: &Value,
    precision: ProviderPrecision,
) -> Option<(
    runmat_accelerate_api::ProviderRepresentation,
    runmat_execution::ResourceByteBound,
)> {
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
        ProviderStorage,
    };
    let shape = match value {
        Value::Num(_) => vec![1, 1],
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => {
            tensor.shape.clone()
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F64) =>
        {
            handle.shape().to_vec()
        }
        _ => return None,
    };
    let len = shape.iter().try_fold(1_u64, |count, &extent| {
        count.checked_mul(u64::try_from(extent).ok()?)
    })?;
    let scratch = len
        .checked_mul(if precision == ProviderPrecision::F32 {
            4
        } else {
            0
        })
        .and_then(|bytes| bytes.checked_add(u64::from(matches!(value, Value::Num(_))) * 8))
        .map(runmat_execution::ResourceByteBound::Known)
        .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ));
    Some((
        ProviderRepresentation {
            element_type: ProviderElementType::from(precision),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape
                .into_iter()
                .map(|extent| u64::try_from(extent).ok())
                .collect::<Option<Vec<_>>>()?,
            residency: ProviderResidency::Device,
        },
        scratch,
    ))
}

fn select_provider_route(
    runtime: Option<&crate::context::RuntimeContext>,
    image: &GpuTensorHandle,
    kernel: &Value,
    options: &ImfilterOptions,
    output_class: ImageOutputClass,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<(
    Option<crate::context::provider_operation::SelectedProviderOperation>,
    Option<(runmat_accelerate_api::ProviderFeasibilityQuery, u64)>,
)> {
    let Some(image_precision) = output_class.provider_precision() else {
        return Ok((None, None));
    };
    let required_kernel_precision = match floating_accumulation(
        image_precision,
        runmat_accelerate_api::handle_is_explicit(image),
    ) {
        FilterAccumulation::HostDouble => ProviderPrecision::F64,
        FilterAccumulation::GpuSingle => ProviderPrecision::F32,
    };
    let valid_public_kernel = valid_public_kernel(kernel);
    if !valid_public_kernel {
        return Ok((None, None));
    }
    if let Value::GpuTensor(handle) = kernel {
        if required_kernel_precision == ProviderPrecision::F64
            && gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            let (scratch, result_bytes) = placement::conditional_host_bounds(
                IMFILTER_BUILTIN,
                image.shape(),
                handle.shape(),
                options,
                image_precision == ProviderPrecision::F32,
                false,
                false,
            );
            let fallback_readbacks = [image];
            let explicit = runmat_accelerate_api::handle_is_explicit(image);
            let upload_scratch = placement::host_scratch_excluding_result(scratch, result_bytes)?;
            let conditional = if explicit {
                placement::ConditionalFilterRoute::Provider(placement::ConditionalProviderRoute {
                    operation_scratch: upload_scratch,
                    readbacks: &fallback_readbacks,
                })
            } else {
                placement::ConditionalFilterRoute::Host(placement::ConditionalHostRoute {
                    result_bytes,
                    operation_scratch: scratch,
                    readbacks: &fallback_readbacks,
                })
            };
            return Ok((
                placement::select_direct(
                    placement::FilterRouteRequest {
                        runtime,
                        semantic_operation: IMFILTER_BUILTIN,
                        image,
                        options,
                        conditional: Some(conditional),
                    },
                    handle,
                    true,
                )?,
                None,
            ));
        }
    }
    let Some((projected, scratch)) = projected_public_kernel(kernel, required_kernel_precision)
    else {
        return Ok((None, None));
    };
    let original = match kernel {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    };
    let kernel_shape = match kernel {
        Value::Num(_) => vec![1, 1],
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::GpuTensor(handle) => handle.shape().to_vec(),
        _ => return Ok((None, None)),
    };
    let (fallback_scratch, result_bytes) = placement::conditional_host_bounds(
        IMFILTER_BUILTIN,
        image.shape(),
        &kernel_shape,
        options,
        image_precision == ProviderPrecision::F32,
        false,
        true,
    );
    let fallback_readbacks = [image];
    let explicit = runmat_accelerate_api::handle_is_explicit(image);
    let upload_scratch = placement::host_scratch_excluding_result(fallback_scratch, result_bytes)?;
    let conditional = if explicit {
        placement::ConditionalFilterRoute::Provider(placement::ConditionalProviderRoute {
            operation_scratch: upload_scratch,
            readbacks: &fallback_readbacks,
        })
    } else {
        placement::ConditionalFilterRoute::Host(placement::ConditionalHostRoute {
            result_bytes,
            operation_scratch: fallback_scratch,
            readbacks: &fallback_readbacks,
        })
    };
    Ok(
        match placement::select_prepared(
            placement::FilterRouteRequest {
                runtime,
                semantic_operation: IMFILTER_BUILTIN,
                image,
                options,
                conditional: Some(conditional),
            },
            original,
            projected,
            scratch,
        )? {
            Some(prepared) => (
                Some(prepared.selection),
                Some((prepared.query, prepared.expected_kernel_requested_bytes)),
            ),
            None => (None, None),
        },
    )
}

fn valid_public_kernel(kernel: &Value) -> bool {
    match kernel {
        Value::Num(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F64,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F64)
                && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && gpu_helpers::gpu_class_metadata_matches(
                    handle,
                    Some(ProviderPrecision::F64),
                    None,
                    false,
                )
        }
        _ => false,
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

fn validate_conditional_restore(
    result: BuiltinResult<Value>,
    selected: Option<&crate::context::provider_operation::SelectedProviderOperation>,
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

fn valid_provider_output(
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

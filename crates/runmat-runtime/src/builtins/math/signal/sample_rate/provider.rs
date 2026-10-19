use super::*;

pub(super) async fn resample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    p: usize,
    q: usize,
    options: &ResampleOptions,
) -> BuiltinResult<Option<ResampleEval>> {
    // Logical samples are admitted by the host conversion contract and become
    // double. A class-preserving provider filter cannot publish that result.
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Ok(None);
    }
    let Some(provider) = resolved_actual_downsample_owner(handle) else {
        return Ok(None);
    };
    let storage = runmat_accelerate_api::handle_storage(handle);
    let handle_len = checked_product(handle.shape())
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut shape = canonical_shape(handle.shape().to_vec(), handle_len);
    let dim = options
        .dim
        .unwrap_or_else(|| first_non_singleton_dim(&shape));
    if dim >= shape.len() {
        shape.resize(dim + 1, 1);
    }

    let input_len = shape[dim];
    let output_len = resample_output_len(input_len, p, q)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    if input_len == 0 || output_len == 0 {
        return match provider.zeros_with_storage(&output_shape, storage) {
            Ok(output)
                if valid_class_preserving_gpu_output(
                    provider,
                    handle,
                    &output,
                    &output_shape,
                    true,
                ) =>
            {
                Ok(Some(ResampleEval {
                    y: wrap_resampled_gpu(output)?,
                    filter: options.filter.clone(),
                }))
            }
            Ok(output) => Err(gpu_helpers::reject_gpu_output(
                output,
                sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_GATHER_FAILED,
                    "provider returned an incompatible empty resample result",
                ),
            )),
            Err(source) => {
                let error = sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_GATHER_FAILED,
                    "empty resample allocation failed without certified pre-effect status",
                );
                Err(crate::runtime_error::with_provider_execution_source(
                    error,
                    runmat_accelerate_api::ProviderExecutionError::from(source)
                        .with_unproven_effects(),
                ))
            }
        };
    }

    let filter_len = options.filter.len();
    let delay = filter_len / 2;
    let gather_span_len = output_len
        .checked_sub(1)
        .and_then(|last| last.checked_mul(q))
        .and_then(|last| last.checked_add(delay))
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let scatter_span_len = input_len
        .checked_sub(1)
        .and_then(|last| last.checked_mul(p))
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let padded_len = gather_span_len.max(scatter_span_len);
    let mut padded_shape = shape.clone();
    padded_shape[dim] = padded_len;

    let Some(scatter_indices) =
        upsample_linear_indices(&shape, dim, input_len, padded_len, p, 0, RESAMPLE_NAME)?
    else {
        return Ok(None);
    };
    let Some(gather_indices) = downsample_linear_indices(
        &padded_shape,
        dim,
        padded_len,
        output_len,
        q,
        delay,
        RESAMPLE_NAME,
    )?
    else {
        return Ok(None);
    };

    let padded = match provider.zeros_with_storage(&padded_shape, storage) {
        Ok(handle) => handle,
        Err(source) => {
            let error = sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "resample allocation failed without certified pre-effect status",
            );
            return Err(crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects(),
            ));
        }
    };
    if !valid_class_preserving_gpu_output(provider, handle, &padded, &padded_shape, true) {
        return Err(gpu_helpers::reject_gpu_output(
            padded,
            sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "provider returned an incompatible resample allocation",
            ),
        ));
    }
    provider
        .scatter_linear(&padded, &scatter_indices, handle)
        .map_err(|source| {
            let error = sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "resample scatter failed after allocation",
            );
            crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).after_effect(),
            )
        })?;

    let filter_shape = [filter_len, 1usize];
    let b_handle = if let Some(handle) = options.filter.same_provider_handle(handle, provider) {
        handle.clone()
    } else {
        let filter = options.filter.host_values().await?;
        let mut protected = vec![handle, &padded];
        if let ResampleFilter::Gpu {
            handle: resident, ..
        } = &options.filter
        {
            protected.push(resident);
        }
        let transfer = runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F64(&filter),
            shape: &filter_shape,
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        };
        match runmat_accelerate_api::upload_numeric_protecting(provider, &transfer, &protected) {
            Ok(handle) => handle,
            Err(source) => {
                let error = sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_GATHER_FAILED,
                    "filter upload failed after resample scatter",
                );
                return Err(crate::runtime_error::with_provider_execution_source(
                    error,
                    source.into_execution_error().after_effect(),
                ));
            }
        }
    };
    if b_handle.device_id() != provider.device_id()
        || resolved_actual_downsample_owner(&b_handle)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(gpu_helpers::reject_gpu_output(
            b_handle,
            sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "filter owner changed during resample",
            ),
        ));
    }
    let mut filter_result = match provider
        .iir_filter(
            runmat_accelerate_api::ProviderIirCoefficientSource::UnitDenominatorFir {
                numerator: &b_handle,
            },
            &padded,
            runmat_accelerate_api::ProviderIirFilterOptions { dim, zi: None },
        )
        .await
    {
        Ok(result) => result,
        Err(source) => {
            let error = sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "resample filter failed after scatter",
            );
            return Err(crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).after_effect(),
            ));
        }
    };
    if filter_result.output.shares_allocation_with(&padded)
        || filter_result.output.shares_allocation_with(&b_handle)
        || !valid_class_preserving_gpu_output(
            provider,
            handle,
            &filter_result.output,
            &padded_shape,
            true,
        )
    {
        let error = sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_GATHER_FAILED,
            "provider returned an incompatible filtered signal",
        );
        let error = gpu_helpers::reject_gpu_output(filter_result.output, error);
        return Err(match filter_result.final_state {
            Some(state) => gpu_helpers::reject_gpu_output(state, error),
            None => error,
        });
    }

    if let Some(state) = filter_result.final_state.take() {
        state.release().map_err(|source| {
            let error = sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "unused provider filter state could not be retired",
            );
            crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from_source(source).after_effect(),
            )
        })?;
    }

    let output = provider.gather_linear(&filter_result.output, &gather_indices, &output_shape);

    match output {
        Ok(output)
            if !output.shares_allocation_with(&padded)
                && !output.shares_allocation_with(&filter_result.output)
                && !output.shares_allocation_with(&b_handle)
                && valid_class_preserving_gpu_output(
                    provider,
                    handle,
                    &output,
                    &output_shape,
                    true,
                ) =>
        {
            Ok(Some(ResampleEval {
                y: wrap_resampled_gpu(output)?,
                filter: options.filter.clone(),
            }))
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "provider returned an incompatible resample result",
            ),
        )),
        Err(source) => {
            let error = sample_error_with_detail(
                RESAMPLE_NAME,
                &SAMPLE_ERROR_GATHER_FAILED,
                "resample gather failed after filtering",
            );
            Err(crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).after_effect(),
            ))
        }
    }
}

pub(super) fn upsample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let handle_len = checked_product(handle.shape())
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let shape = canonical_shape(handle.shape().to_vec(), handle_len);
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, SampleOp::Up)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    if factor == 1 && phase == 0 {
        resolved_actual_downsample_owner(handle).ok_or_else(|| {
            sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_GATHER_FAILED,
                "identity input has no valid live provider owner",
            )
        })?;
        return Ok(Some(wrap_sample_rate_gpu(handle, handle.clone())?));
    }

    let Some(provider) = resolved_actual_downsample_owner(handle) else {
        return Ok(None);
    };
    let Some(indices) =
        upsample_linear_indices(&shape, dim, input_len, output_len, factor, phase, builtin)?
    else {
        return Ok(None);
    };
    let allocation = provider.zeros_like(handle, &output_shape);
    let output = match allocation {
        Ok(Some(output)) => output,
        Ok(None) => return Ok(None),
        Err(source) => {
            let error = sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_GATHER_FAILED,
                "upsample allocation failed without certified pre-effect status",
            );
            return Err(crate::runtime_error::with_provider_execution_source(
                error,
                source.with_unproven_effects(),
            ));
        }
    };
    if !valid_class_preserving_gpu_output(provider, handle, &output, &output_shape, true) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_GATHER_FAILED,
                "provider returned an incompatible upsample allocation",
            ),
        ));
    }
    match provider.scatter_linear(&output, &indices, handle) {
        Ok(()) => Ok(Some(wrap_sample_rate_gpu(handle, output)?)),
        Err(source) => {
            let error = sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_GATHER_FAILED,
                "upsample scatter failed after output allocation",
            );
            Err(crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).after_effect(),
            ))
        }
    }
}

pub(super) fn downsample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let handle_len = checked_product(handle.shape())
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let shape = canonical_shape(handle.shape().to_vec(), handle_len);
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, SampleOp::Down)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    let provider = resolved_actual_downsample_owner(handle).ok_or_else(|| {
        sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_GATHER_FAILED,
            "GPU input has no proven owner for its device",
        )
    })?;

    if factor == 1 && phase == 0 {
        return Ok(Some(wrap_sample_rate_gpu(handle, handle.clone())?));
    }

    let Some(indices) =
        downsample_linear_indices(&shape, dim, input_len, output_len, factor, phase, builtin)?
    else {
        return Ok(None);
    };

    match provider.gather_linear(handle, &indices, &output_shape) {
        Ok(output) => {
            let valid =
                valid_class_preserving_gpu_output(provider, handle, &output, &output_shape, true);
            if valid {
                Ok(Some(wrap_sample_rate_gpu(handle, output)?))
            } else {
                Err(gpu_helpers::reject_gpu_output(
                    output,
                    sample_error_with_detail(
                        builtin,
                        &SAMPLE_ERROR_GATHER_FAILED,
                        "provider returned an incompatible linear-gather result",
                    ),
                ))
            }
        }
        Err(source) => {
            let error = sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_GATHER_FAILED,
                "linear gather failed without certified pre-effect status",
            );
            Err(crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects(),
            ))
        }
    }
}

pub(super) fn resolved_actual_downsample_owner(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    let provider = gpu_helpers::exact_provider_for_handle(handle)?;
    handle.view().validate(handle.physical()).ok()?;
    let contract = gpu_helpers::UnaryGpuOutputContract {
        storage: runmat_accelerate_api::handle_storage(handle),
        precision: runmat_accelerate_api::handle_precision(handle),
        integer: runmat_accelerate_api::handle_integer_type(handle),
        logical: runmat_accelerate_api::handle_is_logical(handle),
        alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
    };
    gpu_helpers::unary_gpu_output_matches_shape(handle, handle, provider, handle.shape(), contract)
        .then_some(provider)
}

pub(super) fn wrap_sample_rate_gpu(
    source: &runmat_accelerate_api::GpuTensorHandle,
    output: runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_is_logical(source) {
        gpu_helpers::logical_gpu_value(output)
    } else {
        gpu_helpers::resident_gpu_value(output)
    }
}

pub(super) fn wrap_resampled_gpu(
    output: runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_is_logical(&output) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            build_runtime_error(
                "resampling provider returned logical storage instead of numeric output",
            )
            .build(),
        ));
    }
    gpu_helpers::resident_gpu_value(output)
}

pub(super) fn valid_class_preserving_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    source: &runmat_accelerate_api::GpuTensorHandle,
    output: &runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    require_distinct: bool,
) -> bool {
    let alias = if require_distinct {
        gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation
    } else {
        gpu_helpers::GpuOutputAliasPolicy::AllowInput
    };
    gpu_helpers::unary_gpu_output_matches_shape(
        output,
        source,
        provider,
        expected_shape,
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::handle_storage(source),
            precision: runmat_accelerate_api::handle_precision(source),
            integer: runmat_accelerate_api::handle_integer_type(source),
            logical: runmat_accelerate_api::handle_is_logical(source),
            alias,
        },
    ) && gpu_helpers::numeric_descriptor_matches_source(source, output)
}

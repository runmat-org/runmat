use super::*;

pub(super) async fn gradient_gpu_outputs(
    handle: GpuTensorHandle,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
    requires_host_gpu: bool,
    call: &super::super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<Vec<Value>> {
    let owner = Some(gpu_helpers::require_resident_compute_owner(
        &handle,
        || gradient_internal_error("gradient: resident input has no exact owning provider"),
    )?);
    let complex_storage =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;

    if requires_host_gpu
        || all_spacings
            .iter()
            .any(|spacing| matches!(spacing, GradientSpacing::Coordinates(_)))
    {
        return gradient_host_fallback(&handle, owner, requested_dims, all_spacings, call).await;
    }

    if let Some(provider) = owner {
        if runmat_accelerate_api::handle_precision(&handle) != Some(provider.precision()) {
            return gradient_host_fallback(&handle, owner, requested_dims, all_spacings, call)
                .await;
        }
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let mut outputs = Vec::with_capacity(requested_dims.len());
        for &dim in requested_dims {
            let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
            check_gradient_outputs(call, &mut outputs)?;
            let device_result = match spacing {
                GradientSpacing::Scalar(spacing) => {
                    provider.gradient_dim(&handle, dim.saturating_sub(1), *spacing)
                }
                GradientSpacing::Coordinates(_) => unreachable!("coordinates use host fallback"),
            };
            let device_result = call.completed_handle(device_result.map_err(|source| {
                kernel_error::with_provider_source(
                    gradient_internal_error("provider execution failed"),
                    source.with_unproven_effects(),
                )
            }));
            match device_result {
                Ok(Some(mut device_result))
                    if valid_gradient_provider_result(
                        &device_result,
                        &handle,
                        &outputs,
                        provider,
                        complex_storage,
                    ) =>
                {
                    let adoption =
                        gpu_helpers::propagate_output_provenance(&mut device_result, [&handle])
                            .and_then(|()| {
                                device_result
                                    .allocation_lease()
                                    .validate_live()
                                    .map_err(|source| {
                                        crate::build_runtime_error(format!(
                                            "gradient: cannot adopt provider output: {source}"
                                        ))
                                        .with_source(source)
                                        .build()
                                    })
                            });
                    if let Err(error) = adoption {
                        let error = gpu_helpers::reject_gpu_output(device_result, error);
                        return Err(reject_gradient_outputs(outputs, error));
                    }
                    runmat_accelerate_api::mark_residency(&device_result);
                    outputs.push(Value::GpuTensor(device_result));
                }
                Ok(Some(device_result)) => {
                    let mut error = gradient_internal_error(
                        "provider returned malformed or aliased gradient output",
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    let error = gpu_helpers::reject_gpu_output(device_result, error);
                    return Err(reject_gradient_outputs(outputs, error));
                }
                Ok(None) => {
                    if !outputs.is_empty() {
                        let mut failure = gradient_internal_error(
                            "provider declined gradient after producing an earlier output",
                        );

                        failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        return Err(reject_gradient_outputs(outputs, failure));
                    }
                    return gradient_host_fallback(
                        &handle,
                        owner,
                        requested_dims,
                        all_spacings,
                        call,
                    )
                    .await;
                }
                Err(error) => return Err(reject_gradient_outputs(outputs, error)),
            }
        }
        return Ok(outputs);
    }

    gradient_host_fallback(&handle, None, requested_dims, all_spacings, call).await
}

pub(super) async fn gradient_host_fallback(
    handle: &GpuTensorHandle,
    owner: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
    call: &super::super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<Vec<Value>> {
    call.fallback([&Value::GpuTensor(handle.clone())])?;
    let owner = owner
        .ok_or_else(|| gradient_internal_error("gradient input has no exact owning provider"))?;
    let gathered = gpu_helpers::download_value_preserving_residency_async(owner, handle).await?;
    call.check()?;
    let outputs = evaluate_host_gradient_outputs(gathered, requested_dims, all_spacings)?;
    restore_gradient_outputs_if_supported(Some(owner), handle, outputs)
}

pub(super) fn gradient_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.shares_allocation_with(rhs)
}

pub(super) fn gradient_output_handle(value: &Value) -> Option<&GpuTensorHandle> {
    match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }
}

pub(super) fn reject_gradient_outputs(outputs: Vec<Value>, error: RuntimeError) -> RuntimeError {
    gpu_helpers::reject_gpu_outputs(
        outputs.into_iter().filter_map(|output| match output {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        }),
        error,
    )
}

pub(super) fn check_gradient_outputs(
    call: &super::super::numerical_execution::NumericalCall<'_>,
    outputs: &mut Vec<Value>,
) -> BuiltinResult<()> {
    call.check()
        .map_err(|error| reject_gradient_outputs(std::mem::take(outputs), error))
}

pub(super) fn valid_gradient_provider_result(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    prior_outputs: &[Value],
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    complex: bool,
) -> bool {
    let expected_storage = if complex {
        GpuTensorStorage::ComplexInterleaved
    } else {
        GpuTensorStorage::Real
    };
    output.shape() == matlab_gradient_shape(input.shape(), product(input.shape()))
        && output.device_id() == input.device_id()
        && !gradient_handles_alias(output, input)
        && prior_outputs
            .iter()
            .filter_map(gradient_output_handle)
            .all(|prior| !gradient_handles_alias(output, prior))
        && runmat_accelerate_api::handle_storage(output) == expected_storage
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

pub(super) fn restore_gradient_outputs_if_supported(
    provider: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    prototype: &GpuTensorHandle,
    outputs: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    if provider.is_none() {
        return Ok(outputs);
    }
    let mut restored = Vec::with_capacity(outputs.len());
    for output in outputs {
        let protected = restored
            .iter()
            .filter_map(gradient_output_handle)
            .collect::<Vec<_>>();
        let output = match gpu_helpers::restore_class_preserving_value_protecting(
            prototype, output, NAME, &protected,
        ) {
            Ok(output) => output,
            Err(error) => return Err(reject_gradient_outputs(restored, error)),
        };
        restored.push(output);
    }
    Ok(restored)
}

pub(super) fn spacing_for_dim<'a>(
    dim: usize,
    available_dims: &[usize],
    spacings: &'a [GradientSpacing],
) -> &'a GradientSpacing {
    let index = available_dims
        .iter()
        .position(|candidate| *candidate == dim)
        .expect("spacing lookup requires matching dimension");
    &spacings[index]
}

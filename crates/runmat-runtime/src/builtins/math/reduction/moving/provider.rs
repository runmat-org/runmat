use super::*;
use super::{
    admission::{terminal, MovingCall},
    provider_shape::ProviderShape,
};

pub(super) async fn moving_gpu(
    work: &MovingCall<'_>,
    name: &'static str,
    op: MovingOp,
    handle: GpuTensorHandle,
    parsed: &ParsedMoving,
) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_owner(&handle, || {
        invalid_input(
            name,
            format!("{name}: resident input has no exact owning provider"),
        )
    })?;
    if let Some(output) = moving_gpu_provider(work, name, op, &handle, parsed).await? {
        return gpu_helpers::resident_gpu_value(output).map_err(terminal);
    }
    // Only the existing typed no-attempt/unsupported route reaches here. The
    // finite host scratch gap is not permission to replay after a provider error.
    if runmat_accelerate_api::handle_is_explicit(&handle) {
        return Err(terminal(invalid_input(name, format!("{name}: this form requires host fallback; explicitly gather the input before calling it"))));
    }
    work.host()?;
    let tensor = work.wait(gpu_helpers::gather_tensor_async(&handle)).await?;
    work.host()?;
    let result = moving_real(name, op, tensor, parsed)?;
    work.host()?;
    gpu_helpers::restore_class_preserving_value(&handle, tensor::tensor_into_value(result), name)
        .map_err(terminal)
}

async fn moving_gpu_provider(
    work: &MovingCall<'_>,
    name: &'static str,
    op: MovingOp,
    handle: &GpuTensorHandle,
    parsed: &ParsedMoving,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    work.check()?;
    let Some(provider_op) = op.provider_op() else {
        return Ok(None);
    };
    if parsed.sample_points.is_some() {
        return Ok(None);
    }
    let Some((before, after)) = parsed.window.count_radius() else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Ok(None);
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) else {
        return Ok(None);
    };
    // These existing hooks compute in the provider's real floating precision.
    // Select documented native-integer/logical or mixed-precision host work
    // before the hook can read/allocate anything, never after a lossy attempt.
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_precision(handle) != Some(provider.precision())
    {
        return Ok(None);
    }
    let dim = parsed
        .dim
        .unwrap_or_else(|| default_dimension(handle.shape()));
    if dim == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: dimension must be >= 1"),
        ));
    }
    let output_shape = ProviderShape::new(work, name, handle.shape(), dim, parsed)?;
    checked_element_count(name, output_shape.as_slice())?;
    let axis_len = handle.shape().get(dim - 1).copied().unwrap_or(1);
    validate_sample_points_len(name, axis_len, parsed)?;
    let request = ProviderMovingWindowRequest {
        input: handle,
        output_shape: output_shape.as_slice(),
        dim: dim - 1,
        before,
        after,
        op: provider_op,
        endpoints: provider_endpoints(op, parsed.endpoints),
        nan_mode: provider_nan_mode(parsed.nan_mode),
        normalization: provider_normalization(parsed.normalization),
    };
    output_shape.check(work)?;
    match provider.moving_window(&request).await {
        Ok(output) => {
            if let Err(error) = output_shape.check(work) {
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            let source_integer = runmat_accelerate_api::handle_integer_type(handle);
            let expected_integer = if matches!(op, MovingOp::Min | MovingOp::Max | MovingOp::Median)
            {
                source_integer
            } else {
                None
            };
            let expected_precision = if (source_integer.is_some() && expected_integer.is_none())
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            } else {
                runmat_accelerate_api::handle_precision(handle)
            };
            let expected_element = expected_integer
                .map(runmat_accelerate_api::NumericElementType::from)
                .or_else(|| {
                    expected_precision.map(|precision| match precision {
                        runmat_accelerate_api::ProviderPrecision::F32 => {
                            runmat_accelerate_api::NumericElementType::F32
                        }
                        runmat_accelerate_api::ProviderPrecision::F64 => {
                            runmat_accelerate_api::NumericElementType::F64
                        }
                    })
                });
            let valid = output.shape() == output_shape.as_slice()
                && output.device_id() == handle.device_id()
                && runmat_accelerate_api::provider_for_handle(&output).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
                && runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::Real
                && output.physical().storage() == GpuTensorStorage::Real
                && Some(output.physical().element_type()) == expected_element
                && runmat_accelerate_api::handle_integer_type(&output) == expected_integer
                && runmat_accelerate_api::handle_precision(&output) == expected_precision
                && !runmat_accelerate_api::handle_is_logical(&output)
                && gpu_helpers::gpu_class_metadata_matches(
                    &output,
                    expected_precision,
                    expected_integer,
                    false,
                );
            if !valid {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    terminal(internal(
                        name,
                        format!("{name}: provider returned a malformed moving-window result"),
                    )),
                ));
            }
            let output = gpu_helpers::propagate_owned_output_provenance(output, [handle])
                .map_err(terminal)?;
            if let Err(error) = output_shape.check(work) {
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            Ok(Some(output))
        }
        Err(source) if gpu_helpers::provider_hook_allows_host_fallback(&source) => {
            output_shape.check(work)?;
            Ok(None)
        }
        Err(source) => {
            let error = internal(name, format!("{name}: provider execution failed: {source}"));
            Err(terminal(
                crate::runtime_error::with_provider_execution_source(error, source.into()),
            ))
        }
    }
}

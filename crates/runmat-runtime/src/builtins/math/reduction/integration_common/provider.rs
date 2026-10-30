//! One exact-owner provider attempt shared by final and cumulative integration.
use super::super::numerical_execution::{completed, NumericalCall};
use super::*;
use runmat_accelerate_api::{handle_integer_type, handle_is_logical, handle_precision};
use runmat_builtins::NumericalReductionKind;

pub(in crate::builtins::math::reduction) async fn try_provider_integration(
    args: &IntegrationArguments<'_>,
    call: &NumericalCall<'_>,
) -> BuiltinResult<Option<Value>> {
    gpu_helpers::require_resident_compute_values(
        &[args.y, args.spacing.unwrap_or(args.y)],
        false,
        || {
            super::super::numerical_execution::error(
                call.kind,
                "sampled data has no exact owning provider",
            )
        },
    )?;
    let Value::GpuTensor(input) = args.y else {
        return Ok(None);
    };
    let provider = gpu_helpers::require_resident_compute_owner(input, || {
        super::super::numerical_execution::error(
            call.kind,
            "sampled data has no exact owning provider",
        )
    })?;
    input.allocation_lease().validate_live().map_err(|source| {
        crate::build_runtime_error("integration input allocation is unavailable")
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    if handle_integer_type(input).is_some()
        || handle_is_logical(input)
        || handle_precision(input) != Some(provider.precision())
    {
        return Ok(None);
    }
    if args.spacing.is_some_and(|value| matches!(value,Value::Int(_)|Value::Bool(_)|Value::LogicalArray(_))
        || matches!(value,Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value,Value::GpuTensor(h) if handle_integer_type(h).is_some()||handle_is_logical(h)
            ||gpu_helpers::exact_provider_for_handle(h).is_none_or(|owner|!runmat_accelerate_api::same_provider_instance(owner,provider)))) {
        return Ok(None);
    }
    let dimension = args
        .dim
        .unwrap_or_else(|| default_dimension_from_shape(input.shape()));
    let (spacing, staged) = spacing(args.spacing, input, dimension, provider, call)?;
    let result = (|| {
        if staged
            && spacing
                .handle()
                .is_some_and(|handle| handle.shares_allocation_with(input))
        {
            return Err(super::super::numerical_execution::error(
                call.kind,
                "integration spacing aliases sampled data",
            ));
        }
        call.check()?;
        let pending = match call.kind {
            NumericalReductionKind::Trapezoid => {
                provider.trapz_dim(input, dimension - 1, spacing.as_provider_spacing())
            }
            NumericalReductionKind::CumulativeTrapezoid => {
                provider.cumtrapz_dim(input, dimension - 1, spacing.as_provider_spacing())
            }
            _ => unreachable!("only integration operations enter the integration owner"),
        };
        let output = call.completed_handle(completed(call.kind, pending))?;
        let Some(output) = output else {
            if staged {
                return Err(super::super::numerical_execution::error(
                    call.kind,
                    "integration declined after spacing upload",
                ));
            }
            return Ok(None);
        };
        let cumulative = call.kind == NumericalReductionKind::CumulativeTrapezoid;
        let shape = output.shape().len() == input.shape().len()
            && output.shape().iter().enumerate().all(|(axis, &extent)| {
                extent
                    == if !cumulative && axis == dimension - 1 {
                        1
                    } else {
                        input.shape()[axis]
                    }
            });
        let expected_precision = handle_precision(input);
        let valid = shape
            && !output.shares_allocation_with(input)
            && spacing
                .handle()
                .is_none_or(|handle| !output.shares_allocation_with(handle))
            && output.device_id() == input.device_id()
            && runmat_accelerate_api::handle_storage(&output)
                == runmat_accelerate_api::handle_storage(input)
            && handle_precision(&output) == expected_precision
            && handle_integer_type(&output).is_none()
            && !handle_is_logical(&output)
            && gpu_helpers::gpu_class_metadata_matches(&output, expected_precision, None, false)
            && gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            });
        if !valid {
            return Err(gpu_helpers::reject_gpu_output(output,super::super::numerical_execution::error(call.kind,"provider returned malformed, aliased, foreign, or wrong-class integration output")));
        }
        let protected = [
            Some(input),
            args.spacing.and_then(|value| {
                if let Value::GpuTensor(handle) = value {
                    Some(handle)
                } else {
                    None
                }
            }),
        ];
        let output = gpu_helpers::propagate_owned_output_provenance(
            output,
            protected.into_iter().flatten(),
        )?;
        gpu_helpers::resident_gpu_value(output).map(Some)
    })();
    if staged {
        if let Some(handle) = spacing.into_handle() {
            return match result {
                Err(error) => Err(gpu_helpers::reject_gpu_output(handle, error)),
                Ok(value) => match handle.release() {
                    Ok(_) => Ok(value),
                    Err(source) => {
                        let error =
                            crate::build_runtime_error("integration spacing retirement failed")
                                .with_source(source)
                                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                                .build();
                        match value {
                            Some(Value::GpuTensor(output)) => {
                                Err(gpu_helpers::reject_gpu_output(output, error))
                            }
                            _ => Err(error),
                        }
                    }
                },
            };
        }
    }
    result
}

fn spacing(
    value: Option<&Value>,
    input: &GpuTensorHandle,
    dimension: usize,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    call: &NumericalCall<'_>,
) -> BuiltinResult<(GpuIntegrationSpacing, bool)> {
    let Some(value) = value else {
        return Ok((GpuIntegrationSpacing::Unit, false));
    };
    if is_empty_value(value) {
        return Ok((GpuIntegrationSpacing::Unit, false));
    }
    let scalar = match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor.len() == 1 && tensor.integer_storage().is_none() => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    };
    if let Some(scalar) = scalar {
        return Ok((GpuIntegrationSpacing::Scalar(scalar), false));
    }
    // The existing coordinate staging owner allocates host vector/query
    // backing; do not invent finite grants for it in this consumer.
    call.host()?;
    let staged = !matches!(value, Value::GpuTensor(_));
    let result = spacing_from_gpu_or_host_value_for_provider(
        call.kind.catalog_entry().identity.name,
        Some(value.clone()),
        input.shape(),
        dimension,
        provider,
    )
    .map_err(super::super::numerical_execution::terminal)?;
    // Shape/owner validation belongs to the shared existing staging boundary.
    // Any staging error is terminal; no error-string or Unsupported replay.
    Ok((result, staged))
}

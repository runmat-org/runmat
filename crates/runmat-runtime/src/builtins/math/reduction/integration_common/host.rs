//! Admitted host fallback, never implicit permission to gather explicit inputs.
use super::super::numerical_execution::NumericalCall;
use super::arguments::input_error;
use super::*;
pub(in crate::builtins::math::reduction) async fn gather_integration_inputs(
    args: &IntegrationArguments<'_>,
    call: &NumericalCall<'_>,
) -> BuiltinResult<(Value, Option<Value>)> {
    if value_has_gpu_tensor(args.y) || args.spacing.is_some_and(value_has_gpu_tensor) {
        call.fallback(std::iter::once(args.y).chain(args.spacing))?;
    } else {
        call.host()?;
    }
    let y = gather(args.y, call).await?;
    let spacing = match args.spacing {
        Some(value) => Some(gather(value, call).await?),
        None => None,
    };
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(&y)
        || spacing.as_ref().is_some_and(|value| {
            !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
        })
    {
        return Err(input_error(
            call.kind,
            "integer sample data and spacing must be exactly representable as double",
        ));
    }
    Ok((y, spacing))
}
async fn gather(value: &Value, call: &NumericalCall<'_>) -> BuiltinResult<Value> {
    call.check()?;
    let result = if let Value::GpuTensor(handle) = value {
        let readback =
            runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                let error = super::super::numerical_execution::error(
                    call.kind,
                    "integration input has no exact owning provider",
                );
                crate::runtime_error::with_provider_execution_source(error, source.into())
            })?;
        gpu_helpers::download_readback_value_in_context(call.runtime, &readback)
            .await
            .map_err(super::super::numerical_execution::terminal)?
    } else {
        value.clone()
    };
    call.check()?;
    Ok(result)
}
pub(in crate::builtins::math::reduction) fn restore_integration_result(
    value: Value,
    args: &IntegrationArguments<'_>,
    call: &NumericalCall<'_>,
) -> BuiltinResult<Value> {
    let protected = std::iter::once(args.y)
        .chain(args.spacing)
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let Some(owner) = protected.first() else {
        return call.finish_value(Ok(value));
    };
    call.check()?;
    let value = gpu_helpers::restore_class_preserving_value_protecting(
        owner,
        value,
        call.kind.catalog_entry().identity.name,
        &protected,
    )?;
    call.finish_value(Ok(value))
}

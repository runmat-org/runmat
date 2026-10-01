//! Resident RMSE scales both operands before subtraction. Forming a raw
//! residual first can overflow even when the final root mean square is finite.
use super::{
    contract_failure, equivalent_shape, expected_shape, numerics, provider_failure, reduction,
    usable_scale, validate_output, Failure,
};
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers};
use crate::builtins::stats::summary::statistical_means::options::{
    invalid, Options, ReductionKind,
};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::Value;

async fn absolute_components(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    precision: ProviderPrecision,
) -> Result<(GpuTensorHandle, Option<GpuTensorHandle>), Failure> {
    let kind = ReductionKind::RootMeanSquaredError;
    let (real, imaginary) = match runmat_accelerate_api::handle_storage(input) {
        GpuTensorStorage::Real => (input.clone(), None),
        GpuTensorStorage::ComplexInterleaved => {
            let real = provider
                .unary_real(input)
                .await
                .map_err(|e| provider_failure(kind, e))?;
            let imaginary = provider
                .unary_imag(input)
                .await
                .map_err(|e| provider_failure(kind, e))?;
            validate_output(kind, provider, input, &real, input.shape(), precision)?;
            validate_output(kind, provider, input, &imaginary, input.shape(), precision)?;
            (real, Some(imaginary))
        }
        _ => return Err(Failure::Unsupported),
    };
    let absolute_real = provider
        .unary_abs(&real)
        .await
        .map_err(|e| provider_failure(kind, e))?;
    validate_output(
        kind,
        provider,
        input,
        &absolute_real,
        input.shape(),
        precision,
    )?;
    let absolute = if let Some(imaginary) = &imaginary {
        let absolute_imaginary = provider
            .unary_abs(imaginary)
            .await
            .map_err(|e| provider_failure(kind, e))?;
        validate_output(
            kind,
            provider,
            input,
            &absolute_imaginary,
            input.shape(),
            precision,
        )?;
        provider
            .elem_max(&absolute_real, &absolute_imaginary)
            .await
            .map_err(|e| provider_failure(kind, e))?
    } else {
        absolute_real
    };
    validate_output(kind, provider, input, &absolute, input.shape(), precision)?;
    Ok((absolute, imaginary))
}

async fn normalized_component(
    provider: &'static dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    scale: &GpuTensorHandle,
    output_shape: &[usize],
    precision: ProviderPrecision,
) -> Result<GpuTensorHandle, Failure> {
    let kind = ReductionKind::RootMeanSquaredError;
    let left_normalized = provider
        .elem_div(left, scale)
        .await
        .map_err(|e| provider_failure(kind, e))?;
    let right_normalized = provider
        .elem_div(right, scale)
        .await
        .map_err(|e| provider_failure(kind, e))?;
    let left_shape = BroadcastPlan::new(left.shape(), scale.shape())
        .map_err(|source| contract_failure(kind, source.to_string()))?;
    let right_shape = BroadcastPlan::new(right.shape(), scale.shape())
        .map_err(|source| contract_failure(kind, source.to_string()))?;
    validate_output(
        kind,
        provider,
        left,
        &left_normalized,
        left_shape.output_shape(),
        precision,
    )?;
    validate_output(
        kind,
        provider,
        left,
        &right_normalized,
        right_shape.output_shape(),
        precision,
    )?;
    let residual = provider
        .elem_sub(&left_normalized, &right_normalized)
        .await
        .map_err(|e| provider_failure(kind, e))?;
    validate_output(kind, provider, left, &residual, output_shape, precision)?;
    Ok(residual)
}

async fn run(
    provider: &'static dyn AccelProvider,
    forecast: &GpuTensorHandle,
    actual: &GpuTensorHandle,
    options: &Options,
    precision: ProviderPrecision,
    plan: &BroadcastPlan,
) -> Result<GpuTensorHandle, Failure> {
    let kind = ReductionKind::RootMeanSquaredError;
    let output_shape = plan.output_shape();
    let (forecast_abs, forecast_imaginary) =
        absolute_components(provider, forecast, precision).await?;
    let (actual_abs, actual_imaginary) = absolute_components(provider, actual, precision).await?;
    let mut scale = provider
        .elem_max(&forecast_abs, &actual_abs)
        .await
        .map_err(|e| provider_failure(kind, e))?;
    validate_output(kind, provider, forecast, &scale, output_shape, precision)?;
    for axis in reduction::selected_axes(output_shape, &options.axes) {
        scale = provider
            .reduce_max_dim(&scale, axis)
            .await
            .map_err(|e| provider_failure(kind, e))?
            .values;
    }
    let scale = usable_scale(kind, provider, forecast, scale).await?;
    validate_output(
        kind,
        provider,
        forecast,
        &scale,
        &expected_shape(output_shape, &options.axes),
        precision,
    )?;

    let forecast_real = if forecast_imaginary.is_some() {
        provider
            .unary_real(forecast)
            .await
            .map_err(|e| provider_failure(kind, e))?
    } else {
        forecast.clone()
    };
    let actual_real = if actual_imaginary.is_some() {
        provider
            .unary_real(actual)
            .await
            .map_err(|e| provider_failure(kind, e))?
    } else {
        actual.clone()
    };
    let real = normalized_component(
        provider,
        &forecast_real,
        &actual_real,
        &scale,
        output_shape,
        precision,
    )
    .await?;
    let normalized = if let (Some(forecast_imaginary), Some(actual_imaginary)) =
        (&forecast_imaginary, &actual_imaginary)
    {
        let imaginary = normalized_component(
            provider,
            forecast_imaginary,
            actual_imaginary,
            &scale,
            output_shape,
            precision,
        )
        .await?;
        let complex = provider
            .complex_from_real_imag(&real, &imaginary)
            .await
            .map_err(|e| provider_failure(kind, e))?;
        if !equivalent_shape(complex.shape(), output_shape)
            || runmat_accelerate_api::handle_storage(&complex)
                != GpuTensorStorage::ComplexInterleaved
            || runmat_accelerate_api::handle_precision(&complex) != Some(precision)
            || complex.device_id() != forecast.device_id()
            || gpu_helpers::exact_provider_for_handle(&complex)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            return Err(contract_failure(
                kind,
                "resident complex residual has an invalid class, shape, or owner",
            ));
        }
        complex
    } else {
        real
    };
    let rooted = numerics::run(kind, provider, &normalized, options).await?;
    validate_output(
        kind,
        provider,
        forecast,
        &rooted,
        &expected_shape(output_shape, &options.axes),
        precision,
    )?;
    let result = provider
        .elem_mul(&rooted, &scale)
        .await
        .map_err(|e| provider_failure(kind, e))?;
    validate_output(
        kind,
        provider,
        forecast,
        &result,
        &expected_shape(output_shape, &options.axes),
        precision,
    )?;
    Ok(result)
}

pub(super) async fn try_rmse(
    forecast: &GpuTensorHandle,
    actual: &GpuTensorHandle,
    options: &Options,
) -> BuiltinResult<Option<Value>> {
    let kind = ReductionKind::RootMeanSquaredError;
    let provider = gpu_helpers::exact_provider_for_binary_inputs(forecast, actual)
        .map_err(|source| invalid(kind, &source))?;
    let precision = runmat_accelerate_api::handle_precision(forecast);
    let storage = runmat_accelerate_api::handle_storage(forecast);
    let supported = options.weights.is_none()
        && precision.is_some()
        && precision == runmat_accelerate_api::handle_precision(actual)
        && storage == runmat_accelerate_api::handle_storage(actual)
        && matches!(
            storage,
            GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
        )
        && runmat_accelerate_api::handle_integer_type(forecast).is_none()
        && runmat_accelerate_api::handle_integer_type(actual).is_none()
        && !runmat_accelerate_api::handle_is_logical(forecast)
        && !runmat_accelerate_api::handle_is_logical(actual)
        && !forecast.shape().contains(&0)
        && !actual.shape().contains(&0);
    if !supported {
        if runmat_accelerate_api::handle_is_explicit(forecast)
            || runmat_accelerate_api::handle_is_explicit(actual)
        {
            return Err(invalid(
                kind,
                "explicit GPU RMSE requires a supported resident precision and weight route",
            ));
        }
        return Ok(None);
    }
    let plan = BroadcastPlan::new(forecast.shape(), actual.shape())
        .map_err(|source| invalid(kind, &source.to_string()))?;
    let outcome = run(
        provider,
        forecast,
        actual,
        options,
        precision.expect("checked above"),
        &plan,
    )
    .await;
    match outcome {
        Ok(mut output) => {
            if output.shares_allocation_with(forecast) || output.shares_allocation_with(actual) {
                let error = invalid(kind, "resident RMSE aliased an input allocation");
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            gpu_helpers::propagate_output_provenance(&mut output, [forecast, actual])?;
            gpu_helpers::resident_gpu_value(output).map(Some)
        }
        Err(Failure::Unsupported)
            if !runmat_accelerate_api::handle_is_explicit(forecast)
                && !runmat_accelerate_api::handle_is_explicit(actual) =>
        {
            Ok(None)
        }
        Err(Failure::Unsupported) => Err(invalid(
            kind,
            "explicit GPU RMSE requires a supported resident provider route",
        )),
        Err(Failure::Execution(error)) => Err(*error),
    }
}

//! Resident RMSE scales both operands before subtraction. Forming a raw
//! residual first can overflow even when the final root mean square is finite.
use super::{
    admitted_real, contract_failure, equivalent_shape, expected_shape, numerics, reduction,
    reject_output, usable_scale, Failure, ObserveResidentHook, ResidentStageProgress,
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
    progress: &mut ResidentStageProgress,
) -> Result<(GpuTensorHandle, GpuTensorHandle, Option<GpuTensorHandle>), Failure> {
    let kind = ReductionKind::RootMeanSquaredError;
    let (real, imaginary) = match runmat_accelerate_api::handle_storage(input) {
        GpuTensorStorage::Real => (input.clone(), None),
        GpuTensorStorage::ComplexInterleaved => {
            let real = provider.unary_real(input).await.observe(progress, kind)?;
            let imaginary = provider.unary_imag(input).await.observe(progress, kind)?;
            let real = admitted_real(kind, provider, input, real, input.shape(), precision)?;
            let imaginary =
                admitted_real(kind, provider, input, imaginary, input.shape(), precision)?;
            (real, Some(imaginary))
        }
    };
    let absolute_real = provider.unary_abs(&real).await.observe(progress, kind)?;
    let absolute_real = admitted_real(
        kind,
        provider,
        input,
        absolute_real,
        input.shape(),
        precision,
    )?;
    let absolute = if let Some(imaginary) = &imaginary {
        let absolute_imaginary = provider
            .unary_abs(imaginary)
            .await
            .observe(progress, kind)?;
        let absolute_imaginary = admitted_real(
            kind,
            provider,
            input,
            absolute_imaginary,
            input.shape(),
            precision,
        )?;
        provider
            .elem_max(&absolute_real, &absolute_imaginary)
            .await
            .observe(progress, kind)?
    } else {
        absolute_real
    };
    let absolute = admitted_real(kind, provider, input, absolute, input.shape(), precision)?;
    Ok((absolute, real, imaginary))
}

async fn normalized_component(
    provider: &'static dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    scale: &GpuTensorHandle,
    output_shape: &[usize],
    precision: ProviderPrecision,
    progress: &mut ResidentStageProgress,
) -> Result<GpuTensorHandle, Failure> {
    let kind = ReductionKind::RootMeanSquaredError;
    let left_normalized = provider
        .elem_div(left, scale)
        .await
        .observe(progress, kind)?;
    let right_normalized = provider
        .elem_div(right, scale)
        .await
        .observe(progress, kind)?;
    let left_shape = BroadcastPlan::new(left.shape(), scale.shape())
        .map_err(|source| contract_failure(kind, source.to_string()))?;
    let right_shape = BroadcastPlan::new(right.shape(), scale.shape())
        .map_err(|source| contract_failure(kind, source.to_string()))?;
    let left_normalized = admitted_real(
        kind,
        provider,
        left,
        left_normalized,
        left_shape.output_shape(),
        precision,
    )?;
    let right_normalized = admitted_real(
        kind,
        provider,
        left,
        right_normalized,
        right_shape.output_shape(),
        precision,
    )?;
    let residual = provider
        .elem_sub(&left_normalized, &right_normalized)
        .await
        .observe(progress, kind)?;
    admitted_real(kind, provider, left, residual, output_shape, precision)
}

async fn run(
    provider: &'static dyn AccelProvider,
    forecast: &GpuTensorHandle,
    actual: &GpuTensorHandle,
    options: &Options,
    precision: ProviderPrecision,
    plan: &BroadcastPlan,
    progress: &mut ResidentStageProgress,
) -> Result<GpuTensorHandle, Failure> {
    let kind = ReductionKind::RootMeanSquaredError;
    let output_shape = plan.output_shape();
    let (forecast_abs, forecast_real, forecast_imaginary) =
        absolute_components(provider, forecast, precision, progress).await?;
    let (actual_abs, actual_real, actual_imaginary) =
        absolute_components(provider, actual, precision, progress).await?;
    let scale = provider
        .elem_max(&forecast_abs, &actual_abs)
        .await
        .observe(progress, kind)?;
    let mut scale = admitted_real(kind, provider, forecast, scale, output_shape, precision)?;
    if options.missing == super::super::options::MissingMode::Omit {
        scale = provider.map_nan_to_zero(&scale).observe(progress, kind)?;
        scale = admitted_real(kind, provider, forecast, scale, output_shape, precision)?;
    }
    for axis in reduction::selected_axes(output_shape, &options.axes) {
        let next = provider
            .reduce_max_dim(&scale, axis)
            .await
            .observe(progress, kind)?
            .values;
        let expected = expected_shape(
            scale.shape(),
            &super::super::options::Axes::Dims(vec![axis + 1]),
        );
        scale = admitted_real(kind, provider, forecast, next, &expected, precision)?;
    }
    let protected = [
        actual,
        &forecast_abs,
        &actual_abs,
        &forecast_real,
        &actual_real,
        forecast_imaginary.as_ref().unwrap_or(forecast),
        actual_imaginary.as_ref().unwrap_or(actual),
    ];
    let scale = usable_scale(kind, provider, forecast, scale, &protected, progress).await?;
    let scale = admitted_real(
        kind,
        provider,
        forecast,
        scale,
        &expected_shape(output_shape, &options.axes),
        precision,
    )?;
    let real = normalized_component(
        provider,
        &forecast_real,
        &actual_real,
        &scale,
        output_shape,
        precision,
        progress,
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
            progress,
        )
        .await?;
        let complex = provider
            .complex_from_real_imag(&real, &imaginary)
            .await
            .observe(progress, kind)?;
        if !equivalent_shape(complex.shape(), output_shape)
            || complex.allocation_lease().validate_live().is_err()
            || runmat_accelerate_api::handle_storage(&complex)
                != GpuTensorStorage::ComplexInterleaved
            || runmat_accelerate_api::handle_precision(&complex) != Some(precision)
            || complex.device_id() != forecast.device_id()
            || gpu_helpers::exact_provider_for_handle(&complex)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            return Err(reject_output(
                kind,
                complex,
                "resident complex residual has an invalid class, shape, or owner",
            ));
        }
        complex
    } else {
        real
    };
    let rooted = numerics::run(kind, provider, &normalized, options, progress).await?;
    let rooted = admitted_real(
        kind,
        provider,
        forecast,
        rooted,
        &expected_shape(output_shape, &options.axes),
        precision,
    )?;
    let result = provider
        .elem_mul(&rooted, &scale)
        .await
        .observe(progress, kind)?;
    admitted_real(
        kind,
        provider,
        forecast,
        result,
        &expected_shape(output_shape, &options.axes),
        precision,
    )
}

pub(in crate::builtins::stats::summary::statistical_means) async fn try_rmse(
    forecast: &GpuTensorHandle,
    actual: &GpuTensorHandle,
    options: &Options,
) -> BuiltinResult<Option<Value>> {
    let kind = ReductionKind::RootMeanSquaredError;
    if forecast.allocation_lease().validate_live().is_err()
        || actual.allocation_lease().validate_live().is_err()
    {
        return Err(invalid(kind, "resident RMSE input allocation is not live"));
    }
    let Ok(provider) = gpu_helpers::exact_provider_for_binary_inputs(forecast, actual) else {
        return Ok(None);
    };
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
        return Ok(None);
    }
    let plan = BroadcastPlan::new(forecast.shape(), actual.shape())
        .map_err(|source| invalid(kind, &source.to_string()))?;
    let mut progress = ResidentStageProgress::default();
    let outcome = run(
        provider,
        forecast,
        actual,
        options,
        precision.expect("checked above"),
        &plan,
        &mut progress,
    )
    .await;
    match outcome {
        Ok(output) => {
            if output.shares_allocation_with(forecast) || output.shares_allocation_with(actual) {
                let error = invalid(kind, "resident RMSE aliased an input allocation");
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            let output =
                gpu_helpers::propagate_owned_output_provenance(output, [forecast, actual])?;
            gpu_helpers::resident_gpu_value(output).map(Some)
        }
        Err(Failure::Unsupported) => Ok(None),
        Err(Failure::Execution(error)) => Err(*error),
    }
}
